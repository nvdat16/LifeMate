"""
LifeMate Agent StateGraph Orchestrator powered by LangGraph.

Architecture: Dynamic Re-planning with Parallel Worker Pool & Cyclic Feedback:
  START
    ↓
  [Perception Node] (ASR + text normalization)
    ↓
  [Gatekeeper Node] (In-Scope vs Out-of-Scope triage)
    ↓ (conditional: if Out-of-Scope -> Commit Node with empty calls)
    ↓ (conditional: if In-Scope -> Dynamic Planner Node)
  
  ┌─► [Dynamic Planner Node] (Inspects transcript, context & past observations -> emits next parallel batch)
  │     ↓ (conditional: if is_finished or max_iter -> Commit Node)
  │   [Worker Pool Node] (Dispatches parallel independent tasks -> runs Guardrails -> calls Sandbox)
  └─────┘ (Loops back with fresh Observations!)

  [Commit Node] (Extracts final tool calls & state_hash)
    ↓
   END
"""

import json
from typing import Any, Optional, TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END

from ..domain.entities import (
    Sample,
    ExecutionResult,
    Trajectory,
    TrajectoryStep,
    ToolCall,
    Observation,
)
from ..domain.scenario import ScenarioType
from ..ports.asr_port import AbstractASREngine
from ..ports.llm_port import AbstractLLMEngine
from ..ports.sandbox_port import AbstractSandboxEnvFactory
from .agents.perception_agent import PerceptionAgent
from .agents.gatekeeper_agent import GatekeeperAgent
from .agents.dynamic_planner import DynamicPlannerAgent
from .agents.worker_agent import WorkerAgent
from .guardrail.schema_checker import GuardrailInterceptor

class LifeMateGraphState(TypedDict, total=False):
    """Dynamic state object passed through the cyclic StateGraph."""
    sample: Sample
    split: str
    session: Any
    guardrail: GuardrailInterceptor
    trajectory: Trajectory
    step_idx: int
    transcript: str
    scenario: ScenarioType
    is_in_scope: bool
    abstract_tools: List[Dict[str, Any]]
    completed_tasks: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    current_batch: List[Dict[str, Any]]
    is_finished: bool
    iteration: int
    max_iterations: int
    final_tool_calls: List[Dict[str, Any]]
    final_state_hash: str

class LifeMateGraphPipeline:
    """
    LangGraph-based Dynamic Re-planning Pipeline with Parallel Worker Pool.
    Exposes the exact same interface as LifeMatePipeline for drop-in compatibility.
    """

    def __init__(
        self,
        asr_engine: AbstractASREngine,
        llm_engine: AbstractLLMEngine,
        sandbox_factory: AbstractSandboxEnvFactory,
        catalog: dict[str, Any],
        max_planner_iterations: int = 4,
        force_asr: bool = False
    ):
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine
        self.sandbox_factory = sandbox_factory
        self.catalog = catalog
        self.max_planner_iterations = max_planner_iterations
        self.force_asr = force_asr

        # Micro-Agents
        self.perception_agent = PerceptionAgent(self.asr_engine, force_asr=force_asr)
        self.gatekeeper_agent = GatekeeperAgent(self.catalog)
        self.dynamic_planner = DynamicPlannerAgent(self.llm_engine, self.catalog)
        self.worker_agent = WorkerAgent(self.llm_engine, self.catalog)

        # Build and compile LangGraph app
        self.app = self._build_graph()

    def _build_graph(self):
        """Constructs and compiles the dynamic cyclic StateGraph."""
        builder = StateGraph(LifeMateGraphState)

        # 1. Add Nodes
        builder.add_node("perception", self._node_perception)
        builder.add_node("gatekeeper", self._node_gatekeeper)
        builder.add_node("dynamic_planner", self._node_dynamic_planner)
        builder.add_node("worker_pool", self._node_worker_pool)
        builder.add_node("commit", self._node_commit)

        # 2. Add Edges & Conditional Routing
        builder.add_edge(START, "perception")
        builder.add_edge("perception", "gatekeeper")

        # Routing after Gatekeeper (In-Scope vs Out-of-Scope)
        builder.add_conditional_edges(
            "gatekeeper",
            self._route_after_gatekeeper,
            {
                "out_of_scope": "commit",
                "in_scope": "dynamic_planner"
            }
        )

        # Routing after Dynamic Planner (Finished vs Dispatch to Worker Pool)
        builder.add_conditional_edges(
            "dynamic_planner",
            self._route_after_planner,
            {
                "finished": "commit",
                "dispatch": "worker_pool"
            }
        )

        # Worker Pool always loops back to Dynamic Planner with fresh observations!
        builder.add_edge("worker_pool", "dynamic_planner")

        # Commit leads to END
        builder.add_edge("commit", END)

        return builder.compile()

    # --- Node Definitions ---

    def _node_perception(self, state: LifeMateGraphState) -> Dict[str, Any]:
        sample = state["sample"]
        transcript = self.perception_agent.perceive(sample)
        trajectory = Trajectory(
            sample_id=sample.id,
            scenario=ScenarioType.EXECUTE,
            transcript=transcript
        )
        trajectory.add_step(
            TrajectoryStep(
                step_index=1,
                agent_name="PerceptionAgent",
                thought=f"Transcribed audio: '{transcript}'"
            )
        )
        return {
            "transcript": transcript,
            "trajectory": trajectory,
            "step_idx": 1
        }

    def _node_gatekeeper(self, state: LifeMateGraphState) -> Dict[str, Any]:
        transcript = state["transcript"]
        sample = state["sample"]
        trajectory = state["trajectory"]
        step_idx = state["step_idx"] + 1

        scenario = self.gatekeeper_agent.classify_intent(transcript, sample.context)
        trajectory.scenario = scenario
        trajectory.add_step(
            TrajectoryStep(
                step_index=step_idx,
                agent_name="GatekeeperAgent",
                thought=f"Classified scenario as: {scenario.value.upper()}"
            )
        )

        is_in_scope = (scenario != ScenarioType.OUT_OF_SCOPE)
        abstract_tools = self.dynamic_planner.extract_abstract_catalog(sample.context)

        return {
            "scenario": scenario,
            "is_in_scope": is_in_scope,
            "step_idx": step_idx,
            "abstract_tools": abstract_tools,
            "iteration": 0,
            "completed_tasks": [],
            "observations": [],
            "current_batch": [],
            "is_finished": False
        }

    def _node_dynamic_planner(self, state: LifeMateGraphState) -> Dict[str, Any]:
        sample = state["sample"]
        transcript = state["transcript"]
        trajectory = state["trajectory"]
        step_idx = state["step_idx"] + 1
        iteration = state["iteration"]

        # Check iteration boundary
        if iteration >= state.get("max_iterations", 4):
            return {
                "is_finished": True,
                "current_batch": []
            }

        # Plan the next batch of parallel tasks
        plan_result = self.dynamic_planner.plan_next_batch(
            transcript=transcript,
            context=sample.context,
            abstract_tools=state["abstract_tools"],
            observations=state["observations"],
            completed_tasks=state["completed_tasks"],
            iteration=iteration
        )

        is_finished = plan_result.get("is_finished", False)
        parallel_tasks = plan_result.get("parallel_tasks", [])
        analysis = plan_result.get("analysis", "")

        trajectory.add_step(
            TrajectoryStep(
                step_index=step_idx,
                agent_name="DynamicPlanner",
                thought=f"[Iteration {iteration+1}] Plan: {analysis} | Tasks to dispatch: {len(parallel_tasks)}"
            )
        )

        return {
            "is_finished": is_finished or (len(parallel_tasks) == 0),
            "current_batch": parallel_tasks,
            "step_idx": step_idx,
            "iteration": iteration + 1
        }

    def _node_worker_pool(self, state: LifeMateGraphState) -> Dict[str, Any]:
        """
        Dispatches all tasks in the current batch in parallel / concurrent pool.
        Executes each task through the Guardrail into the Sandbox session.
        """
        tasks = state["current_batch"]
        session = state["session"]
        guardrail = state["guardrail"]
        trajectory = state["trajectory"]
        step_idx = state["step_idx"]

        observations = list(state["observations"])
        completed_tasks = list(state["completed_tasks"])

        # Execute parallel tasks in batch
        for task in tasks:
            step_idx += 1
            tool_name = task.get("tool_name", "")
            task_id = task.get("task_id", f"task_{step_idx}")
            desc = task.get("description", tool_name)

            # Delegate to WorkerAgent to resolve arguments based on schema, directive, and context
            resolved_args, sys_p, msgs, raw_thought = self.worker_agent.resolve_arguments(
                task=task,
                transcript=state["transcript"],
                context=state["sample"].context,
                observations=observations
            )

            # Validate with Guardrail
            is_valid, sanitized_args, err = guardrail.intercept(tool_name, resolved_args, session)

            if is_valid:
                # Call Sandbox
                obs = session.call(tool_name, sanitized_args)
                obs_record = {
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "result": obs.result,
                    "ok": obs.ok
                }
                observations.append(obs_record)
                completed_tasks.append({
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "arguments": sanitized_args,
                    "status": "COMPLETED"
                })

                trajectory.add_step(
                    TrajectoryStep(
                        step_index=step_idx,
                        agent_name="WorkerPool",
                        thought=f"Worker executed [{task_id}] '{desc}'",
                        action=ToolCall(name=tool_name, arguments=sanitized_args),
                        guardrail_status="PASSED",
                        observation=obs
                    )
                )
            else:
                # Intercepted schema error recorded for planner feedback
                observations.append({
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "result": {"error": err},
                    "ok": False
                })
                trajectory.add_step(
                    TrajectoryStep(
                        step_index=step_idx,
                        agent_name="GuardrailInterceptor",
                        thought=f"Blocked invalid call for [{task_id}] '{tool_name}': {err}",
                        action=ToolCall(name=tool_name, arguments=resolved_args),
                        guardrail_status="REJECTED_INTERNAL"
                    )
                )

        return {
            "observations": observations,
            "completed_tasks": completed_tasks,
            "step_idx": step_idx
        }

    def _node_commit(self, state: LifeMateGraphState) -> Dict[str, Any]:
        session = state["session"]
        return {
            "final_tool_calls": session.get_tool_calls(),
            "final_state_hash": session.get_state_hash()
        }

    # --- Conditional Edge Routers ---

    def _route_after_gatekeeper(self, state: LifeMateGraphState) -> str:
        return "in_scope" if state["is_in_scope"] else "out_of_scope"

    def _route_after_planner(self, state: LifeMateGraphState) -> str:
        if state.get("is_finished", False) or len(state.get("current_batch", [])) == 0:
            return "finished"
        return "dispatch"

    # --- Public Processing Interface ---

    def process_sample(self, sample: Sample, split: str = "public_test") -> ExecutionResult:
        """Runs the LangGraph Dynamic StateGraph on a single sample."""
        session = self.sandbox_factory.create_session(split, sample.id)
        guardrail = GuardrailInterceptor(self.catalog, sample.context.allowed_tools)

        initial_state: LifeMateGraphState = {
            "sample": sample,
            "split": split,
            "session": session,
            "guardrail": guardrail,
            "step_idx": 0,
            "max_iterations": self.max_planner_iterations,
            "iteration": 0,
            "observations": [],
            "completed_tasks": [],
            "current_batch": [],
            "is_finished": False,
            "final_tool_calls": [],
            "final_state_hash": ""
        }

        # Invoke the compiled state graph
        final_state = self.app.invoke(initial_state)

        return ExecutionResult(
            id=sample.id,
            tool_calls=final_state["final_tool_calls"],
            state_hash=final_state["final_state_hash"],
            trajectory=final_state["trajectory"]
        )

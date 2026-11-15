"""LifeMate Multi-Agent Pipeline Orchestrator."""

import json
from pathlib import Path
from typing import Any, Optional
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
from .agents.planner_agent import PlannerAgent
from .agents.execution_agent import ExecutionAgent
from .guardrail.schema_checker import GuardrailInterceptor

class LifeMatePipeline:
    """
    Main orchestrator implementing Clean Architecture.
    Coordinates Perception -> Gatekeeper -> Planner -> Execution -> Guardrail -> Sandbox.
    """

    def __init__(
        self,
        asr_engine: AbstractASREngine,
        llm_engine: AbstractLLMEngine,
        sandbox_factory: AbstractSandboxEnvFactory,
        catalog: dict[str, Any],
        max_react_steps: int = 5,
        force_asr: bool = False
    ):
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine
        self.sandbox_factory = sandbox_factory
        self.catalog = catalog
        self.max_react_steps = max_react_steps
        self.force_asr = force_asr

        # Micro-Agents
        self.perception_agent = PerceptionAgent(self.asr_engine, force_asr=force_asr)
        self.gatekeeper_agent = GatekeeperAgent(self.catalog)
        self.planner_agent = PlannerAgent(self.catalog)
        self.execution_agent = ExecutionAgent(self.llm_engine, self.catalog)

    def process_sample(self, sample: Sample, split: str = "public_test") -> ExecutionResult:
        """
        Executes the multi-agent workflow for a single sample.
        Ensures exactly ONE ToolEnv session is opened and preserved.
        """
        # 1. Initialize Sandbox Session
        session = self.sandbox_factory.create_session(split, sample.id)
        guardrail = GuardrailInterceptor(self.catalog, sample.context.allowed_tools)

        # 2. Perception Phase
        transcript = self.perception_agent.perceive(sample)
        trajectory = Trajectory(
            sample_id=sample.id,
            scenario=ScenarioType.EXECUTE,
            transcript=transcript
        )

        step_idx = 1
        trajectory.add_step(
            TrajectoryStep(
                step_index=step_idx,
                agent_name="PerceptionAgent",
                thought=f"Transcribed audio: '{transcript}'",
            )
        )

        # 3. Triage & Gatekeeper Phase
        scenario = self.gatekeeper_agent.classify_intent(transcript, sample.context)
        trajectory.scenario = scenario
        step_idx += 1

        trajectory.add_step(
            TrajectoryStep(
                step_index=step_idx,
                agent_name="GatekeeperAgent",
                thought=f"Classified scenario as: {scenario.value.upper()}",
            )
        )

        # Handle Out-of-Scope: immediately return empty calls list
        if scenario == ScenarioType.OUT_OF_SCOPE:
            return ExecutionResult(
                id=sample.id,
                tool_calls=[],
                state_hash=session.get_state_hash(),
                trajectory=trajectory
            )

        # 4. Planner Phase (Read-before-Write)
        read_calls = self.planner_agent.plan_read_tools(transcript, sample.context)
        observations: list[Observation] = []

        for call in read_calls:
            step_idx += 1
            is_valid, sanitized_args, err = guardrail.intercept(call.name, call.arguments, session)
            if is_valid:
                obs = session.call(call.name, sanitized_args)
                observations.append(obs)
                trajectory.add_step(
                    TrajectoryStep(
                        step_index=step_idx,
                        agent_name="PlannerAgent",
                        thought=f"Executed read discovery tool: {call.name}",
                        action=ToolCall(name=call.name, arguments=sanitized_args),
                        guardrail_status="PASSED",
                        observation=obs
                    )
                )

        # 5. Execution Phase (ReAct Loop)
        system_prompt = self.execution_agent.build_system_prompt(sample.context)
        tools_schema = self.execution_agent.extract_tools_schema(sample.context)

        messages = [
            {"role": "user", "content": f"User request: {transcript}"}
        ]
        for obs in observations:
            messages.append({
                "role": "tool",
                "name": obs.tool_name,
                "content": json.dumps(obs.result, ensure_ascii=False)
            })

        for _ in range(self.max_react_steps):
            step_idx += 1
            decision = self.llm_engine.predict_action(
                system_prompt=system_prompt,
                messages=messages,
                tools_schema=tools_schema,
                temperature=0.0
            )

            if decision.is_done or decision.tool_call is None:
                trajectory.add_step(
                    TrajectoryStep(
                        step_index=step_idx,
                        agent_name="ExecutionAgent",
                        thought=decision.thought or "Execution concluded.",
                    )
                )
                break

            call = decision.tool_call
            is_valid, sanitized_args, err = guardrail.intercept(call.name, call.arguments, session)

            if not is_valid:
                # Intercepted schema error: feedback to LLM without polluting ToolEnv
                trajectory.add_step(
                    TrajectoryStep(
                        step_index=step_idx,
                        agent_name="GuardrailInterceptor",
                        thought=f"Blocked invalid call to '{call.name}': {err}",
                        action=call,
                        guardrail_status="REJECTED_INTERNAL"
                    )
                )
                messages.append({
                    "role": "user",
                    "content": f"System error: The tool call '{call.name}' failed schema validation: {err}. Please correct your arguments or call another tool."
                })
                continue

            # Validated call: execute on session
            obs = session.call(call.name, sanitized_args)
            trajectory.add_step(
                TrajectoryStep(
                    step_index=step_idx,
                    agent_name="ExecutionAgent",
                    thought=decision.thought,
                    action=ToolCall(name=call.name, arguments=sanitized_args),
                    guardrail_status="PASSED",
                    observation=obs
                )
            )

            messages.append({
                "role": "assistant",
                "content": decision.thought,
                "tool_calls": [{"name": call.name, "arguments": sanitized_args}]
            })
            messages.append({
                "role": "tool",
                "name": call.name,
                "content": json.dumps(obs.result, ensure_ascii=False)
            })

        # 6. Assemble Final Output
        final_tool_calls = session.get_tool_calls()
        final_state_hash = session.get_state_hash()

        return ExecutionResult(
            id=sample.id,
            tool_calls=final_tool_calls,
            state_hash=final_state_hash,
            trajectory=trajectory
        )


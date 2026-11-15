"""
LifeMate Single Agent ReAct Pipeline.

Architecture:
  High-Speed End-to-End Single Agent ReAct Loop.
  - Zero multi-agent communication overhead.
  - Injects 100% full tool schemas for all installed apps into the system prompt.
  - Direct Closed-Loop execution with Guardrail schema validation and ToolEnv sandbox.
  - Optimal for fine-tuned sub-5B models (Qwen2.5-3B, Qwen3-4B) and flagships (GPT-6.1-Terra, GPT-5.6-Luna).
"""

import json
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..domain.entities import (
    Sample,
    SampleContext,
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
from .guardrail.schema_checker import GuardrailInterceptor

class LifeMateSingleAgentPipeline:
    """
    High-Performance Single Agent Pipeline.
    Replaces complex multi-agent graphs with a direct, robust ReAct loop.
    Fully compatible with LifeMatePipeline and LifeMateGraphPipeline interfaces.
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

        self.perception_agent = PerceptionAgent(self.asr_engine, force_asr=force_asr)

    def extract_full_installed_schemas(self, context: SampleContext) -> list[dict[str, Any]]:
        """Extracts the FULL parameter and property schemas for installed apps."""
        allowed = context.allowed_tools.copy()
        allowed.add("sys.ask_missing_info")

        full_schemas = []
        for app in self.catalog.get("apps", []):
            app_id = app.get("id", "")
            for tool in app.get("tools", []):
                name = tool.get("name", "")
                if name in allowed:
                    full_schemas.append({
                        "name": name,
                        "app": app_id,
                        "kind": tool.get("kind", "write"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                        "returns": tool.get("returns", "")
                    })
        return full_schemas

    def build_system_prompt(self, context: SampleContext, tool_schemas: list[dict[str, Any]]) -> str:
        """Constructs system prompt containing context, full schemas, and ReAct protocol."""
        return (
            "You are LifeMate, an autonomous voice assistant for Vietnamese users.\n"
            "You translate natural Vietnamese voice requests into concrete, precise tool calls to manage user apps and devices.\n\n"
            "### CURRENT CONTEXT:\n"
            f"- Current Time: {context.current_time}\n"
            f"- Timezone: {context.timezone}\n"
            f"- Location: {json.dumps(context.location, ensure_ascii=False)}\n"
            f"- User Profile: {json.dumps(context.user_profile, ensure_ascii=False)}\n\n"
            "### AVAILABLE TOOLS (FULL SCHEMAS):\n"
            "Below are the exact schemas of all installed tools authorized for this session. You MUST strictly adhere to parameter names, types, and formats.\n"
            f"{json.dumps(tool_schemas, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "### REACT OPERATING PROTOCOL:\n"
            "1. Analyze user request, conditional logic (e.g. if weather is rainy, reschedule; otherwise do nothing), relative time references (\"mai\", \"3 giờ chiều\"), and context.\n"
            "2. READ-BEFORE-WRITE: If a write action requires specific entity IDs (e.g. event_id, contact_id, device_id), search/list them first using [read] tools.\n"
            "3. IN-SCOPE vs OUT-OF-SCOPE: If user request cannot be fulfilled by any installed tool, return empty tool_calls with is_finished=true.\n"
            "4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call `sys.ask_missing_info`.\n\n"
            "### PARAMETER INVOCATION RULES (GENERIC MINIMAL ARGUMENTS CONVENTION):\n"
            "1. MINIMAL ARGUMENT CALLS: Only pass parameters that are strictly required by the schema (`required`) or specifically demanded by the user. Never pass optional parameters with default values.\n"
            "2. IMPLICIT ENUM[0] DEFAULT: For any optional parameter with an `enum` list, the first element (`enum[0]`) is the platform's implicit default value. NEVER pass an optional parameter if its value matches `enum[0]`. Only specify it when the user explicitly requests a non-default choice (`enum[1:]`).\n"
            "3. UNFILTERED READ TOOLS: For query/inspection tools (e.g. names starting with `list_` or `get_status`), invoke them with an empty argument object `{}` so the environment returns the full collection; perform all filtering in memory.\n"
            "4. CONCISE FREE-TEXT: For free-text parameters (messages, notes, task titles), use concise text matching user intent without adding polite conversational padding.\n\n"
            "### OUTPUT FORMAT:\n"
            "You MUST respond STRICTLY in JSON format:\n"
            "{\n"
            "  \"thought\": \"<Detailed reasoning explaining logic and tools selected>\",\n"
            "  \"tool_calls\": [\n"
            "    {\n"
            "      \"name\": \"<tool_name>\",\n"
            "      \"arguments\": {\n"
            "        \"<parameter_name>\": <value>\n"
            "      }\n"
            "    }\n"
            "  ],\n"
            "  \"is_finished\": false,\n"
            "  \"final_response\": null\n"
            "}\n\n"
            "When all actions are completed, or no actions are needed, return:\n"
            "{\n"
            "  \"thought\": \"<Explanation that all tasks are finished or out of scope>\",\n"
            "  \"tool_calls\": [],\n"
            "  \"is_finished\": true,\n"
            "  \"final_response\": \"<Natural Vietnamese reply to the user>\"\n"
            "}\n"
        )

    def _extract_json_response(self, text: str) -> dict[str, Any]:
        """Robustly extracts JSON from raw model text (stripping thinking blocks & markdown)."""
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not clean_text:
            clean_text = text.strip()

        # Try direct parse
        try:
            return json.loads(clean_text)
        except Exception:
            pass

        # Try code block extraction
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except Exception:
                pass

        # Try outermost curly braces
        match = re.search(r"\{[\s\S]*\}", clean_text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {
            "thought": clean_text,
            "tool_calls": [],
            "is_finished": True,
            "final_response": clean_text
        }

    def _build_initial_messages(self, sample: Sample, transcript: str) -> list[dict[str, Any]]:
        """Constructs conversational messages including prior dialog turns and current utterance."""
        messages: list[dict[str, Any]] = []

        # Unpack prior conversation history if present
        for turn in sample.history:
            if turn.role == "user":
                user_text = turn.audio_text or turn.text or ""
                messages.append({"role": "user", "content": user_text})
            elif turn.role == "assistant":
                if turn.tool_calls:
                    tc_dicts = [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in turn.tool_calls]
                    thought_json = json.dumps({
                        "thought": f"Assisting user request: {turn.text or ''}",
                        "tool_calls": tc_dicts,
                        "is_finished": False
                    }, ensure_ascii=False)
                    messages.append({"role": "assistant", "content": thought_json})
                    if turn.tool_results:
                        messages.append({
                            "role": "user",
                            "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(turn.tool_results, ensure_ascii=False, indent=2)}"
                        })
                if turn.text:
                    messages.append({"role": "assistant", "content": turn.text})

        # Append current user voice request
        messages.append({"role": "user", "content": f"User Voice Request: \"{transcript}\""})
        return messages

    def process_sample(self, sample: Sample, split: str = "train") -> ExecutionResult:
        """
        Executes end-to-end processing of a sample using the single-agent ReAct loop.
        """
        # 1. Perception
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

        # 2. Context & Environment
        context = sample.context
        session = self.sandbox_factory.create_session(split, sample.id)
        guardrail = GuardrailInterceptor(self.catalog, context.allowed_tools)
        tool_schemas = self.extract_full_installed_schemas(context)
        system_prompt = self.build_system_prompt(context, tool_schemas)

        messages = self._build_initial_messages(sample, transcript)

        step_idx = 1
        turn = 0

        # 3. Single Agent ReAct Loop
        while turn < self.max_react_steps:
            turn += 1
            step_idx += 1

            decision = self.llm_engine.predict_action(
                system_prompt=system_prompt,
                messages=messages,
                tools_schema=[],
                temperature=0.0
            )

            raw_thought = decision.thought or ""
            parsed = self._extract_json_response(raw_thought)

            thought = parsed.get("thought", raw_thought)
            tool_calls = parsed.get("tool_calls", [])
            is_finished = parsed.get("is_finished", False)
            final_resp = parsed.get("final_response")

            trajectory.add_step(
                TrajectoryStep(
                    step_index=step_idx,
                    agent_name="SingleAgent",
                    thought=f"[Turn {turn}] {thought} | Tool calls dispatched: {len(tool_calls)}"
                )
            )

            # If finished or empty tool calls
            if is_finished or not tool_calls:
                break

            # Execute tool calls
            observations = []
            for call in tool_calls:
                step_idx += 1
                tool_name = call.get("name", "")
                tool_args = call.get("arguments", {})

                # Validate with Guardrail
                is_valid, sanitized_args, err = guardrail.intercept(tool_name, tool_args, session)

                if is_valid:
                    obs = session.call(tool_name, sanitized_args)
                    observations.append({
                        "tool_name": tool_name,
                        "ok": obs.ok,
                        "result": obs.result
                    })
                    trajectory.add_step(
                        TrajectoryStep(
                            step_index=step_idx,
                            agent_name="SingleAgent",
                            thought=f"Executed: {tool_name}",
                            action=ToolCall(name=tool_name, arguments=sanitized_args),
                            guardrail_status="PASSED",
                            observation=obs
                        )
                    )
                else:
                    observations.append({
                        "tool_name": tool_name,
                        "ok": False,
                        "error": err
                    })
                    trajectory.add_step(
                        TrajectoryStep(
                            step_index=step_idx,
                            agent_name="GuardrailInterceptor",
                            thought=f"Blocked: {tool_name}: {err}",
                            action=ToolCall(name=tool_name, arguments=tool_args),
                            guardrail_status="REJECTED_INTERNAL"
                        )
                    )

            # Append assistant reasoning & observations for next turn
            messages.append({"role": "assistant", "content": raw_thought})
            messages.append({
                "role": "user",
                "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(observations, ensure_ascii=False, indent=2)}\n\nPlease provide your next tool_calls or final response."
            })

        # 4. Final Submission Assembly
        final_tool_calls = session.get_tool_calls()
        final_state_hash = session.get_state_hash()

        return ExecutionResult(
            id=sample.id,
            tool_calls=final_tool_calls,
            state_hash=final_state_hash,
            trajectory=trajectory
        )


"""
Worker Agent: 100% Generic & Schema-Driven Execution Agent.

Executes a single task assigned by DynamicPlanner:
- Dynamically fetches the full JSON schema for the specified tool from catalog.
- Injects tool schema, user context, task directive, and prior observations into LLM prompt.
- Extracts and validates structured arguments for tool execution.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from ...domain.entities import SampleContext, ToolCall
from ...ports.llm_port import AbstractLLMEngine


class WorkerAgent:
    """
    Execution Worker Agent within the Worker Pool.
    Specializes in turning a Planner task directive into concrete,
    schema-compliant tool arguments.
    """

    def __init__(self, llm_engine: AbstractLLMEngine, catalog: dict[str, Any]):
        self.llm_engine = llm_engine
        self.catalog = catalog
        self._tool_schema_map = self._build_tool_schema_map()
        self.last_system_prompt: Optional[str] = None
        self.last_messages: Optional[list[dict[str, str]]] = None
        self.last_raw_response: Optional[str] = None

    def _build_tool_schema_map(self) -> dict[str, dict[str, Any]]:
        schemas = {}
        for app in self.catalog.get("apps", []):
            for tool in app.get("tools", []):
                schemas[tool["name"]] = tool
        return schemas

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        return self._tool_schema_map.get(tool_name, {
            "name": tool_name,
            "description": "Generic external tool",
            "parameters": {"type": "object", "properties": {}, "required": []}
        })

    def build_system_prompt(self, tool_name: str, context: SampleContext) -> str:
        tool_schema = self.get_tool_schema(tool_name)
        return (
            "You are LifeMate's Tool Execution Worker, an expert in generating precise tool arguments in Vietnamese.\n"
            f"Current Time: {context.current_time} (Timezone: {context.timezone})\n"
            f"User Location: {context.location}\n"
            f"User Profile: {context.user_profile}\n\n"
            f"# TOOL SCHEMA SPECIFICATION for `{tool_name}`:\n"
            f"{json.dumps(tool_schema, ensure_ascii=False, indent=2)}\n\n"
            "# EXECUTION PRINCIPLES:\n"
            "1. Extract and compute accurate arguments strictly according to the tool schema types, formats, and required fields.\n"
            "2. Use the task directive, original user request, and observations from previous steps to resolve entity IDs (e.g. event_id, contact_id, device_id) and values.\n"
            "3. Return strictly a valid JSON object matching:\n"
            "{\n"
            "  \"reasoning\": \"<Concise reasoning for parameter values>\",\n"
            "  \"arguments\": {\n"
            "    \"<parameter_name>\": <value>\n"
            "  }\n"
            "}"
        )

    def resolve_arguments(
        self,
        task: dict[str, Any],
        transcript: str,
        context: SampleContext,
        observations: list[dict[str, Any]]
    ) -> Tuple[dict[str, Any], str, list[dict[str, str]], str]:
        """
        Resolves tool arguments using LLM reasoning over the tool schema.
        Returns: (arguments, system_prompt, messages, raw_thought)
        """
        tool_name = task.get("tool_name", "")
        tool_schema = self.get_tool_schema(tool_name)
        params = tool_schema.get("parameters", {})
        required = params.get("required", [])
        props = params.get("properties", {})

        system_prompt = self.build_system_prompt(tool_name, context)

        # Fast-path optimization: if tool has zero required parameters and zero properties, return {}
        if not required and not props:
            self.last_system_prompt = system_prompt
            self.last_messages = [{"role": "user", "content": "No parameters required."}]
            self.last_raw_response = "Tool has no parameters."
            return {}, system_prompt, self.last_messages, self.last_raw_response

        # Fast-path: if planner already provided valid non-empty arguments matching required fields
        provided_args = task.get("arguments", {})
        if provided_args and all(r in provided_args for r in required):
            self.last_system_prompt = system_prompt
            self.last_messages = [{"role": "user", "content": json.dumps(provided_args)}]
            self.last_raw_response = json.dumps(provided_args)
            return provided_args, system_prompt, self.last_messages, self.last_raw_response

        user_content = {
            "task_id": task.get("task_id", ""),
            "tool_name": tool_name,
            "instruction": task.get("description", ""),
            "original_user_request": transcript,
            "observations_received": observations
        }

        messages = [
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False, indent=2)}
        ]

        self.last_system_prompt = system_prompt
        self.last_messages = messages

        decision = self.llm_engine.predict_action(
            system_prompt=system_prompt,
            messages=messages,
            tools_schema=[],
            temperature=0.0
        )

        raw_thought = decision.thought or ""
        self.last_raw_response = raw_thought

        args = self._parse_worker_output(raw_thought, tool_schema, context, task)
        return args, system_prompt, messages, raw_thought

    def _parse_worker_output(
        self,
        raw_text: str,
        tool_schema: dict[str, Any],
        context: SampleContext,
        task: dict[str, Any]
    ) -> dict[str, Any]:
        """Extracts JSON arguments from LLM output, with generic schema fallback."""
        json_obj = None

        # Strip thinking blocks if present (e.g. Qwen3-Thinking, DeepSeek-R1)
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        if not clean_text:
            clean_text = raw_text

        block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if block_match:
            try:
                json_obj = json.loads(block_match.group(1))
            except Exception:
                pass

        if json_obj is None:
            raw_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if raw_match:
                try:
                    json_obj = json.loads(raw_match.group(1))
                except Exception:
                    pass

        if json_obj and isinstance(json_obj, dict):
            if "arguments" in json_obj and isinstance(json_obj["arguments"], dict):
                return json_obj["arguments"]
            # If the output itself is the arguments dict
            params = tool_schema.get("parameters", {}).get("properties", {})
            if any(k in params for k in json_obj.keys()):
                return json_obj

        # Fallback to provided args or schema defaults
        if "arguments" in task and isinstance(task["arguments"], dict):
            return task["arguments"]
        return {}


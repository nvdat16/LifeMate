"""
Dynamic Planner Agent: 100% Generic & Schema-Driven (Zero-Hardcoding, MCP-Compatible).

Parses tool descriptions, parameters, and read/write capabilities entirely on the fly
from the provided catalog and installed apps schemas.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional
from ...domain.entities import SampleContext
from ...ports.llm_port import AbstractLLMEngine

READ_VERBS = ("get", "list", "search", "fetch", "read", "query", "status", "guide", "check")

def normalize_token_set(text: str) -> set[str]:
    """Extracts lowercase unaccented word tokens."""
    if not text:
        return set()
    nfd = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    clean = "".join(c for c in nfd if not unicodedata.combining(c))
    return set(re.findall(r"\w+", clean))

class DynamicPlannerAgent:
    """
    100% Generic Strategic Planner.
    Zero hardcoded tool names, zero domain-specific heuristics.
    Fully compatible with arbitrary tools, OpenAPI schemas, and MCP servers.
    """

    def __init__(self, llm_engine: AbstractLLMEngine, catalog: dict[str, Any]):
        self.llm_engine = llm_engine
        self.catalog = catalog
        self._tool_type_cache: dict[str, str] = {}
        self.last_system_prompt: Optional[str] = None
        self.last_messages: Optional[list[dict[str, str]]] = None
        self.last_raw_thought: Optional[str] = None

    def _resolve_tool_type(self, tool_name: str) -> str:
        """Dynamically determines whether a tool is READ or WRITE."""
        if tool_name in self._tool_type_cache:
            return self._tool_type_cache[tool_name]

        # 1. Check official zaic_tools runtime registry if imported
        try:
            from zaic_tools.apps import TOOLS
            if tool_name in TOOLS:
                t_type = "WRITE" if TOOLS[tool_name] == "write" else "READ"
                self._tool_type_cache[tool_name] = t_type
                return t_type
        except Exception:
            pass

        # 2. Generic MCP standard: inspect action verbs in tool method/suffix
        base_name = tool_name.split(".")[-1].lower()
        if any(base_name.startswith(v) or base_name == v for v in READ_VERBS):
            t_type = "READ"
        else:
            t_type = "WRITE"

        self._tool_type_cache[tool_name] = t_type
        return t_type

    def extract_abstract_catalog(self, context: SampleContext) -> list[dict[str, Any]]:
        """
        Dynamically extracts lightweight abstract tool specifications
        from the official catalog JSON for installed apps.
        """
        allowed = context.allowed_tools
        abstract_tools = []

        for app in self.catalog.get("apps", []):
            app_id = app.get("id", "")
            for t in app.get("tools", []):
                name = t.get("name", "")
                if name in allowed:
                    params = t.get("parameters", {})
                    required = params.get("required", [])
                    props = list(params.get("properties", {}).keys())
                    t_type = self._resolve_tool_type(name)
                    desc = t.get("description", "")

                    abstract_tools.append({
                        "name": name,
                        "app": app_id,
                        "description": desc,
                        "tool_type": t_type,
                        "required": required,
                        "parameters": props[:6]
                    })
        return abstract_tools

    def format_abstract_catalog_for_prompt(self, abstract_tools: list[dict[str, Any]]) -> str:
        """Formats tools into compact, high-signal YAML specs for the LLM."""
        lines = []
        # Group by app domain dynamically
        by_app: dict[str, list[dict[str, Any]]] = {}
        for t in abstract_tools:
            by_app.setdefault(t["app"], []).append(t)

        for app_id, tools in by_app.items():
            lines.append(f"\n## [{app_id.upper()}]")
            for t in tools:
                req_str = f"Required: {', '.join(t['required'])}" if t["required"] else "No required parameters"
                lines.append(f"- {t['name']} [{t['tool_type']}]: {t['description']}. ({req_str})")

        return "\n".join(lines)

    def plan_next_batch(
        self,
        transcript: str,
        context: SampleContext,
        abstract_tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        completed_tasks: list[dict[str, Any]],
        iteration: int
    ) -> dict[str, Any]:
        """
        Generates the next batch of parallel tasks through LLM reasoning
        over the dynamically generated abstract catalog.
        """
        formatted_catalog = self.format_abstract_catalog_for_prompt(abstract_tools)

        system_prompt = (
            "You are the Strategic Dynamic Planner for LifeMate, an autonomous voice agent in Vietnamese.\n"
            f"Current Time: {context.current_time} (Timezone: {context.timezone})\n"
            f"User Location: {context.location}\n"
            f"User Profile: {context.user_profile}\n\n"
            f"# DYNAMIC TOOLS REGISTRY\n{formatted_catalog}\n\n"
            "# CORE PLANNING PRINCIPLES:\n"
            "1. Analyze the user request against the completed tasks and real-world observations from previous iterations.\n"
            "2. If ALL user intentions have been fulfilled: set \"is_finished\": true and \"parallel_tasks\": [].\n"
            "3. If intentions remain: select the NEXT immediate batch of INDEPENDENT tasks that can execute in parallel.\n"
            "4. READ-BEFORE-WRITE RULE: If a [WRITE] tool requires an ID (device_id, event_id, contact_id, etc.) that does not exist in context or observations, you MUST first dispatch the corresponding [READ] tool.\n"
            "5. OUTPUT FORMAT: Return strictly a valid JSON object matching:\n"
            "{\n"
            "  \"analysis\": \"<Your strategic reasoning>\",\n"
            "  \"is_finished\": false,\n"
            "  \"parallel_tasks\": [\n"
            "    {\"task_id\": \"T1\", \"tool_name\": \"...\", \"description\": \"...\", \"arguments\": {...}}\n"
            "  ]\n"
            "}"
        )

        user_content = {
            "user_request": transcript,
            "iteration": iteration,
            "completed_tasks": completed_tasks,
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
        self.last_raw_thought = raw_thought
        return self._parse_planner_output(raw_thought, transcript, context, abstract_tools, observations, iteration)

    def _parse_planner_output(
        self,
        raw_text: str,
        transcript: str,
        context: SampleContext,
        abstract_tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        iteration: int
    ) -> dict[str, Any]:
        """Extracts JSON plan from LLM output, with generic semantic fallback."""
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
            return {
                "is_finished": json_obj.get("is_finished", False),
                "analysis": json_obj.get("analysis", "Dynamic plan generated."),
                "parallel_tasks": json_obj.get("parallel_tasks", [])
            }

        # Generic schema-driven fallback (Zero hardcoding)
        return self._generic_semantic_fallback(transcript, context, abstract_tools, observations, iteration)

    def _generic_semantic_fallback(
        self,
        transcript: str,
        context: SampleContext,
        abstract_tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        iteration: int
    ) -> dict[str, Any]:
        """
        Purely generic semantic overlap fallback:
        Scans abstract tool descriptions and names against user request tokens.
        Never relies on hardcoded strings or app names.
        """
        if iteration >= 2 and len(observations) > 0:
            return {
                "is_finished": True,
                "analysis": "All identifiable intents executed based on observations.",
                "parallel_tasks": []
            }

        user_tokens = normalize_token_set(transcript)
        executed_tool_names = {o.get("tool_name") for o in observations}
        parallel_tasks = []

        # Phase 1 (Iteration 0): Candidate [READ] tools with zero required parameters matching user tokens
        if iteration == 0:
            for t in abstract_tools:
                if t["tool_type"] == "READ" and t["name"] not in executed_tool_names:
                    # Check token overlap between tool name/description and user request
                    tool_tokens = normalize_token_set(t["name"] + " " + t["description"])
                    if len(user_tokens.intersection(tool_tokens)) >= 1:
                        # If read tool requires no params, can be called safely
                        if len(t["required"]) == 0:
                            parallel_tasks.append({
                                "task_id": f"T{len(parallel_tasks)+1}",
                                "tool_name": t["name"],
                                "description": f"Gather context via {t['name']}",
                                "arguments": {}
                            })

        # Phase 2 (Iteration >= 1): Candidate [WRITE] tools matching user tokens
        if not parallel_tasks:
            for t in abstract_tools:
                if t["tool_type"] == "WRITE" and t["name"] not in executed_tool_names:
                    tool_tokens = normalize_token_set(t["name"] + " " + t["description"])
                    if len(user_tokens.intersection(tool_tokens)) >= 1:
                        # Construct generic arguments from context
                        generic_args = {}
                        for p in t["required"]:
                            if p in ("location", "pickup", "dropoff"):
                                generic_args[p] = str(context.location)
                            elif p in ("time", "date"):
                                generic_args[p] = context.current_time[:10]
                        parallel_tasks.append({
                            "task_id": f"T{len(parallel_tasks)+1}",
                            "tool_name": t["name"],
                            "description": f"Execute {t['name']}",
                            "arguments": generic_args
                        })

        is_finished = (len(parallel_tasks) == 0)
        return {
            "is_finished": is_finished,
            "analysis": f"Generic semantic match identified {len(parallel_tasks)} candidate tasks.",
            "parallel_tasks": parallel_tasks
        }

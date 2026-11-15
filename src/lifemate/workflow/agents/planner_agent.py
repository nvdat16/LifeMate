"""
Planner Agent: Generic Read-before-Write discovery and tool dependency planner.
Zero hardcoding: Dynamically inspects catalog schemas for candidate [READ] tools.
"""

import re
import unicodedata
from typing import Any, List
from ...domain.entities import ToolCall, SampleContext

READ_VERBS = ("get", "list", "search", "fetch", "read", "query", "status", "guide", "check")

def extract_tokens(text: str) -> set[str]:
    if not text:
        return set()
    s_norm = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r"[a-zà-ỹ0-9_]+", s_norm)
    return {w for w in words if len(w) >= 3}

class PlannerAgent:
    """Plans preliminary Read tools dynamically based on schema definitions."""

    def __init__(self, catalog: dict[str, Any]):
        self.catalog = catalog
        self._read_tools_cache = self._index_read_tools()

    def _index_read_tools(self) -> list[dict[str, Any]]:
        """Finds all candidate [READ] discovery tools with zero required parameters."""
        candidates = []
        for app in self.catalog.get("apps", []):
            for t in app.get("tools", []):
                name = t.get("name", "")
                base = name.split(".")[-1].lower()
                params = t.get("parameters", {})
                required = params.get("required", [])

                # A read discovery tool has a read verb and zero required parameters
                is_read = any(base.startswith(v) or base == v for v in READ_VERBS)
                if is_read and len(required) == 0:
                    tokens = extract_tokens(name + " " + t.get("description", ""))
                    candidates.append({
                        "name": name,
                        "tokens": tokens
                    })
        return candidates

    def plan_read_tools(self, transcript: str, context: SampleContext) -> list[ToolCall]:
        """
        Dynamically emits preliminary read tool calls if the transcript matches
        their schema tokens and the tool is installed.
        """
        calls: list[ToolCall] = []
        u_tokens = extract_tokens(transcript)
        allowed = context.allowed_tools

        for item in self._read_tools_cache:
            name = item["name"]
            if name in allowed:
                # If user transcript references concepts in this discovery tool
                if u_tokens.intersection(item["tokens"]):
                    calls.append(ToolCall(name=name, arguments={}))

        return calls

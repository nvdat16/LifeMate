"""Execution Agent: Generates write tool actions from intent and read observations."""

from typing import Any, Optional
from ...domain.entities import ToolCall, Observation, SampleContext
from ...ports.llm_port import AbstractLLMEngine

class ExecutionAgent:
    """Orchestrates LLM prompt construction and generates write actions."""

    def __init__(self, llm_engine: AbstractLLMEngine, catalog: dict[str, Any]):
        self.llm_engine = llm_engine
        self.catalog = catalog

    def build_system_prompt(self, context: SampleContext) -> str:
        return (
            "You are LifeMate, an agentic voice assistant executing tasks in Vietnamese.\n"
            f"Current time: {context.current_time} (Timezone: {context.timezone})\n"
            f"User profile: {context.user_profile}\n"
            f"Location: {context.location}\n"
            "Rules:\n"
            "1. Only call tools present in the provided tools catalog.\n"
            "2. Read tools must precede write tools when entity IDs are needed.\n"
            "3. If crucial information is missing and cannot be found in state, call sys.ask_missing_info.\n"
            "4. Return DONE when all tasks are complete."
        )

    def extract_tools_schema(self, context: SampleContext) -> list[dict[str, Any]]:
        """Filters catalog to only include installed tools for this sample."""
        allowed = context.allowed_tools
        schemas = []
        for app in self.catalog.get("apps", []):
            for t in app.get("tools", []):
                if t["name"] in allowed:
                    schemas.append(t)
        return schemas


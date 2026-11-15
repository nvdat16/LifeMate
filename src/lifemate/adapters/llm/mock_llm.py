"""Mock LLM adapter for testing and deterministic evaluation on CPU."""

from typing import Any, Optional
from ...domain.entities import ToolCall
from ...ports.llm_port import AbstractLLMEngine, LLMDecision

class MockLLMEngine(AbstractLLMEngine):
    """
    Simulates LLM tool calling. Can replay predefined tool calls for specific samples
    or return smart heuristic decisions for zero-GPU local execution.
    """

    def __init__(self):
        self._preset_calls: dict[str, list[ToolCall]] = {}
        self._sample_call_idx: dict[str, int] = {}

    def register_scenario_calls(self, sample_id: str, calls: list[ToolCall]):
        """Sets the sequence of tool calls that this mock engine will emit for sample_id."""
        self._preset_calls[sample_id] = calls
        self._sample_call_idx[sample_id] = 0

    def predict_action(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        temperature: float = 0.0
    ) -> LLMDecision:
        # Check if the active sample has preset calls
        # We can extract the sample_id or current step from messages
        # If no calls remain -> is_done = True
        return LLMDecision(
            thought="Mock decision: Task execution completed or no more actions needed.",
            tool_call=None,
            is_done=True
        )


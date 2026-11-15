"""Port interface for LLM reasoning and tool generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from ..domain.entities import ToolCall

@dataclass
class LLMDecision:
    """Represents a reasoning step output by the LLM."""
    thought: str
    tool_call: Optional[ToolCall] = None
    is_done: bool = False
    raw_response: Optional[str] = None

class AbstractLLMEngine(ABC):
    """Contract for any LLM backend (Qwen2.5, Mock, Ollama, vLLM, HF)."""

    @abstractmethod
    def predict_action(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        temperature: float = 0.0
    ) -> LLMDecision:
        """
        Generate the next thought and tool call decision given the current context.

        Args:
            system_prompt: System prompt instructing the agent and injecting constraints.
            messages: Conversation history, observations, and user utterances.
            tools_schema: List of schemas for the installed tools.
            temperature: Sampling temperature (must be 0.0 for deterministic evaluation).

        Returns:
            LLMDecision containing thought and optionally a ToolCall.
        """
        pass


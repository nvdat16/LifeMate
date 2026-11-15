"""Port interface for Sandbox Environment execution (zaic_tools abstraction)."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from ..domain.entities import ToolCall, Observation

class AbstractSandboxSession(ABC):
    """Contract for an active sandbox session for a single sample."""

    @abstractmethod
    def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate schema only. Does NOT execute or mutate state.
        
        Returns:
            {"ok": bool, "error": Optional[str]}
        """
        pass

    @abstractmethod
    def call(self, tool_name: str, arguments: dict[str, Any]) -> Observation:
        """
        Execute the tool in the sandbox session.
        Mutates state in memory and records the call.
        
        Returns:
            Observation containing result dict and ok status.
        """
        pass

    @abstractmethod
    def get_tool_calls(self) -> list[dict[str, Any]]:
        """
        Returns all calls made in this session in execution order,
        including any rejected calls.
        """
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Returns deep copy of current in-memory state for all apps."""
        pass

    @abstractmethod
    def get_state_hash(self) -> str:
        """Returns sha256 canonical hash of the final state."""
        pass

class AbstractSandboxEnvFactory(ABC):
    """Factory to create a fresh sandbox session for a given sample."""

    @abstractmethod
    def create_session(self, split: str, sample_id: str) -> AbstractSandboxSession:
        """Instantiates a new independent sandbox session for sample_id."""
        pass


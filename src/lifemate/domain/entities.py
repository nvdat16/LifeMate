"""Domain Entities for LifeMate Voice Agent."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from .scenario import ScenarioType

@dataclass
class ToolCall:
    """Represents a single tool call."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {})
        )

@dataclass
class Observation:
    """Represents the result returned by ToolEnv for a tool call."""
    tool_name: str
    result: dict[str, Any]
    ok: bool = True
    error_message: Optional[str] = None

@dataclass
class ConversationTurn:
    """Represents a prior turn in conversation history."""
    role: str  # 'user' or 'assistant'
    audio: Optional[str] = None
    audio_text: Optional[str] = None
    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class SampleContext:
    """Context holding user profile, location, current time and installed apps."""
    current_time: str
    timezone: str
    location: dict[str, float]
    user_profile: dict[str, Any]
    installed_apps: dict[str, dict[str, list[str]]]

    @property
    def allowed_tools(self) -> set[str]:
        """Returns the set of all tool names allowed for this sample + sys tools."""
        tools = {"sys.ask_missing_info"}
        for app_info in self.installed_apps.values():
            for t in app_info.get("tools", []):
                tools.add(t)
        return tools

@dataclass
class Sample:
    """Complete representation of a challenge sample."""
    id: str
    audio: str
    context: SampleContext
    history: list[ConversationTurn] = field(default_factory=list)
    audio_text: Optional[str] = None  # Available in train set
    label: Optional[dict[str, Any]] = None  # Available in train set (gold tool_calls, final_state)

@dataclass
class TrajectoryStep:
    """A step within the Agent's ReAct execution trajectory."""
    step_index: int
    agent_name: str
    thought: str
    action: Optional[ToolCall] = None
    guardrail_status: str = "PASSED"  # 'PASSED', 'AUTOCASTED', 'REJECTED_INTERNAL'
    observation: Optional[Observation] = None

@dataclass
class Trajectory:
    """Complete trace of an agent session execution."""
    sample_id: str
    scenario: ScenarioType
    transcript: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

@dataclass
class ExecutionResult:
    """Final result matching the competition submission format."""
    id: str
    tool_calls: list[dict[str, Any]]
    state_hash: str
    trajectory: Optional[Trajectory] = None

    def to_submission_dict(self) -> dict[str, Any]:
        """Serializes exactly to the format required by Zalo AI Challenge."""
        return {
            "id": self.id,
            "tool_calls": self.tool_calls,
            "state_hash": self.state_hash
        }


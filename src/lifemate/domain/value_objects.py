"""Domain Value Objects for LifeMate."""

from dataclasses import dataclass
import re

@dataclass(frozen=True)
class ToolName:
    """Represents a qualified tool name '<app>.<tool>'."""
    value: str

    def __post_init__(self):
        if "." not in self.value:
            raise ValueError(f"Invalid qualified tool name: '{self.value}'. Must be in '<app>.<tool>' format.")

    @property
    def app(self) -> str:
        return self.value.split(".", 1)[0]

    @property
    def method(self) -> str:
        return self.value.split(".", 1)[1]

    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class StateHash:
    """Represents a canonical SHA-256 state hash."""
    value: str

    def __post_init__(self):
        if not re.match(r"^[a-f0-9]{64}$", self.value):
            raise ValueError(f"Invalid SHA-256 state hash: '{self.value}'")

    def __str__(self) -> str:
        return self.value


"""Domain Enum defining the core operational scenarios for LifeMate Voice Agent."""

from enum import Enum

class ScenarioType(str, Enum):
    """
    Three distinct scenarios defined by Zalo AI Challenge LifeMate:
    1. EXECUTE: The user's request is actionable by installed apps -> perform multi-step tool calls.
    2. ASK_MISSING_INFO: Mandatory parameters are missing and cannot be inferred -> call sys.ask_missing_info.
    3. OUT_OF_SCOPE: None of the installed apps can handle this intent -> return empty tool calls list [].
    """
    EXECUTE = "execute"
    ASK_MISSING_INFO = "ask_missing_info"
    OUT_OF_SCOPE = "out_of_scope"

    def __str__(self) -> str:
        return self.value


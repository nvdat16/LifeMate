"""Domain exceptions for LifeMate Voice Agent."""

class LifeMateDomainException(Exception):
    """Base exception for all domain errors."""
    pass

class ToolNotInstalledException(LifeMateDomainException):
    """Raised when an agent attempts to invoke a tool outside installed_apps."""
    def __init__(self, tool_name: str, allowed_tools: list[str]):
        super().__init__(
            f"Tool '{tool_name}' is not in context.installed_apps. Allowed: {len(allowed_tools)} tools."
        )
        self.tool_name = tool_name
        self.allowed_tools = allowed_tools

class SchemaViolationException(LifeMateDomainException):
    """Raised when arguments violate the JSON schema catalog."""
    def __init__(self, tool_name: str, error_message: str):
        super().__init__(f"Schema violation for '{tool_name}': {error_message}")
        self.tool_name = tool_name
        self.error_message = error_message

class MissingInformationException(LifeMateDomainException):
    """Raised when mandatory params cannot be discovered or resolved."""
    def __init__(self, tool_name: str, missing_params: list[str]):
        super().__init__(f"Tool '{tool_name}' is missing mandatory parameters: {missing_params}")
        self.tool_name = tool_name
        self.missing_params = missing_params

class OutOfScopeException(LifeMateDomainException):
    """Raised when user intent does not match any installed app."""
    def __init__(self, intent_description: str):
        super().__init__(f"Request intent is out of scope: {intent_description}")
        self.intent_description = intent_description


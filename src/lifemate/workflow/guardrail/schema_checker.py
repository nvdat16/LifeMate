"""Active Guardrail Interceptor protecting ToolEnv from schema errors."""

from typing import Any, Optional
from ...ports.sandbox_port import AbstractSandboxSession
from .type_caster import TypeCaster

class GuardrailInterceptor:
    """
    Prevents rejected tool calls from entering ToolEnv session.
    A rejected call in zaic_tools counts as an unmatched call (U penalty),
    damaging the F1 score and failing to update the state.
    """

    def __init__(self, catalog: dict[str, Any], allowed_tools: set[str]):
        self.catalog = catalog
        self.allowed_tools = allowed_tools
        self._tools_map = self._build_tools_map()

    def _build_tools_map(self) -> dict[str, Any]:
        mapping = {}
        for app in self.catalog.get("apps", []):
            for t in app.get("tools", []):
                mapping[t["name"]] = t
        if "sys.ask_missing_info" not in mapping:
            mapping["sys.ask_missing_info"] = {
                "name": "sys.ask_missing_info",
                "parameters": {
                    "required": ["tool_name", "params"],
                    "properties": {
                        "tool_name": {"type": "string"},
                        "params": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        return mapping

    def intercept(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: AbstractSandboxSession
    ) -> tuple[bool, dict[str, Any], Optional[str]]:
        """
        Intercepts and inspects a predicted tool call.

        Returns:
            (is_valid, sanitized_arguments, error_message)
        """
        # 1. Whitelist Check
        if tool_name not in self.allowed_tools:
            return False, arguments, f"Tool '{tool_name}' is not in context.installed_apps"

        # 2. Type Casting & Sanitization
        tool_spec = self._tools_map.get(tool_name, {})
        sanitized_args = TypeCaster.sanitize_arguments(arguments, tool_spec)

        # 3. Pre-validation via Sandbox (validate-only, no state mutation)
        val_res = session.validate(tool_name, sanitized_args)
        if not val_res.get("ok", True):
            error_msg = val_res.get("error", "Schema validation failed")
            return False, sanitized_args, error_msg

        return True, sanitized_args, None


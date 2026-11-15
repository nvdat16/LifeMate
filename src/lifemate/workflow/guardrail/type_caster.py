"""Active Type Caster and Parameter Sanitizer for Guardrail Engine."""

import re
from datetime import datetime
from typing import Any

class TypeCaster:
    """
    Sanitizes and auto-casts tool arguments based on catalog JSON schema.
    Prevents schema rejections for minor type mismatches (e.g., string integers, unpadded times).
    """

    @staticmethod
    def sanitize_arguments(
        arguments: dict[str, Any],
        tool_spec: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return {}

        sanitized = dict(arguments)
        params_spec = tool_spec.get("parameters", {})
        props = params_spec.get("properties", {})

        for key, val in list(sanitized.items()):
            if val is None or val == "":
                continue

            prop_spec = props.get(key, {})
            expected_type = prop_spec.get("type")
            expected_format = prop_spec.get("format")

            # 1. Integer casting
            if expected_type == "integer":
                try:
                    if isinstance(val, str):
                        # Extract digits if user wrote '26 độ' or '1000đ'
                        match = re.search(r"-?\d+", val)
                        if match:
                            sanitized[key] = int(match.group(0))
                    elif isinstance(val, (float, int)):
                        sanitized[key] = int(val)
                except Exception:
                    pass

            # 2. Number/float casting
            elif expected_type == "number":
                try:
                    if isinstance(val, str):
                        match = re.search(r"-?\d+(\.\d+)?", val)
                        if match:
                            sanitized[key] = float(match.group(0))
                    elif isinstance(val, (int, float)):
                        sanitized[key] = float(val)
                except Exception:
                    pass

            # 3. Boolean casting
            elif expected_type == "boolean":
                if isinstance(val, str):
                    lower = val.lower().strip()
                    if lower in ("true", "1", "yes", "bật", "on"):
                        sanitized[key] = True
                    elif lower in ("false", "0", "no", "tắt", "off"):
                        sanitized[key] = False

            # 4. Format normalization (date, time, date-time)
            elif expected_type == "string" and isinstance(val, str):
                if expected_format == "time":
                    # Convert '7:00' to '07:00' or '7h' to '07:00'
                    m = re.match(r"^(\d{1,2})[:h](\d{2})", val)
                    if m:
                        hh, mm = int(m.group(1)), int(m.group(2))
                        sanitized[key] = f"{hh:02d}:{mm:02d}"

                elif expected_format == "date":
                    # Ensure YYYY-MM-DD
                    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", val)
                    if m:
                        y, m_val, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        sanitized[key] = f"{y:04d}-{m_val:02d}-{d:02d}"

        return sanitized


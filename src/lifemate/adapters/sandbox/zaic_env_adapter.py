"""Adapter wrapping zaic_tools for sandbox session execution."""

import copy
import importlib
import sys
from pathlib import Path
from typing import Any, Union
from ...domain.entities import Observation
from ...ports.sandbox_port import AbstractSandboxSession, AbstractSandboxEnvFactory

class ZaicSandboxSession(AbstractSandboxSession):
    """Wraps a concrete zaic_tools.Session object."""

    def __init__(self, raw_session: Any):
        self._session = raw_session

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Runs zaic_tools schema check only (no state mutation, not recorded)."""
        return self._session.validate(tool_name, arguments or {})

    def call(self, tool_name: str, arguments: dict[str, Any]) -> Observation:
        """Executes tool on state in-memory and appends to session tool_calls log."""
        raw_res = self._session.call(tool_name, arguments or {})
        ok = raw_res.get("ok", True)
        err = raw_res.get("error") if not ok else None
        return Observation(
            tool_name=tool_name,
            result=raw_res,
            ok=ok,
            error_message=err
        )

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._session.get_tool_calls())

    def get_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._session.get_state())

    def get_state_hash(self) -> str:
        return self._session.get_state_hash()


class ZaicSandboxEnvFactory(AbstractSandboxEnvFactory):
    """Factory creating ZaicSandboxSession instances for any split."""

    def __init__(self, data_root: Union[str, Path] = "."):
        self.data_root = Path(data_root).resolve()
        self._ensure_zaic_tools_imported()

    def _ensure_zaic_tools_imported(self):
        """Ensures zaic_tools from the workspace is available in sys.path."""
        # Prefer public_test/zaic_tools or train/zaic_tools
        candidates = [
            self.data_root / "public_test",
            self.data_root / "train",
        ]
        for c in candidates:
            if (c / "zaic_tools").is_dir() and str(c) not in sys.path:
                sys.path.insert(0, str(c))
                break

    def create_session(self, split: str, sample_id: str) -> AbstractSandboxSession:
        try:
            # Import ToolEnv dynamically from the active split path
            split_dir = self.data_root / split
            if str(split_dir) not in sys.path and (split_dir / "zaic_tools").is_dir():
                sys.path.insert(0, str(split_dir))

            zaic_module = importlib.import_module("zaic_tools")
            tool_env_cls = getattr(zaic_module, "ToolEnv")
            raw_session = tool_env_cls(split, sample_id, root=str(self.data_root))
            return ZaicSandboxSession(raw_session)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize zaic_tools ToolEnv for '{split}:{sample_id}': {e}") from e


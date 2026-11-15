"""Shared project and dev-kit paths for one-off research scripts."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ZAIC_DATA_ROOT", PROJECT_ROOT)).expanduser().resolve()
DISTILLATION_DIR = PROJECT_ROOT / "training" / "llm" / "data" / "distillation"
SFT_DIR = PROJECT_ROOT / "training" / "llm" / "data" / "sft"

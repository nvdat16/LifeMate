"""Shared path definitions for the training utilities."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
DATA_ROOT = Path(os.environ.get("ZAIC_DATA_ROOT", PROJECT_ROOT)).expanduser().resolve()

LLM_DATA_ROOT = TRAINING_ROOT / "llm" / "data"
DISTILLATION_DIR = LLM_DATA_ROOT / "distillation"
SFT_DIR = LLM_DATA_ROOT / "sft"
LLM_OUTPUT_DIR = TRAINING_ROOT / "llm" / "outputs"
ASR_OUTPUT_DIR = TRAINING_ROOT / "asr" / "outputs"

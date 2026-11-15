#!/usr/bin/env python3
"""CLI entry point for the modular PhoWhisper LoRA training pipeline."""
import sys
from pathlib import Path

# Support direct invocation from any working directory, without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phowhisper.pipeline import main


if __name__ == '__main__':
    main()

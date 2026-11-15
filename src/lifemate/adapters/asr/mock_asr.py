"""Mock ASR adapter for instantaneous CPU execution and unit testing."""

from pathlib import Path
from typing import Optional, Union
from ...ports.asr_port import AbstractASREngine

class MockASREngine(AbstractASREngine):
    """
    Returns ground truth transcripts (when available) or simulated transcripts
    for rapid local development on CPU.
    """

    def __init__(self, fallback_text: str = ""):
        self.fallback_text = fallback_text
        self._transcript_cache: dict[str, str] = {}

    def register_transcript(self, audio_path: str, transcript: str):
        """Allows pre-populating transcripts by audio path."""
        self._transcript_cache[str(audio_path)] = transcript

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        key = str(audio_path).replace("\\", "/")
        for k, v in self._transcript_cache.items():
            if k in key or key in k:
                return v
        return self.fallback_text

    def transcribe_batch(self, audio_paths: list[Union[str, Path]]) -> list[str]:
        return [self.transcribe(p) for p in audio_paths]


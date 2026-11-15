"""Port interface for Speech-to-Text (ASR) engines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

class AbstractASREngine(ABC):
    """Contract for any ASR implementation (PhoWhisper, Whisper, Mock)."""

    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """
        Transcribe an audio file into a raw text transcript.
        
        Args:
            audio_path: Path to the .wav file.
            
        Returns:
            Transcribed text.
        """
        pass

    @abstractmethod
    def transcribe_batch(self, audio_paths: list[Union[str, Path]]) -> list[str]:
        """
        Transcribe multiple audio files efficiently.
        
        Args:
            audio_paths: List of paths to .wav files.
            
        Returns:
            List of transcribed texts in matching order.
        """
        pass


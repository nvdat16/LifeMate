"""Production ASR adapter using faster-whisper with PhoWhisper or Whisper models."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import soundfile as sf
from ...ports.asr_port import AbstractASREngine

class WhisperASREngine(AbstractASREngine):
    """Production ASR pipeline using faster-whisper."""

    def __init__(
        self,
        model_size_or_path: str = "mrdoanit/PhoWhisper-small-ct2-int8",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        data_root: Optional[Union[str, Path]] = None
    ):
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None
        self._model = None

    def _resolve_device_and_compute(self):
        """Auto detects CUDA availability and appropriate compute type."""
        dev = self.device
        ct = self.compute_type

        if dev == "auto":
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except ImportError:
                has_cuda = False
            dev = "cuda" if has_cuda else "cpu"

        if ct == "auto":
            ct = "float16" if dev == "cuda" else "int8"

        return dev, ct

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            dev, ct = self._resolve_device_and_compute()
            print(f"🎙️ Loading ASR Model '{self.model_size_or_path}' (device={dev}, compute_type={ct})...")
            self._model = WhisperModel(
                self.model_size_or_path,
                device=dev,
                compute_type=ct
            )
        return self._model

    def _resolve_audio_path(self, audio_path: Union[str, Path]) -> Path:
        p = Path(audio_path)
        if p.is_file():
            return p

        candidates = []
        if self.data_root:
            candidates.extend([self.data_root / "public_test" / p, self.data_root / "train" / p])
        candidates.extend([Path("public_test") / p, Path("train") / p, Path(".") / p])
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return p

    def _read_audio_array(self, path: Path) -> np.ndarray:
        """Reads audio as float32 mono numpy array, bypassing PyAV incompatibilities."""
        audio, sr = sf.read(str(path))
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        resolved_path = self._resolve_audio_path(audio_path)
        if not resolved_path.is_file():
            print(f"⚠️ Audio file not found at: {resolved_path} (original: {audio_path})")
            return ""

        model = self._get_model()
        audio_array = self._read_audio_array(resolved_path)
        segments, info = model.transcribe(
            audio_array,
            language="vi",
            beam_size=self.beam_size
        )
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(texts).strip()

    def transcribe_batch(self, audio_paths: list[Union[str, Path]]) -> list[str]:
        return [self.transcribe(p) for p in audio_paths]

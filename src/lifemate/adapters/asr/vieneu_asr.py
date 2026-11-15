"""VieNeu Cloud transcription adapter using its documented dubbing endpoint."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ...ports.asr_port import AbstractASREngine


class VieNeuASREngine(AbstractASREngine):
    """Return VieNeu's Whisper transcript from its audio dubbing endpoint.

    VieNeu currently exposes transcription as part of ``POST /api/v1/dub``;
    that endpoint also synthesizes dubbed audio and bills for the output audio.
    An API key and a valid ``voice_id`` are therefore required.
    """

    DEFAULT_ENDPOINT = "https://api.vieneu.io/api/v1/dub"
    MAX_AUDIO_BYTES = 10 * 1024 * 1024

    def __init__(
        self,
        voice_id: Optional[str],
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 120.0,
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("VIENEU_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "VIENEU_API_KEY is not set. Add it to the environment/Kaggle Secrets."
            )
        self.voice_id = (voice_id or "").strip()
        if not self.voice_id:
            raise ValueError("A VieNeu voice ID is required for the dubbing endpoint.")
        self.endpoint = endpoint
        self.timeout = timeout
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None

    def _resolve_audio_path(self, audio_path: Union[str, Path]) -> Path:
        path = Path(audio_path).expanduser()
        if path.is_file():
            return path.resolve()

        candidates: list[Path] = []
        if self.data_root:
            candidates.extend(
                [self.data_root / "public_test" / path, self.data_root / "train" / path]
            )
        candidates.extend([Path("public_test") / path, Path("train") / path])
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return path

    def _encode_multipart(self, path: Path) -> tuple[bytes, str]:
        boundary = f"----LifeMate{uuid.uuid4().hex}"
        filename = path.name.replace('"', "_")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks = [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            path.read_bytes(),
            f"\r\n--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="voiceId"\r\n\r\n{self.voice_id}\r\n'.encode(
                "utf-8"
            ),
            f"--{boundary}--\r\n".encode(),
        ]
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        path = self._resolve_audio_path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path} (resolved to {path})")
        if path.stat().st_size > self.MAX_AUDIO_BYTES:
            raise ValueError(
                f"VieNeu dubbing accepts audio up to 10 MB; {path.name} is "
                f"{path.stat().st_size / (1024 * 1024):.2f} MB."
            )

        body, content_type = self._encode_multipart(path)
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": content_type,
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"VieNeu STT returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach VieNeu STT: {exc.reason}") from exc

        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("VieNeu returned a non-JSON dubbing response.") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("transcript"), str):
            raise RuntimeError(
                "VieNeu dubbing response did not contain a transcript. "
                f"Response: {response_body[:500]}"
            )
        return payload["transcript"].strip()

    def transcribe_batch(self, audio_paths: list[Union[str, Path]]) -> list[str]:
        return [self.transcribe(path) for path in audio_paths]

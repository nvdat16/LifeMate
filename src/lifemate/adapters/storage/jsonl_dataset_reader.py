"""Adapter for reading samples from dataset JSONL files."""

import json
from pathlib import Path
from typing import Iterator, Optional, Union
from ...domain.entities import Sample, SampleContext, ConversationTurn, ToolCall
from ...ports.dataset_port import AbstractDatasetReader

class JsonlDatasetReader(AbstractDatasetReader):
    """Concrete reader loading samples from <split>/<split>.jsonl."""

    def __init__(self, data_root: Union[str, Path] = "."):
        self.data_root = Path(data_root)

    def _resolve_jsonl_path(self, split: str) -> Path:
        # Check standard layout: <root>/<split>/<split>.jsonl
        p = self.data_root / split / f"{split}.jsonl"
        if not p.is_file():
            # Fallback to direct path
            p = self.data_root / f"{split}.jsonl"
        if not p.is_file():
            raise FileNotFoundError(f"Cannot locate dataset file for split '{split}' at {p}")
        return p

    def read_samples(self, split: str, limit: Optional[int] = None) -> Iterator[Sample]:
        file_path = self._resolve_jsonl_path(split)
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield self._dict_to_sample(data)
                count += 1
                if limit is not None and count >= limit:
                    break

    def get_sample_by_id(self, split: str, sample_id: str) -> Sample:
        for s in self.read_samples(split):
            if s.id == sample_id:
                return s
        raise KeyError(f"Sample '{sample_id}' not found in split '{split}'")

    def _dict_to_sample(self, raw: dict) -> Sample:
        ctx_raw = raw.get("context", {})
        context = SampleContext(
            current_time=ctx_raw.get("current_time", ""),
            timezone=ctx_raw.get("timezone", "Asia/Ho_Chi_Minh"),
            location=ctx_raw.get("location", {}),
            user_profile=ctx_raw.get("user_profile", {}),
            installed_apps=ctx_raw.get("installed_apps", {})
        )

        history_turns: list[ConversationTurn] = []
        for turn_raw in raw.get("history", []):
            role = turn_raw.get("role", "user")
            t_calls = [
                ToolCall(name=c["name"], arguments=c.get("arguments", {}))
                for c in turn_raw.get("tool_calls", [])
            ]
            history_turns.append(
                ConversationTurn(
                    role=role,
                    audio=turn_raw.get("audio"),
                    audio_text=turn_raw.get("audio_text"),
                    text=turn_raw.get("text"),
                    tool_calls=t_calls,
                    tool_results=turn_raw.get("tool_results", [])
                )
            )

        return Sample(
            id=raw["id"],
            audio=raw.get("audio", ""),
            context=context,
            history=history_turns,
            audio_text=raw.get("audio_text"),
            label=raw.get("label")
        )


"""Adapter for exporting results to JSONL format."""

import json
from pathlib import Path
from typing import Union
from ...domain.entities import ExecutionResult
from ...ports.exporter_port import AbstractSubmissionExporter

class JsonlSubmissionExporter(AbstractSubmissionExporter):
    """Exports predictions strictly following Zalo AI Challenge specifications."""

    def export(
        self,
        results: list[ExecutionResult],
        output_path: Union[str, Path]
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for item in results:
                line_obj = item.to_submission_dict()
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

        return out


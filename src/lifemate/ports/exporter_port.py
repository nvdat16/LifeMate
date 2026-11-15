"""Port interface for exporting results to submission file."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from ..domain.entities import ExecutionResult

class AbstractSubmissionExporter(ABC):
    """Contract for persisting predictions to submission.jsonl."""

    @abstractmethod
    def export(
        self,
        results: list[ExecutionResult],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Exports a list of execution results to the target JSONL file.

        Args:
            results: List of ExecutionResult objects.
            output_path: Destination path.

        Returns:
            Resolved Path of the written file.
        """
        pass


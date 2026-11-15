"""Port interface for reading dataset samples."""

from abc import ABC, abstractmethod
from typing import Iterator
from ..domain.entities import Sample

class AbstractDatasetReader(ABC):
    """Contract for dataset loaders (JSONL, streaming, filtered)."""

    @abstractmethod
    def read_samples(self, split: str, limit: int | None = None) -> Iterator[Sample]:
        """
        Yields domain Sample entities from the specified split folder.

        Args:
            split: 'train' or 'public_test' or 'private_test'
            limit: Maximum samples to yield (None for all).
        """
        pass

    @abstractmethod
    def get_sample_by_id(self, split: str, sample_id: str) -> Sample:
        """Retrieves a single sample entity by its unique ID."""
        pass


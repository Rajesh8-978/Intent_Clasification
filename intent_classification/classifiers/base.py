from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from intent_classification.models import ClassificationPrediction


class IIntentClassifier(ABC):
    """Async boundary between business orchestration and model inference."""

    @abstractmethod
    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        """Rank the supplied labels for the given text."""

        raise NotImplementedError

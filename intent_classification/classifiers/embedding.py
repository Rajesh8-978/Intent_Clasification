from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from intent_classification.classifiers._result_normalizer import normalize_provider_result
from intent_classification.classifiers.base import IIntentClassifier
from intent_classification.models import ClassificationPrediction


class EmbeddingIntentClassifier(IIntentClassifier):
    """Embedding-based open-label classifier adapter."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name
        self._classifier_by_labels: dict[tuple[str, ...], Any] = {}

    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        classifier = await self._get_classifier(tuple(labels))
        result = await asyncio.to_thread(classifier.predict, text)
        prediction = normalize_provider_result(result, labels)
        return ClassificationPrediction(
            label=prediction.label,
            confidence=prediction.confidence,
            top_predictions=prediction.top_predictions[:top_k],
        )

    async def _get_classifier(self, labels: tuple[str, ...]) -> Any:
        if labels not in self._classifier_by_labels:
            self._classifier_by_labels[labels] = await asyncio.to_thread(self._create_classifier, labels)
        return self._classifier_by_labels[labels]

    def _create_classifier(self, labels: tuple[str, ...]) -> Any:
        try:
            from open_intent_classifier.embedder import StaticLabelsEmbeddingClassifier
        except ImportError as exc:
            raise RuntimeError(
                "EmbeddingIntentClassifier requires the 'open-intent-classifier' package."
            ) from exc

        if self._model_name:
            return StaticLabelsEmbeddingClassifier(list(labels), model_name=self._model_name)
        return StaticLabelsEmbeddingClassifier(list(labels))

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from intent_classification.classifiers._result_normalizer import normalize_provider_result
from intent_classification.classifiers.base import IIntentClassifier
from intent_classification.models import ClassificationPrediction


class SmolLM2IntentClassifier(IIntentClassifier):
    """SmolLM2 adapter exposed by open-intent-classifier."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name
        self._classifier: Any | None = None

    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        classifier = await self._get_classifier()
        result = await asyncio.to_thread(classifier.predict, text, list(labels))
        prediction = normalize_provider_result(result, labels)
        return ClassificationPrediction(
            label=prediction.label,
            confidence=prediction.confidence,
            top_predictions=prediction.top_predictions[:top_k],
        )

    async def _get_classifier(self) -> Any:
        if self._classifier is None:
            self._classifier = await asyncio.to_thread(self._create_classifier)
        return self._classifier

    def _create_classifier(self) -> Any:
        try:
            from open_intent_classifier.model import SmolLm2Classifier
        except ImportError as exc:
            raise RuntimeError(
                "SmolLM2IntentClassifier requires the 'open-intent-classifier' package."
            ) from exc

        if self._model_name:
            return SmolLm2Classifier(model_name=self._model_name)
        return SmolLm2Classifier()

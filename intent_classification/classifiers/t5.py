from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from intent_classification.classifiers._result_normalizer import normalize_provider_result
from intent_classification.classifiers.base import IIntentClassifier
from intent_classification.models import ClassificationPrediction


class T5IntentClassifier(IIntentClassifier):
    """Open Intent Classifier T5 adapter."""

    _FLAN_T5_BASE = "flan-t5-base"
    _SMALL = "small"

    def __init__(self, model: str = _SMALL, device: str | None = None) -> None:
        self._model_name = model
        self._device = device
        self._classifier: Any | None = None

    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        classifier = await self._get_classifier()
        # The package's default generation length can truncate longer labels.
        result = await asyncio.to_thread(
            classifier.predict,
            text,
            list(labels),
            max_new_tokens=32,
        )
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
            from open_intent_classifier.consts import INTENT_CLASSIFIER_248M_FLAN_T5_BASE
            from open_intent_classifier.model import IntentClassifier
        except ImportError as exc:
            raise RuntimeError(
                "T5IntentClassifier requires the 'open-intent-classifier' package."
            ) from exc

        kwargs = {}
        if self._device:
            kwargs["device"] = self._device

        if self._model_name.lower() in {self._SMALL, "t5-small", "default"}:
            return IntentClassifier(**kwargs)
        if self._model_name.lower() in {self._FLAN_T5_BASE, "base", "flan"}:
            return IntentClassifier(INTENT_CLASSIFIER_248M_FLAN_T5_BASE, **kwargs)

        return IntentClassifier(self._model_name, **kwargs)

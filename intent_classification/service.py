from __future__ import annotations

import logging
from collections.abc import Sequence

from intent_classification.classifiers import IIntentClassifier
from intent_classification.config import IntentClassifierOptions
from intent_classification.labels import IIntentLabelProvider, StaticIntentLabelProvider
from intent_classification.models import IntentClassificationResult, PredictionCandidate
from intent_classification.text_cleaning import clean_email_text


class IntentClassificationService:
    """Coordinates text cleanup, classifier execution, and EntityType mapping."""

    def __init__(
        self,
        classifier: IIntentClassifier,
        label_provider: IIntentLabelProvider | None = None,
        options: IntentClassifierOptions | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._classifier = classifier
        self._label_provider = label_provider or StaticIntentLabelProvider()
        self._options = options or IntentClassifierOptions()
        self._logger = logger or logging.getLogger(__name__)

    async def classify_email_text(
        self,
        email_text: str,
        labels: Sequence[str] | None = None,
    ) -> IntentClassificationResult:
        normalized_text = clean_email_text(email_text)
        if not normalized_text:
            raise ValueError("Email text is empty after cleaning.")

        available_labels = tuple(labels or self._label_provider.get_labels())
        if not available_labels:
            raise ValueError("At least one intent label is required.")

        try:
            prediction = await self._classifier.predict(
                normalized_text,
                available_labels,
                top_k=self._options.top_k,
            )
            if prediction.label not in available_labels:
                raise ValueError(
                    f"Classifier returned label {prediction.label!r}, which is not in the configured labels."
                )

            entity_type = self._label_provider.map_to_entity_type(prediction.label)
            top_predictions = self._normalize_top_predictions(
                prediction.label,
                prediction.confidence,
                prediction.top_predictions,
                available_labels,
            )

            self._logger.info(
                "Classified email intent.",
                extra={
                    "intent_label": prediction.label,
                    "entity_type": entity_type.value,
                    "confidence": prediction.confidence,
                },
            )
            return IntentClassificationResult(
                entity_type=entity_type,
                intent_label=prediction.label,
                confidence=prediction.confidence,
                top_predictions=top_predictions,
            )
        except Exception:
            self._logger.exception("Email intent classification failed.")
            raise

    def _normalize_top_predictions(
        self,
        predicted_label: str,
        confidence: float | None,
        candidates: Sequence[PredictionCandidate],
        available_labels: Sequence[str],
    ) -> tuple[PredictionCandidate, ...]:
        filtered = tuple(
            candidate
            for candidate in candidates
            if candidate.label in available_labels
        )
        if filtered:
            return filtered[: self._options.top_k]
        return (PredictionCandidate(label=predicted_label, score=confidence),)

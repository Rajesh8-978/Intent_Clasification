from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from intent_classification.classifiers.base import IIntentClassifier
from intent_classification.label_definitions import build_label_hypothesis
from intent_classification.models import ClassificationPrediction, PredictionCandidate


DEFAULT_ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class ZeroShotIntentClassifier(IIntentClassifier):
    """Rank business intents with a local Natural Language Inference model.

    The model evaluates each detailed label description as a hypothesis about
    the email. No model fine-tuning or external classification API is required.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ZERO_SHOT_MODEL,
        device: str | int | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None

    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        """Return the highest-ranked label and up to ``top_k`` candidates."""

        if not labels:
            raise ValueError("At least one intent label is required.")

        classifier = await self._get_pipeline()
        # Detailed hypotheses give the general-purpose model the business meaning
        # of each short label, while this mapping restores the public label names.
        hypothesis_to_label = {
            build_label_hypothesis(label): label
            for label in labels
        }
        # Transformers inference is synchronous and CPU-heavy. Running it in a
        # worker thread keeps callers' asyncio event loops responsive.
        result = await asyncio.to_thread(
            classifier,
            text,
            candidate_labels=list(hypothesis_to_label),
            hypothesis_template="{}",
            multi_label=False,  # Exactly one primary intent should win.
        )
        candidates = self._normalize_result(result, hypothesis_to_label)
        selected = candidates[:top_k]
        if not selected:
            raise ValueError("The zero-shot classifier returned no valid labels.")

        return ClassificationPrediction(
            label=selected[0].label,
            confidence=selected[0].score,
            top_predictions=selected,
        )

    async def _get_pipeline(self) -> Any:
        """Load the large model once, on the first classification request."""

        if self._pipeline is None:
            self._pipeline = await asyncio.to_thread(self._create_pipeline)
        return self._pipeline

    def _create_pipeline(self) -> Any:
        """Create the Hugging Face zero-shot pipeline for local inference."""

        pipeline_factory = self._pipeline_factory
        if pipeline_factory is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "ZeroShotIntentClassifier requires the 'transformers' and 'torch' packages."
                ) from exc
            pipeline_factory = pipeline

        options: dict[str, Any] = {
            "task": "zero-shot-classification",
            "model": self._model_name,
        }
        if self._device is not None:
            options["device"] = self._device
        return pipeline_factory(**options)

    @staticmethod
    def _normalize_result(
        result: Any,
        hypothesis_to_label: dict[str, str],
    ) -> tuple[PredictionCandidate, ...]:
        """Convert the Transformers response into application result objects."""

        if not isinstance(result, dict):
            raise ValueError(f"Zero-shot classifier returned an unsupported result: {result!r}")

        ranked_hypotheses = result.get("labels")
        scores = result.get("scores")
        if not isinstance(ranked_hypotheses, list) or not isinstance(scores, list):
            raise ValueError("Zero-shot classifier result must contain label and score lists.")

        candidates: list[PredictionCandidate] = []
        for hypothesis, score in zip(ranked_hypotheses, scores):
            label = hypothesis_to_label.get(str(hypothesis))
            if label is None:
                continue
            try:
                normalized_score = float(score)
            except (TypeError, ValueError):
                normalized_score = None
            candidates.append(PredictionCandidate(label=label, score=normalized_score))
        return tuple(candidates)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intent_classification.entity_types import EntityType


@dataclass(frozen=True)
class PredictionCandidate:
    """One ranked intent returned by the model."""

    label: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score}


@dataclass(frozen=True)
class ClassificationPrediction:
    """Internal result produced by an intent classifier implementation."""

    label: str
    confidence: float | None = None
    top_predictions: tuple[PredictionCandidate, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IntentClassificationResult:
    """Validated, workflow-ready classification returned to callers."""

    entity_type: EntityType
    intent_label: str
    confidence: float | None = None
    top_predictions: tuple[PredictionCandidate, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entityType": self.entity_type.value,
            "intentLabel": self.intent_label,
            "confidence": self.confidence,
            "topPredictions": [candidate.to_dict() for candidate in self.top_predictions],
        }

"""Business intent classification for extracted email text."""

from intent_classification.classifiers import ZeroShotIntentClassifier
from intent_classification.config import IntentClassifierOptions
from intent_classification.entity_types import EntityType
from intent_classification.labels import IIntentLabelProvider, StaticIntentLabelProvider
from intent_classification.models import (
    ClassificationPrediction,
    IntentClassificationResult,
    PredictionCandidate,
)
from intent_classification.service import IntentClassificationService

__all__ = [
    "ClassificationPrediction",
    "EntityType",
    "IIntentLabelProvider",
    "IntentClassificationResult",
    "IntentClassifierOptions",
    "IntentClassificationService",
    "PredictionCandidate",
    "StaticIntentLabelProvider",
    "ZeroShotIntentClassifier",
]

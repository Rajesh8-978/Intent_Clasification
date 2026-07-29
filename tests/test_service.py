import logging
import unittest
from collections.abc import Sequence

from intent_classification.classifiers import IIntentClassifier
from intent_classification.config import IntentClassifierOptions
from intent_classification.entity_types import EntityType
from intent_classification.labels import StaticIntentLabelProvider
from intent_classification.models import ClassificationPrediction, PredictionCandidate
from intent_classification.service import IntentClassificationService


class FakeClassifier(IIntentClassifier):
    def __init__(self, prediction: ClassificationPrediction) -> None:
        self.prediction = prediction
        self.last_text = ""
        self.last_labels: Sequence[str] = ()
        self.last_top_k = 0

    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        self.last_text = text
        self.last_labels = labels
        self.last_top_k = top_k
        return self.prediction


class IntentClassificationServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logger = logging.getLogger(f"{__name__}.{self.id()}")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False

    async def test_classifies_and_maps_result(self):
        classifier = FakeClassifier(
            ClassificationPrediction(
                label="Statement of Affairs",
                confidence=0.94,
                top_predictions=(
                    PredictionCandidate("Statement of Affairs", 0.94),
                    PredictionCandidate("General Case", 0.72),
                    PredictionCandidate("Email", 0.64),
                ),
            )
        )
        service = IntentClassificationService(
            classifier,
            options=IntentClassifierOptions(top_k=2),
            logger=self.logger,
        )

        result = await service.classify_email_text("\n  Attached is my statement of affairs form.  ")

        self.assertEqual(result.entity_type, EntityType.STATEMENT_OF_AFFAIRS)
        self.assertEqual(result.intent_label, "Statement of Affairs")
        self.assertEqual(result.confidence, 0.94)
        self.assertEqual(len(result.top_predictions), 2)
        self.assertEqual(classifier.last_text, "Attached is my statement of affairs form.")
        self.assertEqual(classifier.last_top_k, 2)
        self.assertEqual(
            result.to_dict(),
            {
                "entityType": "StatementOfAffairs",
                "intentLabel": "Statement of Affairs",
                "confidence": 0.94,
                "topPredictions": [
                    {"label": "Statement of Affairs", "score": 0.94},
                    {"label": "General Case", "score": 0.72},
                ],
            },
        )

    async def test_supports_custom_label_provider(self):
        classifier = FakeClassifier(ClassificationPrediction(label="Custom Intent", confidence=0.8))
        service = IntentClassificationService(
            classifier,
            label_provider=StaticIntentLabelProvider({"Custom Intent": EntityType.MISC_REQUEST}),
            logger=self.logger,
        )

        result = await service.classify_email_text("Please handle this custom request.")

        self.assertEqual(result.entity_type, EntityType.MISC_REQUEST)
        self.assertEqual(result.intent_label, "Custom Intent")

    async def test_rejects_unknown_classifier_label(self):
        classifier = FakeClassifier(ClassificationPrediction(label="Unknown Label"))
        service = IntentClassificationService(classifier, logger=self.logger)

        with self.assertRaises(ValueError):
            await service.classify_email_text("Attached are travel dates.")

    async def test_rejects_empty_email_text(self):
        classifier = FakeClassifier(ClassificationPrediction(label="Email"))
        service = IntentClassificationService(classifier, logger=self.logger)

        with self.assertRaises(ValueError):
            await service.classify_email_text(" \n\t ")


if __name__ == "__main__":
    unittest.main()

import unittest

from intent_classification.classifiers.zero_shot import ZeroShotIntentClassifier
from intent_classification.entity_types import DEFAULT_INTENT_LABELS
from intent_classification.label_definitions import LABEL_DESCRIPTIONS, build_label_hypothesis


class FakeZeroShotPipeline:
    def __init__(self) -> None:
        self.text = ""
        self.candidate_labels = []
        self.hypothesis_template = ""
        self.multi_label = None

    def __call__(
        self,
        text,
        *,
        candidate_labels,
        hypothesis_template,
        multi_label,
    ):
        self.text = text
        self.candidate_labels = candidate_labels
        self.hypothesis_template = hypothesis_template
        self.multi_label = multi_label
        return {
            "sequence": text,
            "labels": [
                build_label_hypothesis("Vehicle Documents"),
                build_label_hypothesis("General Case"),
                build_label_hypothesis("Email"),
            ],
            "scores": [0.91, 0.06, 0.03],
        }


class ZeroShotIntentClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_ranks_described_labels_and_preserves_business_names(self):
        fake_pipeline = FakeZeroShotPipeline()
        factory_options = {}

        def pipeline_factory(**kwargs):
            factory_options.update(kwargs)
            return fake_pipeline

        classifier = ZeroShotIntentClassifier(
            model_name="local-test-model",
            device="cpu",
            pipeline_factory=pipeline_factory,
        )

        prediction = await classifier.predict(
            "Please provide the COE renewal and registration details for vehicle SMD4125Y.",
            ("Vehicle Documents", "General Case", "Email"),
            top_k=2,
        )

        self.assertEqual(prediction.label, "Vehicle Documents")
        self.assertEqual(prediction.confidence, 0.91)
        self.assertEqual(
            [candidate.label for candidate in prediction.top_predictions],
            ["Vehicle Documents", "General Case"],
        )
        self.assertEqual(factory_options["task"], "zero-shot-classification")
        self.assertEqual(factory_options["model"], "local-test-model")
        self.assertEqual(factory_options["device"], "cpu")
        self.assertEqual(fake_pipeline.hypothesis_template, "{}")
        self.assertFalse(fake_pipeline.multi_label)
        self.assertIn("COE renewal", fake_pipeline.candidate_labels[0])

    async def test_rejects_result_without_ranked_candidates(self):
        classifier = ZeroShotIntentClassifier(
            pipeline_factory=lambda **_: lambda *args, **kwargs: {
                "labels": [],
                "scores": [],
            }
        )

        with self.assertRaises(ValueError):
            await classifier.predict("Vehicle details", ("Vehicle Documents",))


class LabelDescriptionTests(unittest.TestCase):
    def test_every_default_label_has_a_description(self):
        self.assertEqual(set(DEFAULT_INTENT_LABELS), set(LABEL_DESCRIPTIONS))

    def test_custom_label_gets_a_generic_hypothesis(self):
        self.assertEqual(
            build_label_hypothesis("Custom Intent"),
            "The primary business intent of this message is Custom Intent.",
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from intent_classification.classifiers._result_normalizer import normalize_provider_result


class ResultNormalizerTests(unittest.TestCase):
    def test_normalizes_t5_marker_and_truncated_label(self):
        result = normalize_provider_result(
            "Email # Miscel",
            ("Email", "Miscellaneous Request", "Vehicle Documents"),
        )

        self.assertEqual(result.label, "Miscellaneous Request")

    def test_preserves_unknown_label_for_service_validation(self):
        result = normalize_provider_result("Unknown Label", ("Email",))

        self.assertEqual(result.label, "Unknown Label")

    def test_normalizes_dict_label_and_preserves_zero_score(self):
        result = normalize_provider_result(
            {
                "label": "email",
                "confidence": 0.4,
                "topPredictions": [
                    {"label": "Email", "score": 0.0, "confidence": 0.9},
                ],
            },
            ("Email", "Miscellaneous Request"),
        )

        self.assertEqual(result.label, "Email")
        self.assertEqual(result.top_predictions[0].score, 0.0)

    def test_supports_generator_labels_and_missing_score(self):
        labels = (label for label in ("Email", "Miscellaneous Request"))

        result = normalize_provider_result({"label": "email"}, labels)

        self.assertEqual(result.label, "Email")
        self.assertIsNone(result.confidence)

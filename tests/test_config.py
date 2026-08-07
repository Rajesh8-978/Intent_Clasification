import unittest

from intent_classification.config import IntentClassifierOptions


class IntentClassifierOptionsTests(unittest.TestCase):
    def test_reads_nested_configuration(self):
        options = IntentClassifierOptions.from_dict(
            {
                "IntentClassifier": {
                    "Model": "local-test-model",
                    "TopK": 5,
                    "Device": "cpu",
                }
            }
        )

        self.assertEqual(options.model, "local-test-model")
        self.assertEqual(options.top_k, 5)
        self.assertEqual(options.device, "cpu")

    def test_uses_local_zero_shot_defaults(self):
        options = IntentClassifierOptions()

        self.assertEqual(
            options.model,
            "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        )
        self.assertEqual(options.top_k, 3)
        self.assertIsNone(options.device)

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            IntentClassifierOptions.from_dict({"IntentClassifier": {"TopK": 0}})

        with self.assertRaises(ValueError):
            IntentClassifierOptions(top_k=0)

    def test_rejects_invalid_model_and_device(self):
        with self.assertRaises(ValueError):
            IntentClassifierOptions(model="  ")

        with self.assertRaises(ValueError):
            IntentClassifierOptions(device={"name": "cpu"})


if __name__ == "__main__":
    unittest.main()

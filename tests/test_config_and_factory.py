import unittest

from intent_classification.classifiers import (
    EmbeddingIntentClassifier,
    OpenAIIntentClassifier,
    SmolLM2IntentClassifier,
    T5IntentClassifier,
)
from intent_classification.config import IntentClassifierOptions
from intent_classification.factory import create_intent_classifier


class ConfigAndFactoryTests(unittest.TestCase):
    def test_reads_nested_intent_classifier_config(self):
        options = IntentClassifierOptions.from_dict(
            {
                "IntentClassifier": {
                    "Provider": "T5",
                    "Model": "flan-t5-base",
                    "TopK": 5,
                    "ProviderOptions": {"device": "cpu"},
                }
            }
        )

        self.assertEqual(options.provider, "T5")
        self.assertEqual(options.model, "flan-t5-base")
        self.assertEqual(options.top_k, 5)
        self.assertEqual(options.provider_options["device"], "cpu")

    def test_reads_string_boolean_config(self):
        options = IntentClassifierOptions.from_dict(
            {"IntentClassifier": {"FailOnUnknownLabel": "false"}}
        )

        self.assertFalse(options.fail_on_unknown_label)

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            IntentClassifierOptions.from_dict({"IntentClassifier": {"TopK": 0}})

        with self.assertRaises(ValueError):
            IntentClassifierOptions(top_k=0)

    def test_rejects_invalid_direct_options(self):
        with self.assertRaises(ValueError):
            IntentClassifierOptions(provider_options=[])

        with self.assertRaises(ValueError):
            IntentClassifierOptions(fail_on_unknown_label="sometimes")

    def test_factory_selects_t5_provider(self):
        classifier = create_intent_classifier(IntentClassifierOptions(provider="T5", model="small"))

        self.assertIsInstance(classifier, T5IntentClassifier)

    def test_factory_selects_other_providers(self):
        self.assertIsInstance(
            create_intent_classifier(IntentClassifierOptions(provider="Embedding")),
            EmbeddingIntentClassifier,
        )
        self.assertIsInstance(
            create_intent_classifier(IntentClassifierOptions(provider="OpenAI", model="gpt-4.1-mini")),
            OpenAIIntentClassifier,
        )
        self.assertIsInstance(
            create_intent_classifier(IntentClassifierOptions(provider="SmolLM2")),
            SmolLM2IntentClassifier,
        )

    def test_factory_does_not_pass_t5_default_model_to_other_providers(self):
        embedding = create_intent_classifier(IntentClassifierOptions(provider="Embedding"))
        openai = create_intent_classifier(IntentClassifierOptions(provider="OpenAI"))
        smollm2 = create_intent_classifier(IntentClassifierOptions(provider="SmolLM2"))

        self.assertIsNone(embedding._model_name)
        self.assertEqual(openai._model, "gpt-4.1-mini")
        self.assertIsNone(smollm2._model_name)

    def test_factory_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            create_intent_classifier(IntentClassifierOptions(provider="Unknown"))


if __name__ == "__main__":
    unittest.main()

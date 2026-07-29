from __future__ import annotations

from intent_classification.classifiers import (
    EmbeddingIntentClassifier,
    IIntentClassifier,
    OpenAIIntentClassifier,
    SmolLM2IntentClassifier,
    T5IntentClassifier,
)
from intent_classification.config import DEFAULT_MODEL, IntentClassifierOptions


OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"


def create_intent_classifier(options: IntentClassifierOptions) -> IIntentClassifier:
    provider = options.provider.strip().lower()
    model = options.model.strip()
    provider_options = dict(options.provider_options)

    if provider == "t5":
        return T5IntentClassifier(
            model=model,
            device=provider_options.get("device"),
        )
    if provider in {"embedding", "embeddings"}:
        return EmbeddingIntentClassifier(model_name=_provider_model(model))
    if provider == "openai":
        return OpenAIIntentClassifier(
            model=_provider_model(model, default=OPENAI_DEFAULT_MODEL),
            api_key=provider_options.get("api_key"),
        )
    if provider in {"smollm2", "smol-lm2", "smol_lm2"}:
        return SmolLM2IntentClassifier(model_name=_provider_model(model))

    raise ValueError(f"Unsupported intent classifier provider: {options.provider!r}")


def _provider_model(model: str, default: str | None = None) -> str | None:
    if not model or model == DEFAULT_MODEL:
        return default
    return model

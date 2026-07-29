from __future__ import annotations

import json
from collections.abc import Sequence

from intent_classification.classifiers.base import IIntentClassifier
from intent_classification.models import ClassificationPrediction, PredictionCandidate


class OpenAIIntentClassifier(IIntentClassifier):
    """OpenAI-backed classifier for deployments that prefer an API model."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client = None

    async def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        top_k: int = 3,
    ) -> ClassificationPrediction:
        client = self._get_client()
        response = await client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Classify the business purpose of an extracted email. "
                        "Return JSON only with keys label, confidence, and topPredictions. "
                        "The label must exactly match one of the provided labels."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "labels": list(labels),
                            "topK": top_k,
                            "emailText": text,
                        }
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "intent_classification",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string", "enum": list(labels)},
                            "confidence": {"type": ["number", "null"]},
                            "topPredictions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {"type": "string", "enum": list(labels)},
                                        "score": {"type": ["number", "null"]},
                                    },
                                    "required": ["label", "score"],
                                },
                            },
                        },
                        "required": ["label", "confidence", "topPredictions"],
                    },
                    "strict": True,
                }
            },
        )
        payload = json.loads(response.output_text)
        return ClassificationPrediction(
            label=payload["label"],
            confidence=payload.get("confidence"),
            top_predictions=tuple(
                PredictionCandidate(label=item["label"], score=item.get("score"))
                for item in payload.get("topPredictions", [])
            ),
        )

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAIIntentClassifier requires the 'openai' package.") from exc

        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

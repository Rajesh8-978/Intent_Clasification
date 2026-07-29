from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from intent_classification.models import ClassificationPrediction, PredictionCandidate


def normalize_provider_result(result: Any, labels: Iterable[str]) -> ClassificationPrediction:
    """Convert common provider return shapes into ClassificationPrediction."""

    label_tuple = tuple(labels)
    label_set = set(label_tuple)

    if isinstance(result, ClassificationPrediction):
        return result

    if isinstance(result, str):
        label = _canonicalize_label(result, label_tuple)
        return ClassificationPrediction(
            label=label,
            top_predictions=(PredictionCandidate(label),),
        )

    if isinstance(result, dict):
        label = _first_present(result, ("label", "intent", "prediction", "predicted_label"))
        score = _first_present_or_none(result, ("score", "confidence", "probability"))
        candidates = _extract_candidates(result, label_tuple)
        normalized_label = _canonicalize_label(str(label), label_tuple)
        return ClassificationPrediction(
            label=normalized_label,
            confidence=_as_float(score),
            top_predictions=tuple(candidates) or (PredictionCandidate(normalized_label, _as_float(score)),),
        )

    label = getattr(result, "label", None) or getattr(result, "intent", None) or getattr(result, "prediction", None)
    if label is not None:
        score = _first_non_none(
            getattr(result, "score", None),
            getattr(result, "confidence", None),
        )
        normalized_label = _canonicalize_label(str(label), label_tuple)
        return ClassificationPrediction(
            label=normalized_label,
            confidence=_as_float(score),
            top_predictions=(PredictionCandidate(normalized_label, _as_float(score)),),
        )

    result_text = str(result)
    if result_text in label_set:
        return ClassificationPrediction(
            label=result_text,
            top_predictions=(PredictionCandidate(label=result_text),),
        )

    raise ValueError(f"Classifier returned an unsupported prediction result: {result!r}")


def _canonicalize_label(raw_label: str, labels: tuple[str, ...]) -> str:
    """Handle provider output decorated with prompt markers or truncated text."""

    if raw_label in labels:
        return raw_label

    cleaned = " ".join(raw_label.strip().split())
    for candidate in labels:
        if cleaned.casefold() == candidate.casefold():
            return candidate

    # Open Intent Classifier can return the prompt's ``# Label`` format.
    marker_match = re.search(r"#\s*(.+)$", cleaned)
    marker_text = marker_match.group(1).strip() if marker_match else cleaned
    marker_text = re.sub(r"[^\w\s]", " ", marker_text)
    marker_text = " ".join(marker_text.split()).casefold()
    if marker_text:
        partial_matches = [
            label
            for label in labels
            if label.casefold().startswith(marker_text)
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]

    return raw_label


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    raise ValueError(f"Classifier result is missing one of: {', '.join(keys)}")


def _first_present_or_none(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _extract_candidates(result: dict[str, Any], labels: tuple[str, ...]) -> list[PredictionCandidate]:
    raw_candidates = (
        result.get("top_predictions")
        or result.get("topPredictions")
        or result.get("candidates")
        or result.get("scores")
        or []
    )
    label_set = set(labels)
    candidates: list[PredictionCandidate] = []

    if isinstance(raw_candidates, dict):
        raw_candidates = [{"label": label, "score": score} for label, score in raw_candidates.items()]

    if not isinstance(raw_candidates, list):
        return candidates

    for candidate in raw_candidates:
        if isinstance(candidate, str):
            if candidate in label_set:
                candidates.append(PredictionCandidate(label=candidate))
            continue

        if not isinstance(candidate, dict):
            continue

        label = candidate.get("label") or candidate.get("intent")
        if label is None:
            continue
        candidates.append(
            PredictionCandidate(
                label=_canonicalize_label(str(label), labels),
                score=_as_float(_first_non_none(candidate.get("score"), candidate.get("confidence"))),
            )
        )

    return candidates


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

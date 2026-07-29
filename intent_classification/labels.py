from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from intent_classification.entity_types import DEFAULT_INTENT_LABELS, EntityType, map_label_to_entity_type


class IIntentLabelProvider(ABC):
    @abstractmethod
    def get_labels(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def map_to_entity_type(self, label: str) -> EntityType:
        raise NotImplementedError


class StaticIntentLabelProvider(IIntentLabelProvider):
    def __init__(self, label_to_entity_type: dict[str, EntityType] | None = None) -> None:
        self._label_to_entity_type = label_to_entity_type

    def get_labels(self) -> Sequence[str]:
        if self._label_to_entity_type is not None:
            return tuple(self._label_to_entity_type.keys())
        return DEFAULT_INTENT_LABELS

    def map_to_entity_type(self, label: str) -> EntityType:
        if self._label_to_entity_type is not None:
            try:
                return self._label_to_entity_type[label]
            except KeyError as exc:
                raise ValueError(f"Intent label is not mapped to an EntityType: {label!r}") from exc
        return map_label_to_entity_type(label)

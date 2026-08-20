from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from intent_classification.entity_types import DEFAULT_INTENT_LABELS, EntityType, map_label_to_entity_type


class IIntentLabelProvider(ABC):
    """Source of allowed labels and their application mappings."""

    @abstractmethod
    def get_labels(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def map_to_entity_type(self, label: str) -> EntityType:
        raise NotImplementedError


class StaticIntentLabelProvider(IIntentLabelProvider):
    """Provide the default taxonomy or a caller-supplied static taxonomy."""

    def __init__(self, label_to_entity_type: dict[str, EntityType] | None = None) -> None:
        self._label_to_entity_type = label_to_entity_type

    def get_labels(self) -> Sequence[str]:
        """Return the labels that the classifier is allowed to select."""

        if self._label_to_entity_type is not None:
            return tuple(self._label_to_entity_type.keys())
        return DEFAULT_INTENT_LABELS

    def map_to_entity_type(self, label: str) -> EntityType:
        """Map a selected label to the corresponding application entity."""

        if self._label_to_entity_type is not None:
            try:
                return self._label_to_entity_type[label]
            except KeyError as exc:
                raise ValueError(f"Intent label is not mapped to an EntityType: {label!r}") from exc
        return map_label_to_entity_type(label)

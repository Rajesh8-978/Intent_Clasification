from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from intent_classification.classifiers.zero_shot import DEFAULT_ZERO_SHOT_MODEL

DEFAULT_MODEL = DEFAULT_ZERO_SHOT_MODEL
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class IntentClassifierOptions:
    """Validated runtime settings for the local zero-shot classifier."""

    model: str = DEFAULT_MODEL
    top_k: int = DEFAULT_TOP_K
    device: str | int | None = None

    def __post_init__(self) -> None:
        # Validate once at construction so the model layer receives safe values.
        top_k = int(self.top_k)
        if top_k < 1:
            raise ValueError("IntentClassifier.TopK must be at least 1.")
        model = str(self.model).strip()
        if not model:
            raise ValueError("IntentClassifier.Model must not be empty.")
        if self.device is not None and (
            isinstance(self.device, bool) or not isinstance(self.device, (str, int))
        ):
            raise ValueError("IntentClassifier.Device must be a device name or index.")

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "top_k", top_k)
        if isinstance(self.device, str):
            normalized_device = self.device.strip()
            object.__setattr__(self, "device", normalized_device or None)

    @classmethod
    def from_dict(cls, settings: Mapping[str, Any] | None) -> "IntentClassifierOptions":
        """Create options from either a root mapping or IntentClassifier section."""

        if not settings:
            return cls()

        section = settings.get("IntentClassifier", settings)
        if not isinstance(section, Mapping):
            raise ValueError("IntentClassifier configuration must be an object.")

        return cls(
            model=section.get("Model", DEFAULT_MODEL),
            top_k=section.get("TopK", DEFAULT_TOP_K),
            device=section.get("Device"),
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_PROVIDER = "T5"
DEFAULT_MODEL = "small"
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class IntentClassifierOptions:
    """Configuration for selecting and tuning an intent classifier."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    top_k: int = DEFAULT_TOP_K
    fail_on_unknown_label: bool = True
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        top_k = int(self.top_k)
        if top_k < 1:
            raise ValueError("IntentClassifier.TopK must be at least 1.")
        if not isinstance(self.provider_options, Mapping):
            raise ValueError("IntentClassifier.ProviderOptions must be an object.")

        object.__setattr__(self, "provider", str(self.provider))
        object.__setattr__(self, "model", str(self.model))
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(
            self,
            "fail_on_unknown_label",
            _as_bool(self.fail_on_unknown_label, "IntentClassifier.FailOnUnknownLabel"),
        )
        object.__setattr__(self, "provider_options", dict(self.provider_options))

    @classmethod
    def from_dict(cls, settings: Mapping[str, Any] | None) -> "IntentClassifierOptions":
        if not settings:
            return cls()

        section = settings.get("IntentClassifier", settings)
        if not isinstance(section, Mapping):
            raise ValueError("IntentClassifier configuration must be an object.")

        return cls(
            provider=section.get("Provider", DEFAULT_PROVIDER),
            model=section.get("Model", DEFAULT_MODEL),
            top_k=section.get("TopK", DEFAULT_TOP_K),
            fail_on_unknown_label=section.get("FailOnUnknownLabel", True),
            provider_options=section.get("ProviderOptions", {}),
        )


def _as_bool(value: Any, setting_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{setting_name} must be a boolean.")

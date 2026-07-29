from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def clean_email_text(text: str, max_chars: int = 12000) -> str:
    """Normalize extracted PDF text before classification."""

    cleaned = _WHITESPACE_RE.sub(" ", text or "").strip()
    return cleaned[:max_chars]

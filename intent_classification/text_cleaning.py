from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def clean_email_text(text: str, max_chars: int = 12000) -> str:
    """Normalize PDF whitespace and cap input size before classification.

    The character limit prevents unusually large documents from creating an
    unnecessarily expensive model input; the tokenizer may apply its own limit.
    """

    cleaned = _WHITESPACE_RE.sub(" ", text or "").strip()
    return cleaned[:max_chars]

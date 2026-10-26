"""Shared text normalization used for both training and inference."""
from __future__ import annotations

import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Lowercase, unescape HTML, drop URLs/tags and collapse whitespace."""
    if text is None:
        return ""
    value = html.unescape(str(text))
    value = _TAG.sub(" ", value)
    value = _URL.sub(" ", value)
    return _SPACE.sub(" ", value).strip().lower()

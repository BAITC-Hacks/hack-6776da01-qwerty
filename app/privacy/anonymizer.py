"""Локальная эвристическая анонимизация транскриптов."""

from __future__ import annotations

import re


PATTERNS = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?7|8)[\s()-]*\d{3}[\s()-]*\d{2,3}[\s-]*\d{2}[\s-]*\d{2}(?!\w)")),
    ("IIN", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    ("BIN", re.compile(r"(?<!\d)\d{12}(?!\d)")),
)


def anonymize_text(text: str) -> tuple[str, dict[str, int]]:
    """Заменяет распространённые идентификаторы на метки и возвращает статистику."""
    result = text
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS:
        result, count = pattern.subn(f"[{label}]", result)
        counts[label] = counts.get(label, 0) + count
    return result, counts

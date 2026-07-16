"""Deterministic draft sanitizer for forbidden cliche vocabulary."""

from __future__ import annotations

import re


_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("consequently", "so"),
    ("in conclusion", "in short"),
    ("in summary", "in short"),
    ("it is important to note", "keep in mind"),
    ("needless to say", "keep in mind"),
    ("delve", "look into"),
    ("leverage", "use"),
    ("utilize", "use"),
    ("foster", "build"),
    ("cultivate", "build"),
    ("optimize", "improve"),
    ("enhance", "improve"),
    ("revolutionize", "change"),
    ("transform", "reshape"),
    ("seamlessly", "smoothly"),
    ("furthermore", "also"),
    ("moreover", "also"),
    ("testament", "proof"),
    ("tapestry", "mix"),
    ("beacon", "guide"),
    ("labyrinth", "maze"),
    ("paramount", "key"),
    ("invaluable", "vital"),
    ("game-changing", "really effective"),
    ("groundbreaking", "really effective"),
    ("multifaceted", "complex"),
    ("bespoke", "tailored"),
    ("pioneer", "lead"),
)


def sanitize_draft(text: str) -> str:
    """Replace forbidden words with safer alternatives.

    Matching is case-insensitive and whole-word based so only exact vocabulary
    hits are rewritten.
    """

    sanitized = text
    for source_word, replacement in _REPLACEMENTS:
        pattern = re.compile(rf"\b{re.escape(source_word)}\b", flags=re.IGNORECASE)
        sanitized = pattern.sub(lambda match: _preserve_case(match.group(0), replacement), sanitized)
    return sanitized


def _preserve_case(original: str, replacement: str) -> str:
    """Apply replacement while retaining a simple casing style."""

    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement
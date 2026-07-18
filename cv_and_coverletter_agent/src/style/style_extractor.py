"""One-time style extraction from writing samples."""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
import re
from pathlib import Path
from typing import Any, Sequence

from .style_profile import StyleProfile


class StyleExtractor:
    """Build and persist a StyleProfile from user writing samples.

    The actual Deepseek call is intentionally left as a stub in this scaffold.
    """

    def __init__(self, client: Any | None = None, profile_path: str | Path | None = None) -> None:
        self.client = client
        self.profile_path = Path(profile_path) if profile_path is not None else None

    async def extract(self, samples: Sequence[str]) -> StyleProfile:
        """Analyze writing samples and return a structured style profile."""

        cleaned_samples = [sample.strip() for sample in samples if sample.strip()]
        if not cleaned_samples:
            raise ValueError("at least one non-empty writing sample is required")

        # A compact local pass keeps the project usable before the Deepseek call is wired in.
        sentences = _split_sentences(" ".join(cleaned_samples))
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.split()]
        avg_sentence_length = mean(sentence_lengths) if sentence_lengths else 0.0
        sentence_length_variance = pstdev(sentence_lengths) ** 2 if len(sentence_lengths) > 1 else 0.0
        characteristic_phrases = _top_phrases(cleaned_samples)
        phrases_to_avoid = _infer_phrases_to_avoid(cleaned_samples)
        tone_description = _infer_tone_description(avg_sentence_length, cleaned_samples)
        structural_notes = _infer_structural_notes(cleaned_samples, sentences)
        anchor_snippets = _extract_anchor_snippets(cleaned_samples)

        return StyleProfile(
            tone_description=tone_description,
            avg_sentence_length=avg_sentence_length,
            sentence_length_variance=sentence_length_variance,
            characteristic_phrases=characteristic_phrases,
            phrases_to_avoid=phrases_to_avoid,
            structural_notes=structural_notes,
            anchor_snippets=anchor_snippets,
        )

    def save_profile(self, profile: StyleProfile, path: str | Path | None = None) -> None:
        """Persist a profile so style extraction only needs to run once."""

        target_path = Path(path) if path is not None else self.profile_path
        if target_path is None:
            raise ValueError("a profile path must be provided")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def load_profile(self, path: str | Path | None = None) -> StyleProfile:
        """Load a previously saved style profile from disk."""

        target_path = Path(path) if path is not None else self.profile_path
        if target_path is None:
            raise ValueError("a profile path must be provided")
        return StyleProfile.model_validate_json(target_path.read_text(encoding="utf-8"))


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences for lightweight style analysis."""

    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def _top_phrases(samples: Sequence[str], limit: int = 5) -> list[str]:
    """Pick repeated short phrases to preserve in the extracted style."""

    words = []
    for sample in samples:
        words.extend(re.findall(r"[A-Za-z][A-Za-z'-]+", sample.lower()))
    counts = Counter(words)
    phrases = [word for word, count in counts.items() if count > 1 and len(word) > 3]
    return sorted(phrases)[:limit]


def _infer_phrases_to_avoid(samples: Sequence[str]) -> list[str]:
    """Flag generic filler language that commonly makes writing feel templated."""

    generic_phrases = [
        "i am passionate about",
        "i am excited to",
        "leveraging my skills",
        "dynamic environment",
        "team player",
    ]
    combined = " \n".join(sample.lower() for sample in samples)
    return [phrase for phrase in generic_phrases if phrase in combined]


def _infer_tone_description(avg_sentence_length: float, samples: Sequence[str]) -> str:
    """Summarize tone using a small set of readable buckets."""

    sample_text = " ".join(samples).lower()
    if avg_sentence_length < 12:
        tone = "direct and concise"
    elif avg_sentence_length < 20:
        tone = "balanced and professional"
    else:
        tone = "reflective and detailed"

    if any(marker in sample_text for marker in ("thank you", "appreciate", "kind regards")):
        tone = f"polite, {tone}"
    return tone


def _infer_structural_notes(samples: Sequence[str], sentences: Sequence[str]) -> str:
    """Capture coarse structure so future prompts can mimic it."""

    paragraph_count = sum(1 for sample in samples if "\n\n" in sample or "\r\n\r\n" in sample)
    sentence_count = len(sentences)
    return (
        f"Observed {len(samples)} sample(s), about {paragraph_count + 1} paragraph block(s), "
        f"and {sentence_count} sentence(s) across the corpus."
    )


def _extract_anchor_snippets(samples: Sequence[str]) -> list[str]:
    """Select 3-4 distinct characteristic paragraphs from the user's writing."""

    paragraphs: list[str] = []
    for sample in samples:
        split = re.split(r"\n\s*\n", sample)
        for paragraph in split:
            cleaned = paragraph.strip()
            if cleaned:
                paragraphs.append(cleaned)

    if not paragraphs:
        return []

    unique: dict[str, str] = {}
    for paragraph in paragraphs:
        key = re.sub(r"\s+", " ", paragraph.lower())
        unique.setdefault(key, paragraph)
    deduped = list(unique.values())

    scored = sorted(
        deduped,
        key=lambda paragraph: (_paragraph_score(paragraph), len(paragraph)),
        reverse=True,
    )
    limit = 4 if len(scored) >= 4 else len(scored)
    selected = scored[:limit]
    if len(selected) > 3:
        return selected[:4]
    return selected


def _paragraph_score(paragraph: str) -> float:
    """Score paragraphs by lexical richness and sentence structure."""

    words = re.findall(r"[A-Za-z][A-Za-z'-]+", paragraph)
    if not words:
        return 0.0
    unique_ratio = len(set(word.lower() for word in words)) / len(words)
    sentence_count = max(1, len(_split_sentences(paragraph)))
    length_factor = min(len(words), 120) / 120
    return unique_ratio * 0.6 + (sentence_count / 8) * 0.2 + length_factor * 0.2

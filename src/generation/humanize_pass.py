"""Optional rewrite pass for reducing templated rhythm."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from src.style.style_profile import StyleProfile


async def rewrite_for_natural_rhythm(
    draft: str,
    style_profile: StyleProfile,
    client: Any | None = None,
    model: str | None = None,
    voice_mode: str = "professional",
    scene_mode: str = "public-writing",
) -> str:
    """Rewrite a draft when AI-tone checks flag rhythm as too uniform.

    The rewrite is constrained to preserve factual claims from the original draft
    while increasing sentence-length variance and matching cadence from
    style-profile anchor snippets.
    """

    if not draft.strip():
        raise ValueError("draft must not be empty")

    resolved_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    active_client = client or _create_default_client()
    temperature = _env_float("DEEPSEEK_HUMANIZE_TEMPERATURE", 0.78)
    frequency_penalty = _env_float("DEEPSEEK_HUMANIZE_FREQUENCY_PENALTY", 0.45)
    presence_penalty = _env_float("DEEPSEEK_HUMANIZE_PRESENCE_PENALTY", 0.20)

    system_prompt = (
        "You are an expert editor. Rewrite text to sound naturally human and less templated. "
        "Keep all factual content grounded in the original draft. "
        "Do not introduce any new facts, technologies, employers, dates, or achievements. "
        "Increase the text's burstiness and lexical variety while staying clear and factual. "
        "Never use em dashes or en dashes in the rewritten text."
    )
    user_prompt = _build_humanize_prompt(
        draft=draft,
        style_profile=style_profile,
        voice_mode=voice_mode,
        scene_mode=scene_mode,
    )

    try:
        response = await active_client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as exc:
        raise RuntimeError(f"humanization pass failed: {exc}") from exc

    rewritten = _extract_response_text(response)
    if not rewritten:
        raise RuntimeError("humanization pass returned empty content")
    return _postprocess_humanized_text(rewritten)


def _build_humanize_prompt(
    draft: str,
    style_profile: StyleProfile,
    voice_mode: str,
    scene_mode: str,
) -> str:
    """Build a rewrite prompt anchored to profile cadence and constraints."""

    anchor_block = "\n\n".join(
        f"- {snippet}" for snippet in style_profile.anchor_snippets[:4]
    ) or "- No anchor snippets provided"

    return (
        "Rewrite the draft below with these constraints:\n"
        "1) Vary sentence length noticeably (mix short, medium, and longer sentences).\n"
        "2) Remove generic or boilerplate cover-letter sounding lines.\n"
        "3) Preserve all factual claims from the original draft.\n"
        "4) Do not add any new facts or credentials.\n"
        "5) Match sentence-length variance, cadence, and structural rhythm from the anchor snippets.\n"
        "6) Prefer concrete verbs and specific phrasing over abstract motivational language.\n"
        "7) Keep paragraph lengths uneven and natural; avoid repeated sentence templates.\n"
        "8) Keep role/company references factual and concise, avoiding filler transitions.\n"
        "9) Keep most sentences short-to-medium length; target roughly 12-22 words per sentence.\n"
        "10) If a sentence runs too long, split it into two clear sentences.\n\n"
        "11) Avoid formal scaffolding phrases such as 'I am writing to express my interest', 'In terms of', and 'Furthermore'.\n"
        "12) Prefer direct first-person statements with concrete actions over abstract claims.\n"
        "13) Keep opening and closing concise; avoid generic courtesy filler.\n"
        "14) Strictly avoid generic AI buzzwords and formal academic filler such as Consequently, In conclusion, In summary, Needless to say, Leverage, Utilize, Foster, Cultivate, Optimize, Enhance, Revolutionize, Transform, Tapestry, Testament, Beacon, Labyrinth, Paramount, Invaluable, Game-changing, Groundbreaking, and Multifaceted.\n"
        "15) If one of those appears, replace it with plainer wording like also, so, in short, keep in mind, use, build, improve, change, reshape, mix, proof, guide, maze, key, vital, really effective, or complex.\n\n"
        f"Scene mode target: {scene_mode}\n"
        f"Scene guidance: {_scene_mode_guidance(scene_mode)}\n"
        f"Voice mode target: {voice_mode}\n"
        f"Voice guidance: {_voice_mode_guidance(voice_mode)}\n\n"
        "Style profile:\n"
        f"- Tone: {style_profile.tone_description}\n"
        f"- Avg sentence length: {style_profile.avg_sentence_length}\n"
        f"- Sentence variance: {style_profile.sentence_length_variance}\n"
        f"- Characteristic phrases: {', '.join(style_profile.characteristic_phrases) or 'N/A'}\n"
        f"- Phrases to avoid: {', '.join(style_profile.phrases_to_avoid) or 'N/A'}\n"
        f"- Structural notes: {style_profile.structural_notes}\n"
        f"- Anchor snippets:\n{anchor_block}\n\n"
        f"Original draft:\n{draft}\n"
    )


def _voice_mode_guidance(voice_mode: str) -> str:
    """Return concise, deterministic tone guidance per voice mode."""

    mode = voice_mode.strip().lower()
    if mode == "technical":
        return "Use precise wording, low fluff, and concrete engineering statements."
    if mode == "warm":
        return "Use clear, approachable language while keeping claims factual and concise."
    if mode == "blunt":
        return "Use short direct statements and avoid hedging language."
    return "Use measured professional tone with direct claims and minimal filler."


def _scene_mode_guidance(scene_mode: str) -> str:
    """Return concise scene guardrails for rewriting intent."""

    mode = scene_mode.strip().lower()
    if mode == "docs":
        return "Keep terminology stable, keep claims searchable, and avoid marketing-style rhetoric."
    if mode == "status":
        return "Prioritize factual clarity, concrete ownership, and concise progress statements."
    if mode == "chat":
        return "Use natural direct phrasing and avoid over-formal sentence framing."
    return "Keep a coherent public-writing register and avoid announcement-style fluff."


def _env_float(name: str, default: float) -> float:
    """Read a float from environment with safe fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _postprocess_humanized_text(text: str) -> str:
    """Apply deterministic cleanup that prompt instructions cannot guarantee."""

    # Normalize typographic dashes to plain ASCII punctuation.
    normalized = text.replace("—", "-").replace("–", "-")
    detemplated = _detemplate_phrases(normalized)
    return _split_long_sentences(detemplated, max_words=26)


def apply_deterministic_humanize_cleanup(text: str) -> str:
    """Expose deterministic cleanup for staged orchestration steps."""

    return _postprocess_humanized_text(text)


def _detemplate_phrases(text: str) -> str:
    """Replace a small set of repetitive formal phrases with plainer wording."""

    replacements: tuple[tuple[str, str], ...] = (
        ("I am writing to express my interest in", "I am applying for"),
        ("At the end of the day", ""),
        ("It is important to note that", ""),
        ("It's important to note that", ""),
        ("Needless to say", "Keep in mind"),
        ("In conclusion", "In short"),
        ("In summary", "In short"),
        ("Consequently", "So"),
        ("In today's world", "Today"),
        ("In a world where", "When"),
        ("At its core", ""),
        ("In terms of", "Regarding"),
        ("Furthermore,", "Also,"),
        ("Additionally,", "Also,"),
        ("Moreover,", "Also,"),
        ("Great question!", ""),
    )

    updated = text
    for source, target in replacements:
        pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
        updated = pattern.sub(lambda m: _preserve_case(m.group(0), target), updated)
    updated = _reduce_contrast_scaffolding(updated)
    # Preserve paragraph breaks; only collapse horizontal runs.
    updated = re.sub(r"[ \t]{2,}", " ", updated)
    updated = re.sub(r"(^|\n)[ \t]+", r"\1", updated)
    return updated.strip()


def _preserve_case(original: str, replacement: str) -> str:
    """Preserve simple case patterns for deterministic text replacements."""

    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _split_long_sentences(text: str, max_words: int = 26) -> str:
    """Split excessively long sentences at safe separators to improve readability."""

    if max_words < 8:
        return text

    sentence_parts = re.split(r"([.!?]\s+)", text)
    rebuilt: list[str] = []

    for index in range(0, len(sentence_parts), 2):
        sentence = sentence_parts[index]
        delimiter = sentence_parts[index + 1] if index + 1 < len(sentence_parts) else ""
        sentence = sentence.strip()
        if not sentence:
            if delimiter:
                rebuilt.append(delimiter)
            continue

        if len(sentence.split()) > max_words:
            sentence = _split_single_long_sentence(sentence, max_words=max_words)

        if delimiter:
            rebuilt.append(f"{sentence}{delimiter}")
        else:
            rebuilt.append(sentence)

    return "".join(rebuilt).strip()


def _split_single_long_sentence(sentence: str, max_words: int) -> str:
    """Split one long sentence into two sentences using conservative separators."""

    separators = ("; ", ", and ", ", which ", ", that ")
    words = sentence.split()
    midpoint = len(words) // 2

    for separator in separators:
        pos = sentence.lower().find(separator)
        if pos == -1:
            continue
        left_words = sentence[:pos].split()
        right_words = sentence[pos + len(separator):].split()
        if not left_words or not right_words:
            continue
        if abs(len(left_words) - midpoint) <= max(4, len(words) // 3):
            left = sentence[:pos].rstrip(" ,;")
            right = sentence[pos + len(separator):].lstrip(" ,;")
            if not right:
                continue
            right = right[0].upper() + right[1:] if len(right) > 1 else right.upper()
            return f"{left}. {right}"

    return sentence


def _reduce_contrast_scaffolding(text: str) -> str:
    """Flatten common AI-style binary framing without changing factual content."""

    patterns: tuple[tuple[str, str], ...] = (
        (r"\bnot only\b", ""),
        (r"\bnot just\b", ""),
        (r"\bnot merely\b", ""),
        (r"\bbut also\b", "and"),
    )

    updated = text
    for pattern, replacement in patterns:
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    updated = re.sub(r"\s+,", ",", updated)
    updated = re.sub(r"\s+and\s+and\s+", " and ", updated, flags=re.IGNORECASE)
    return updated


def _create_default_client() -> AsyncOpenAI:
    """Create Deepseek-compatible async client from environment settings."""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for humanization pass")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _extract_response_text(response: Any) -> str:
    """Extract normalized assistant text from OpenAI-compatible responses."""

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(getattr(part, "text", "")) for part in content]
        return "".join(parts).strip()
    return str(content).strip()

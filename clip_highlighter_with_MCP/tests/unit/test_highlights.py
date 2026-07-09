import pytest

from src.highlights import llm_highlighter
from src.highlights.llm_highlighter import HighlightResponseError, detect_highlights
from src.models.schemas import TranscriptSegment


@pytest.mark.asyncio
async def test_detect_highlights_returns_items() -> None:
    transcript = [
        TranscriptSegment(start_seconds=0, end_seconds=12, text="Intro"),
        TranscriptSegment(start_seconds=12, end_seconds=44, text="Main point"),
    ]

    async def fake_request(*_args, **_kwargs) -> str:
        return (
            '{"highlights":[{"start_seconds":0,"end_seconds":35,'
            '"hook_title":"Strong opening","rationale":"Clear standalone moment","confidence":0.91}]}'
        )

    original = llm_highlighter._request_highlights_json
    llm_highlighter._request_highlights_json = fake_request

    result = await detect_highlights(transcript, top_k=1, min_seconds=30, max_seconds=60)

    llm_highlighter._request_highlights_json = original

    assert len(result) == 1
    assert result[0].end_seconds > result[0].start_seconds


@pytest.mark.asyncio
async def test_detect_highlights_rejects_non_json_response() -> None:
    transcript = [
        TranscriptSegment(start_seconds=0, end_seconds=50, text="Useful content"),
    ]

    async def fake_request(*_args, **_kwargs) -> str:
        return "not-json"

    original = llm_highlighter._request_highlights_json
    llm_highlighter._request_highlights_json = fake_request

    with pytest.raises(HighlightResponseError):
        await detect_highlights(transcript, top_k=1, min_seconds=30, max_seconds=60)

    llm_highlighter._request_highlights_json = original


@pytest.mark.asyncio
async def test_detect_highlights_rejects_out_of_range_timestamps() -> None:
    transcript = [
        TranscriptSegment(start_seconds=10, end_seconds=40, text="Useful content"),
    ]

    async def fake_request(*_args, **_kwargs) -> str:
        return (
            '{"highlights":[{"start_seconds":0,"end_seconds":20,'
            '"hook_title":"Out of range","rationale":"Bad window","confidence":0.4}]}'
        )

    original = llm_highlighter._request_highlights_json
    llm_highlighter._request_highlights_json = fake_request

    with pytest.raises(HighlightResponseError):
        await detect_highlights(transcript, top_k=1, min_seconds=30, max_seconds=60)

    llm_highlighter._request_highlights_json = original

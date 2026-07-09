import pytest

from src.ingest.youtube_ingestor import InvalidYouTubeUrlError, _validate_youtube_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
    ],
)
def test_validate_youtube_url_accepts_supported_formats(url: str) -> None:
    _validate_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch",
        "https://youtu.be/",
    ],
)
def test_validate_youtube_url_rejects_invalid_formats(url: str) -> None:
    with pytest.raises(InvalidYouTubeUrlError):
        _validate_youtube_url(url)

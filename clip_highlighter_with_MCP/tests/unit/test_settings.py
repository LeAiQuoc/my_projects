from src.config.settings import AppSettings


def test_settings_defaults() -> None:
    settings = AppSettings()

    assert settings.default_top_clips > 0
    assert settings.default_clip_min_seconds < settings.default_clip_max_seconds

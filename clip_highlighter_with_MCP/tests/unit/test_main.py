from __future__ import annotations

import pytest

from src import main as main_module


def test_main_returns_code_1_on_unhandled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_asyncio_run(coro):
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module.asyncio, "run", fake_asyncio_run)

    with pytest.raises(SystemExit) as exc:
        main_module.main()

    assert exc.value.code == 1


def test_main_returns_code_130_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_asyncio_run(coro):
        coro.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_module.asyncio, "run", fake_asyncio_run)

    with pytest.raises(SystemExit) as exc:
        main_module.main()

    assert exc.value.code == 130

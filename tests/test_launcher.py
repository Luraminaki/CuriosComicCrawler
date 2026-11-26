"""Tests for `curios_comic_crawler.launcher`'s interactive and scripted modes."""

import argparse
import io
from typing import TYPE_CHECKING

from curios_comic_crawler import downloader, launcher, upscaler

if TYPE_CHECKING:
    import pytest

    from curios_comic_crawler.config import AppConfig


class _Terminal(io.StringIO):
    """Scripted stdin that reports itself as an interactive terminal."""

    def isatty(self) -> bool:
        """Claim to be a terminal.

        Returns:
            True.
        """
        return True


def _record_runs(monkeypatch: "pytest.MonkeyPatch") -> dict[str, bool]:
    """Replace both steps with stubs that record the `force` they were given.

    Args:
        monkeypatch: The test's monkeypatch fixture.

    Returns:
        Step name to the `force` value it ran with (absent if it didn't run).
    """
    calls: dict[str, bool] = {}

    def fake_download(_config: "AppConfig", *, force: bool) -> int:
        """Record a download run.

        Returns:
            0.
        """
        calls["download"] = force
        return 0

    def fake_upscale(_config: "AppConfig", *, force: bool) -> int:
        """Record an upscale run.

        Returns:
            0.
        """
        calls["upscale"] = force
        return 0

    monkeypatch.setattr(downloader, "run", fake_download)
    monkeypatch.setattr(upscaler, "run", fake_upscale)
    return calls


def _args(mode: str | None = None, *, force_download: bool = False) -> argparse.Namespace:
    """Build parsed launcher arguments.

    Args:
        mode: The `--mode` value.
        force_download: The `--force-download` flag.

    Returns:
        The arguments.
    """
    return argparse.Namespace(mode=mode, force_download=force_download, force_upscale=False)


def test_force_flag_answers_its_own_question(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """`--force-download` without `--mode` is kept, and only the other question is asked."""
    calls = _record_runs(monkeypatch)
    # "b" picks both steps; "n" is the only answer left to give (re-upscale).
    monkeypatch.setattr("sys.stdin", _Terminal("b\nn\n"))

    assert launcher.run(config, _args(force_download=True)) == 0
    assert calls == {"download": True, "upscale": False}


def test_no_mode_without_a_terminal_fails_instead_of_waiting(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """With no terminal (cron, pipes) and no `--mode`, nothing runs and the exit code is 2."""
    calls = _record_runs(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    assert launcher.run(config, _args()) == 2
    assert not calls


def test_mode_runs_without_asking(config: "AppConfig", monkeypatch: "pytest.MonkeyPatch") -> None:
    """`--mode` never reads stdin."""
    calls = _record_runs(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    assert launcher.run(config, _args("upscale")) == 0
    assert calls == {"upscale": False}

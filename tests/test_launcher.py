"""Tests for `curios_comic_crawler.launcher`'s interactive and scripted modes."""

import argparse
import io
from typing import TYPE_CHECKING

from curios_comic_crawler import downloader, launcher, upscaler
from curios_comic_crawler._exit_codes import INTERRUPTED_EXIT_CODE

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


def _record_runs(
    monkeypatch: "pytest.MonkeyPatch", download_code: int = 0, upscale_code: int = 0
) -> dict[str, bool]:
    """Replace both steps with stubs that record the `force` they were given.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        download_code: What the download step returns.
        upscale_code: What the upscale step returns.

    Returns:
        Step name to the `force` value it ran with (absent if it didn't run).
    """
    calls: dict[str, bool] = {}

    def fake_download(_config: "AppConfig", *, force: bool) -> int:
        """Record a download run.

        Args:
            _config: Ignored.
            force: The `--force-download` choice.

        Returns:
            `download_code`.
        """
        calls["download"] = force
        return download_code

    def fake_upscale(_config: "AppConfig", *, force: bool) -> int:
        """Record an upscale run.

        Args:
            _config: Ignored.
            force: The `--force-upscale` choice.

        Returns:
            `upscale_code`.
        """
        calls["upscale"] = force
        return upscale_code

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


def test_both_mode_upscales_even_after_a_failed_download(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """A failed download still lets already-downloaded pages be upscaled; the failure is kept."""
    calls = _record_runs(monkeypatch, download_code=1)

    assert launcher.run(config, _args("both")) == 1
    assert calls == {"download": False, "upscale": False}


def test_interrupted_upscale_code_wins_over_a_download_failure(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """The worst code is returned: Ctrl+C during the upscale (130) beats a failed download (1)."""
    _ = _record_runs(monkeypatch, download_code=1, upscale_code=INTERRUPTED_EXIT_CODE)

    assert launcher.run(config, _args("both")) == INTERRUPTED_EXIT_CODE


def test_invalid_menu_answer_asks_again(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch", capsys: "pytest.CaptureFixture[str]"
) -> None:
    """An answer outside the menu is reported and the question asked again."""
    calls = _record_runs(monkeypatch)
    # "x" is not a choice; "d" picks download; "n" declines the re-download.
    monkeypatch.setattr("sys.stdin", _Terminal("x\nd\nn\n"))

    assert launcher.run(config, _args()) == 0
    assert calls == {"download": False}
    assert "'x' is not a valid choice." in capsys.readouterr().out

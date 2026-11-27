"""Tests for `curios_comic_crawler._cli.main`: every command's exit codes and config loading."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from curios_comic_crawler import _cli
from curios_comic_crawler._exit_codes import INTERRUPTED_EXIT_CODE

if TYPE_CHECKING:
    import argparse
    import pathlib
    from collections.abc import Callable

    from curios_comic_crawler.config import AppConfig


@pytest.fixture(name="workdir", autouse=True)
def fixture_workdir(
    tmp_path: "pathlib.Path", monkeypatch: "pytest.MonkeyPatch", restore_root_logger: None
) -> "pathlib.Path":
    """Run in `tmp_path` (where `main` writes its log), restoring the root logger afterwards.

    Args:
        tmp_path: The test's own folder.
        monkeypatch: Changes the working directory and `sys.argv`.
        restore_root_logger: Undoes `main`'s logging setup (its log file handler) afterwards.

    Returns:
        The working directory.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["comiccrawler-test"])
    del restore_root_logger  # requested for its teardown only
    return tmp_path


@pytest.fixture(name="config_file")
def fixture_config_file(workdir: "pathlib.Path", valid_config_dict: dict) -> "pathlib.Path":
    """A valid `config.json` in the working directory (the default location).

    Args:
        workdir: The working directory.
        valid_config_dict: The config to write.

    Returns:
        Its path.
    """
    path = workdir / "config.json"
    _ = path.write_text(json.dumps(valid_config_dict), encoding="utf-8")
    return path


def _returning(code: int) -> "Callable[[AppConfig, argparse.Namespace], int]":
    """Build a command body that only returns `code`.

    Args:
        code: The exit code.

    Returns:
        The command body.
    """
    return lambda _config, _args: code


def _raising(error: BaseException) -> "Callable[[AppConfig, argparse.Namespace], int]":
    """Build a command body that raises `error`.

    Args:
        error: What to raise.

    Returns:
        The command body.
    """

    def run(_config: "AppConfig", _args: "argparse.Namespace") -> int:
        """Raise the error.

        Args:
            _config: Ignored.
            _args: Ignored.

        Raises:
            BaseException: `error`.
        """
        raise error

    return run


@pytest.mark.usefixtures("config_file")
def test_the_command_exit_code_is_returned() -> None:
    """With the default `./config.json`, the command's own exit code comes back."""
    assert _cli.main("comiccrawler-test", _returning(7)) == 7


def test_missing_config_file_returns_1(workdir: "pathlib.Path") -> None:
    """No config file: exit 1, the command never runs, and the log says how to fix it."""
    assert _cli.main("comiccrawler-test", _raising(AssertionError("must not run"))) == 1
    assert "or pass its path with -c/--configuration" in _log_text(workdir)


@pytest.mark.parametrize("content", ["{not json", '{"root_site": "https://example.com/"}'])
def test_invalid_config_returns_1(workdir: "pathlib.Path", content: str) -> None:
    """Bad JSON or a config failing validation: exit 1, and the command never runs."""
    _ = (workdir / "config.json").write_text(content, encoding="utf-8")

    assert _cli.main("comiccrawler-test", _raising(AssertionError("must not run"))) == 1


def _log_text(workdir: "pathlib.Path") -> str:
    """Read the run's log file (`main` replaces every handler, pytest's capture included).

    Args:
        workdir: Where `main` ran.

    Returns:
        The log's content.
    """
    (log_file,) = workdir.glob("comiccrawler-test-*.log")
    return log_file.read_text(encoding="utf-8")


@pytest.mark.usefixtures("config_file")
def test_a_crash_returns_1_and_logs_the_traceback(workdir: "pathlib.Path") -> None:
    """An unexpected error is logged with its traceback, and the exit code is 1."""
    assert _cli.main("comiccrawler-test", _raising(RuntimeError("boom"))) == 1
    assert "Traceback" in _log_text(workdir)


@pytest.mark.usefixtures("config_file")
def test_ctrl_c_returns_130_without_a_traceback(workdir: "pathlib.Path") -> None:
    """Ctrl+C anywhere in a command (a download, a prompt) ends it with 130, cleanly."""
    assert _cli.main("comiccrawler-test", _raising(KeyboardInterrupt())) == INTERRUPTED_EXIT_CODE
    log = _log_text(workdir)
    assert "Interrupted" in log
    assert "Traceback" not in log


def test_configuration_option_selects_the_file(
    workdir: "pathlib.Path", valid_config_dict: dict, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """`-c` loads the given file instead of `./config.json`."""
    path = workdir / "elsewhere.json"
    _ = path.write_text(json.dumps(valid_config_dict), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["comiccrawler-test", "-c", str(path)])

    assert _cli.main("comiccrawler-test", _returning(0)) == 0


def test_version_exits_without_creating_a_log(
    workdir: "pathlib.Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """`--version` is answered before logging starts: no log file is left behind."""
    monkeypatch.setattr(sys, "argv", ["comiccrawler-test", "--version"])

    with pytest.raises(SystemExit) as stopped:
        _ = _cli.main("comiccrawler-test", _returning(0))
    assert stopped.value.code == 0
    assert list(workdir.iterdir()) == []


def test_download_only_launcher_does_not_load_the_upscaler() -> None:
    """Importing the launcher leaves cv2 and ONNX Runtime unloaded (~94 MiB saved)."""
    probe = (
        "import sys; import curios_comic_crawler.launcher; "
        + "print(sorted({'cv2', 'onnxruntime'} & set(sys.modules)))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert loaded == "[]"

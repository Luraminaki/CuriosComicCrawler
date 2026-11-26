"""Tests for `curios_comic_crawler.logging_utils`: per-run log files and their pruning."""

# pylint: disable=protected-access  # unit tests exercise the module's private helpers

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import pytest

from curios_comic_crawler import logging_utils

if TYPE_CHECKING:
    import pathlib

STEM = "comiccrawler"
_BASE_MTIME_NS = 1_700_000_000 * 10**9


def _write_run(log_dir: "pathlib.Path", pid: int, age: int, rotated: int = 0) -> list[str]:
    """Write one past run's log files, all modified `age` seconds before a fixed instant.

    Args:
        log_dir: Where the logs live.
        pid: The run's process id (its file names).
        age: How old the run is; a larger age is an older run.
        rotated: How many rotated `.log.N` files the run also left.

    Returns:
        The file names written.
    """
    names = [f"{STEM}-{pid}.log", *(f"{STEM}-{pid}.log.{n}" for n in range(1, rotated + 1))]
    mtime_ns = _BASE_MTIME_NS - age * 10**9
    for name in names:
        path = log_dir / name
        _ = path.write_text("log line\n", encoding="utf-8")
        # pathlib has no way to set a file's times; os.utime is the only API for it.
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return names


def _names(log_dir: "pathlib.Path") -> set[str]:
    """List what's left in the log folder.

    Args:
        log_dir: The log folder.

    Returns:
        The file names.
    """
    return {path.name for path in log_dir.iterdir()}


def test_prune_keeps_the_newest_runs_by_modification_time(tmp_path: "pathlib.Path") -> None:
    """The most recently written runs survive, whatever their PIDs (recycled, not ordered)."""
    newest = _write_run(tmp_path, pid=900, age=1)
    second = _write_run(tmp_path, pid=5, age=2)
    _ = _write_run(tmp_path, pid=999, age=3)
    _ = _write_run(tmp_path, pid=1, age=4)

    logging_utils._prune_old_runs(tmp_path, STEM, max_runs=2)

    assert _names(tmp_path) == {*newest, *second}


def test_prune_removes_rotated_files_with_their_run(tmp_path: "pathlib.Path") -> None:
    """A pruned run's `.log.1`, `.log.2`... go with it; a kept run keeps all of its own."""
    kept = _write_run(tmp_path, pid=20, age=1, rotated=2)
    _ = _write_run(tmp_path, pid=10, age=5, rotated=3)

    logging_utils._prune_old_runs(tmp_path, STEM, max_runs=1)

    assert _names(tmp_path) == set(kept)


def test_prune_leaves_other_files_sharing_the_stem(tmp_path: "pathlib.Path") -> None:
    """Another script's logs (`comiccrawler-upscale-*`) and unrelated files are never pruned."""
    others = {f"{STEM}-upscale-7.log", f"{STEM}-upscale-7.log.1", f"{STEM}-7.txt", "notes.log"}
    for name in others:
        (tmp_path / name).touch()
    kept = _write_run(tmp_path, pid=2, age=1)
    _ = _write_run(tmp_path, pid=1, age=2)

    logging_utils._prune_old_runs(tmp_path, STEM, max_runs=1)

    assert _names(tmp_path) == {*others, *kept}


def test_prune_does_nothing_at_max_runs(tmp_path: "pathlib.Path") -> None:
    """Exactly `max_runs` runs on disk: none is removed, however old."""
    written = {name for pid in range(1, 4) for name in _write_run(tmp_path, pid, age=pid * 1000)}

    logging_utils._prune_old_runs(tmp_path, STEM, max_runs=3)

    assert _names(tmp_path) == written


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_creates_this_runs_log_after_pruning(
    tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new run's PID-named log is opened after pruning, so it's never pruned itself."""
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    kept = _write_run(tmp_path, pid=2, age=1)
    _ = _write_run(tmp_path, pid=1, age=2)

    logging_utils.configure_launcher_logging(STEM, log_dir=tmp_path, max_runs=1)

    assert _names(tmp_path) == {*kept, f"{STEM}-4242.log"}
    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert [h.baseFilename for h in file_handlers] == [str(tmp_path / f"{STEM}-4242.log")]
    assert logging.getLogger().level == logging.INFO

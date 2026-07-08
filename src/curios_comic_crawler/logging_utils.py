#!/usr/bin/env python3
"""Shared logging setup for the CLI entry points."""

import logging
import os
import pathlib
import re
from logging.handlers import RotatingFileHandler

from . import logreset

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5
_DEFAULT_MAX_RUNS = 10


def _prune_old_runs(log_dir: pathlib.Path, log_file_stem: str, max_runs: int) -> None:
    """Delete log files from old runs, keeping only the `max_runs` most recent.

    Each process gets its own `<log_file_stem>-<pid>.log(.N)?` file set (see
    `configure_launcher_logging`), so nothing else ever prunes them; left alone, a script run
    repeatedly (e.g. from cron) accumulates one file set per invocation forever.

    Args:
        log_dir (pathlib.Path): Directory log files are written to.
        log_file_stem (str): The stem shared by this script's log files.
        max_runs (int): Number of most recent runs (by file set) to keep.
    """
    run_pattern = re.compile(rf'^{re.escape(log_file_stem)}-(\d+)\.log(\.\d+)?$')
    runs: dict[str, list[pathlib.Path]] = {}
    for path in log_dir.glob(f'{log_file_stem}-*.log*'):
        match = run_pattern.match(path.name)
        if match is not None:
            runs.setdefault(match.group(1), []).append(path)

    if len(runs) <= max_runs:
        return

    # Secondary key on the numeric pid breaks ties deterministically when several runs land in
    # the same mtime tick (e.g. a burst of quick invocations) -- PIDs increase monotonically
    # within a reasonable timeframe, so a higher one means a more recent run.
    newest_first = sorted(
        runs, key=lambda pid: (max(p.stat().st_mtime for p in runs[pid]), int(pid)), reverse=True,
    )
    for pid in newest_first[max_runs:]:
        for path in runs[pid]:
            path.unlink(missing_ok=True)


def configure_launcher_logging(  # noqa: PLR0913
    logger: logging.Logger,
    log_file_stem: str,
    log_dir: pathlib.Path | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    max_runs: int = _DEFAULT_MAX_RUNS,
) -> None:
    """Initialize logging consistently across CLI entry points.

    The log file is named ``<log_file_stem>-<pid>.log``, one per OS process, rather than one
    shared file: ``RotatingFileHandler``'s rotation (renaming the file once it grows past
    `max_bytes`) is not safe across multiple processes -- if two processes hold the same file
    open, one process rotating it can orphan the other's file handle, silently dropping its log
    lines from then on. Each file still rotates on its own, keeping up to `backup_count` older
    files (`<log_file_stem>-<pid>.log.1`, `.2`, ...) before the oldest is discarded.

    Args:
        logger (logging.Logger): The logger instance to configure.
        log_file_stem (str): The stem for the log file name.
        log_dir (pathlib.Path | None, optional): Directory the log file is written to. Defaults
            to the current working directory at call time.
        max_bytes (int, optional): Size in bytes a log file may reach before it
            rolls over. Defaults to 5 MiB.
        backup_count (int, optional): Number of rotated log files to keep. Defaults to 5.
        max_runs (int, optional): Number of past process runs' log file sets to keep before
            older ones are deleted. Defaults to 10.
    """
    logreset.reset_logging()

    level = logging.INFO
    log_dir = (log_dir or pathlib.Path.cwd())
    log_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_runs(log_dir, log_file_stem, max_runs)
    log_file = log_dir / f'{log_file_stem}-{os.getpid()}.log'
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(process)s] [%(name)s] [%(levelname)s]: %(funcName)s -- %(message)s',
        handlers=[
            RotatingFileHandler(log_file, mode='a', maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    logger.setLevel(level)

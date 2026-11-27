#!/usr/bin/env python3
"""Write files so that they are either absent, previous, or complete -- never truncated."""

import contextlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator


@contextlib.contextmanager
def atomic_target(path: "pathlib.Path") -> "Generator[pathlib.Path]":
    """Give a temporary path next to `path`, renamed over `path` if the block succeeds.

    A file cut off mid-write (Ctrl+C, a full disk, a killed process) would otherwise exist with
    its final name and be taken as complete. The temporary name carries the process ID, so two
    runs writing to the same folder never mix their writes. It's removed whatever happens.

    Args:
        path (pathlib.Path): The final path.

    Yields:
        pathlib.Path: Where to write; it only replaces `path` once the block ends without error.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        yield temporary
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # left only if the block failed

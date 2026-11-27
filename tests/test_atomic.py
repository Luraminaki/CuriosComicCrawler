"""Tests for `curios_comic_crawler._atomic.atomic_target`."""

from typing import TYPE_CHECKING

import pytest

from curios_comic_crawler._atomic import atomic_target

if TYPE_CHECKING:
    import pathlib


def test_the_file_appears_complete_once_the_block_ends(tmp_path: "pathlib.Path") -> None:
    """The content is written under a temporary name, then renamed over the target."""
    target = tmp_path / "page.jpg"
    _ = target.write_bytes(b"previous")

    with atomic_target(target) as temporary:
        _ = temporary.write_bytes(b"new")
        assert temporary.parent == tmp_path
        assert target.read_bytes() == b"previous"  # untouched until the block succeeds

    assert target.read_bytes() == b"new"
    assert sorted(tmp_path.iterdir()) == [target]


def test_a_failed_block_keeps_the_previous_file_and_no_temporary(
    tmp_path: "pathlib.Path",
) -> None:
    """A write cut off midway never replaces the target and leaves nothing behind."""
    target = tmp_path / "page.jpg"
    _ = target.write_bytes(b"previous")

    with pytest.raises(OSError, match="disk full"), atomic_target(target) as temporary:
        _ = temporary.write_bytes(b"part")
        raise OSError("disk full")

    assert target.read_bytes() == b"previous"
    assert sorted(tmp_path.iterdir()) == [target]


def test_the_temporary_name_is_hidden_and_unique_per_process(tmp_path: "pathlib.Path") -> None:
    """Two runs writing the same page use different temporary files."""
    with atomic_target(tmp_path / "page.jpg") as temporary:
        _ = temporary.write_bytes(b"x")
        name = temporary.name

    assert name.startswith(".page.jpg.")
    assert name.endswith(".part")
    assert name != ".page.jpg.part"

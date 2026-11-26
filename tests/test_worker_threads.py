"""Tests for `curios_comic_crawler._worker_threads`: physical cores and the per-worker share."""

from typing import TYPE_CHECKING

import pytest

from curios_comic_crawler import _worker_threads

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator


@pytest.fixture(name="topology")
def fixture_topology(
    tmp_path: "pathlib.Path", monkeypatch: "pytest.MonkeyPatch"
) -> "Iterator[pathlib.Path]":
    """A fake `/sys/devices/system/cpu`: 4 logical CPUs, 2 per core, all usable.

    Yields:
        The fake folder (tests may break it).
    """
    for cpu, siblings in enumerate(("0,2", "1,3", "0,2", "1,3")):
        folder = tmp_path / f"cpu{cpu}" / "topology"
        folder.mkdir(parents=True)
        _ = (folder / "thread_siblings_list").write_text(f"{siblings}\n", encoding="ascii")
    monkeypatch.setattr(_worker_threads, "_CPU_TOPOLOGY", tmp_path)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
    _worker_threads.physical_cores.cache_clear()
    yield tmp_path
    _worker_threads.physical_cores.cache_clear()  # don't leak the fake count to other tests


def test_hyperthreads_of_one_core_count_once(topology: "pathlib.Path") -> None:
    """Two logical CPUs sharing a core are one core."""
    del topology
    assert _worker_threads.physical_cores() == 2


def test_unreadable_topology_falls_back_to_logical_cpus(topology: "pathlib.Path") -> None:
    """If the kernel's topology can't be read, every usable logical CPU counts."""
    (topology / "cpu3" / "topology" / "thread_siblings_list").unlink()
    assert _worker_threads.physical_cores() == 4


def test_each_worker_gets_its_share_of_the_cores(topology: "pathlib.Path") -> None:
    """Cores are split between workers, never below one thread each."""
    del topology
    assert _worker_threads.threads_per_worker(1) == 2
    assert _worker_threads.threads_per_worker(2) == 1
    assert _worker_threads.threads_per_worker(8) == 1

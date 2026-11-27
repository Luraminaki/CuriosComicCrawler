"""Tests for `curios_comic_crawler.upscaler`.

`_process_one`/`_init_worker` are exercised here against a fake `SREngine` (no real
model/worker process needed) to cover the cv2 read/write and thread-capping plumbing around
them; a real super-resolution model actually producing correct pixels is still only covered by
manual end-to-end runs.
"""

# pylint: disable=protected-access  # unit tests exercise the module's private helpers

import logging
import multiprocessing
import pathlib
import signal
from logging.handlers import QueueHandler
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from curios_comic_crawler import upscaler
from curios_comic_crawler.sr_engine_opencv import OpenCVEngineInit

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from curios_comic_crawler.config import AppConfig


class _IdentityEngine:
    """A no-op `SREngine` stand-in: returns the image unchanged."""

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Return `image` unchanged.

        Args:
            image: The page.

        Returns:
            The same page.
        """
        return image


def _write_test_image(path: pathlib.Path, size: tuple[int, int] = (8, 8)) -> None:
    """Write a random (reproducible) colour image.

    Args:
        path: Where to write it.
        size: Its height and width.
    """
    image = np.random.default_rng(0).integers(0, 256, size=(*size, 3), dtype=np.uint8)
    _ = cv2.imwrite(str(path), image)


def test_select_remaining_without_force_skips_completed() -> None:
    """Pages already upscaled are skipped."""
    originals = [pathlib.Path(f"p{i}.jpg") for i in range(5)]
    completed = originals[:3]

    assert upscaler._select_remaining(originals, completed) == originals[3:]


def test_select_remaining_handles_a_gap_in_completed() -> None:
    # Page 2 is missing from `completed` (e.g. it failed or finished out of submission order),
    # even though later pages 3 and 4 are already done -- a positional slice would miss this.
    """A page missing among completed ones is still upscaled."""
    originals = [pathlib.Path(f"p{i}.jpg") for i in range(5)]
    completed = [originals[0], originals[1], originals[3], originals[4]]

    assert upscaler._select_remaining(originals, completed) == [originals[2]]


def test_resolve_worker_count_caps_at_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never more workers than CPUs."""
    monkeypatch.setattr(upscaler, "available_cpus", lambda: 4)

    assert upscaler._resolve_worker_count(500, remaining_count=500) == 4


def test_resolve_worker_count_caps_at_remaining_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never more workers than pages left."""
    monkeypatch.setattr(upscaler, "available_cpus", lambda: 8)

    assert upscaler._resolve_worker_count(8, remaining_count=3) == 3


def test_resolve_worker_count_defaults_to_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a configured count, one worker (one per CPU needed ~30 GB on 24 CPUs)."""
    monkeypatch.setattr(upscaler, "available_cpus", lambda: 24)

    assert upscaler._resolve_worker_count(None, remaining_count=100) == 1


def test_format_duration_formats_days_hours_minutes_seconds() -> None:
    """Durations read as `<d>d - <h>h:<m>m:<s>s`."""
    assert upscaler._format_duration(0) == "0d - 0h:0m:0s"
    assert upscaler._format_duration(3661) == "0d - 1h:1m:1s"
    assert upscaler._format_duration(90000) == "1d - 1h:0m:0s"


def test_format_duration_clamps_negative_to_zero() -> None:
    """A negative duration reads as zero."""
    assert upscaler._format_duration(-5) == "0d - 0h:0m:0s"


def test_run_returns_success_when_nothing_to_upscale(config: "AppConfig") -> None:
    """Nothing to upscale is a success."""
    assert upscaler.run(config) == 0


def test_eta_divides_by_full_worker_count_while_a_queue_remains() -> None:
    """While pages wait in the queue, all workers share the remaining time."""
    # 10 pages left, 4 workers still have queued work waiting behind them.
    assert (
        upscaler._estimate_eta_seconds(avg_per_page=2.0, remaining_count=10, worker_count=4) == 5.0
    )


def test_eta_does_not_shrink_below_one_round_in_the_tail() -> None:
    """The last pages, all running at once, still take one page's time."""
    # Every remaining page is already running concurrently -- no more queue to divide away.
    for remaining_count in (1, 2, 3, 4):
        eta = upscaler._estimate_eta_seconds(
            avg_per_page=2.0, remaining_count=remaining_count, worker_count=4
        )
        assert eta == 2.0


def test_eta_is_zero_when_nothing_remains() -> None:
    """Nothing left: no time left."""
    assert (
        upscaler._estimate_eta_seconds(avg_per_page=2.0, remaining_count=0, worker_count=4) == 0.0
    )


def test_process_one_writes_upscaled_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The written page decodes to the engine's output, posterised to `gray_values` levels.

    A lossless `.png` page and a no-op sharpen keep the posterised levels countable (JPEG and
    the unsharp mask both create in-between values).
    """
    monkeypatch.setattr(upscaler, "_worker_engine", _IdentityEngine())
    monkeypatch.setattr(upscaler, "sharpen_image", lambda image: image)
    img_path = tmp_path / "page.png"
    _write_test_image(img_path)
    upscale_dir = tmp_path / "out"
    upscale_dir.mkdir()

    name, elapsed = upscaler._process_one(
        img_path, upscale_dir, gray_values=4, posterise_method="histogram"
    )

    assert name == "page.png"
    assert elapsed >= 0
    written = cv2.imread(str(upscale_dir / "page.png"), cv2.IMREAD_COLOR)
    assert written is not None
    assert written.shape == (8, 8, 3)
    assert np.unique(written).size <= 4


def test_process_one_raises_when_engine_not_initialized(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker without its engine fails loudly."""
    monkeypatch.setattr(upscaler, "_worker_engine", None)
    img_path = tmp_path / "page.jpg"
    _write_test_image(img_path)

    with pytest.raises(RuntimeError, match="not initialized"):
        _ = upscaler._process_one(img_path, tmp_path, gray_values=4, posterise_method="histogram")


def test_process_one_raises_on_missing_or_corrupt_image(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable image raises `ValueError`."""
    monkeypatch.setattr(upscaler, "_worker_engine", _IdentityEngine())
    bad_path = tmp_path / "not_an_image.jpg"
    _ = bad_path.write_bytes(b"not actually an image")

    with pytest.raises(ValueError, match="Failed to read image"):
        _ = upscaler._process_one(bad_path, tmp_path, gray_values=4, posterise_method="histogram")


def test_unencodable_page_fails_and_leaves_no_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page OpenCV can't encode fails loudly, leaving neither an output nor a temp file."""
    monkeypatch.setattr(upscaler, "_worker_engine", _IdentityEngine())
    img_path = tmp_path / "page.jpg"
    _write_test_image(img_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(cv2, "imencode", lambda *_a, **_k: (False, None))

    with pytest.raises(ValueError, match="can't encode"):
        _ = upscaler._process_one(img_path, out_dir, gray_values=4, posterise_method="histogram")
    assert list(out_dir.iterdir()) == []


def test_failed_write_never_leaves_a_truncated_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If writing fails midway, the final page doesn't exist (it would be skipped forever)."""
    out_path = tmp_path / "page.jpg"

    def fail_after_partial_write(self: pathlib.Path, data: bytes) -> int:
        """Write part of the file, then fail like a full disk.

        Args:
            self: The file being written (this replaces a `pathlib.Path` method).
            data: What should have been written (dropped).

        Raises:
            OSError: Always.
        """
        del data
        _ = pathlib.Path.write_text(self, "partial", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(pathlib.Path, "write_bytes", fail_after_partial_write)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(OSError, match="disk full"):
        upscaler._write_atomically(out_path, image)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("restore_root_logger")
def test_init_worker_sets_engine_and_caps_cv2_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each worker builds its engine, caps OpenCV's threads and logs through the queue."""
    captured: dict[str, int] = {}
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: captured.setdefault("n", n))
    monkeypatch.setattr(upscaler, "threads_per_worker", lambda _worker_count: 3)
    monkeypatch.setattr(
        upscaler, "build_engine", lambda _engine_init, _worker_count: _IdentityEngine()
    )
    # The worker globals `_init_worker` sets, put back by monkeypatch after the test.
    monkeypatch.setattr(upscaler, "_worker_engine", None)
    # Recorded, not installed: ignoring Ctrl+C in the test process would outlive a failure.
    handlers: dict[int, object] = {}
    monkeypatch.setattr(signal, "signal", handlers.setdefault)

    log_queue: Queue[logging.LogRecord] = multiprocessing.Queue()
    try:
        upscaler._init_worker(
            engine_init=OpenCVEngineInit(pathlib.Path("unused.pb"), "edsr", 3),
            worker_count=2,
            log_queue=log_queue,
            level=logging.INFO,
        )

        assert captured["n"] == 3
        assert isinstance(upscaler._worker_engine, _IdentityEngine)
        assert handlers == {signal.SIGINT: signal.SIG_IGN}  # Ctrl+C is the main's job
        root = logging.getLogger()
        assert [type(handler) for handler in root.handlers] == [QueueHandler]
        assert root.level == logging.INFO
    finally:
        log_queue.close()
        log_queue.join_thread()

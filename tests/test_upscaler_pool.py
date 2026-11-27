"""Tests for `upscaler.run`'s worker pool: exit codes, failures, Ctrl+C, unexpected errors.

Most tests swap the process pool for `_ScriptedExecutor`, whose futures settle as scripted
(written, raising, or never started), so every outcome is deterministic and needs no process.
One test runs the real pool with the smallest bundled model, end to end.
"""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from logging.handlers import QueueListener
from typing import TYPE_CHECKING, final

import cv2
import numpy as np
import pytest
import requests

from curios_comic_crawler import upscaler
from curios_comic_crawler._exit_codes import INTERRUPTED_EXIT_CODE

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from conftest import ConfigFactory

    from curios_comic_crawler.config import AppConfig

PENDING = None
"""Script entry for a page a worker never started (it stays pending until cancelled)."""


@final
class _ScriptedExecutor:
    """Stands in for `ProcessPoolExecutor`: futures settle per a script keyed by page name."""

    def __init__(self, script: "dict[str, float | BaseException | None]") -> None:
        """Set the script.

        Args:
            script: Per page name: its processing time if written, the exception it raises,
                or `PENDING`.
        """
        self.script: dict[str, float | BaseException | None] = script
        self.submitted: list[tuple[object, ...]] = []
        self.shutdowns: list[dict[str, bool]] = []
        # Every future handed out, in submit order (so `shutdown` can cancel the pending ones).
        self.order: list[Future[tuple[str, float]]] = []

    def __call__(self, **_pool_options: object) -> "_ScriptedExecutor":
        """Be "constructed" by `run` in place of the real pool class.

        Args:
            **_pool_options: The pool's options (ignored).

        Returns:
            This executor.
        """
        return self

    def __enter__(self) -> "_ScriptedExecutor":
        """Enter the `with` block.

        Returns:
            This executor.
        """
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        """Leave the `with` block like the real pool: wait, without cancelling.

        Args:
            *_exc_info: The exception in flight, if any (ignored).

        Returns:
            False.
        """
        self.shutdown(wait=True)
        return False

    def submit(
        self, _function: "Callable[..., tuple[str, float]]", page: "pathlib.Path", *args: object
    ) -> "Future[tuple[str, float]]":
        """Return the page's future, settled as scripted.

        Args:
            _function: The worker function (never run).
            page: The page to process.
            *args: The function's other arguments (recorded).

        Returns:
            The future.
        """
        self.submitted.append((page, *args))
        future: Future[tuple[str, float]] = Future()
        outcome = self.script[page.name]
        if isinstance(outcome, BaseException):
            _ = future.set_running_or_notify_cancel()
            future.set_exception(outcome)
        elif outcome is not None:
            _ = future.set_running_or_notify_cancel()
            future.set_result((page.name, outcome))
        self.order.append(future)
        return future

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Record the call; cancel the futures still pending when asked to.

        Args:
            wait: Recorded only.
            cancel_futures: Cancel what hasn't started.
        """
        self.shutdowns.append({"wait": wait, "cancel_futures": cancel_futures})
        if cancel_futures:
            for future in self.order:
                _ = future.cancel()


@pytest.fixture(name="pages")
def fixture_pages(config: "AppConfig") -> "AppConfig":
    """Three downloaded pages in `config.dl_dir`.

    Args:
        config: The config whose download folder gets the pages.

    Returns:
        The config they belong to.
    """
    for index in range(3):
        (config.dl_dir / f"SA_000{index}_small.jpg").touch()
    return config


def _install(monkeypatch: "pytest.MonkeyPatch", script: dict) -> _ScriptedExecutor:
    """Put a scripted pool (and a no-op engine preparation) in place of the real ones.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        script: See `_ScriptedExecutor`.

    Returns:
        The executor `run` will use.
    """
    executor = _ScriptedExecutor(script)
    monkeypatch.setattr(upscaler, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(upscaler, "prepare_engine", lambda _config: None)
    return executor


def test_run_returns_0_when_every_page_is_written(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Every page written: success, and each was submitted with the configured options."""
    executor = _install(monkeypatch, dict.fromkeys(_names(pages), 0.5))

    assert upscaler.run(pages) == 0
    assert [args[1:] for args in executor.submitted] == [
        (pages.upscale_dir, pages.gray_values, pages.posterise_method)
    ] * 3


def test_run_returns_1_when_a_page_fails(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
) -> None:
    """One failed page is reported and makes the run fail; the others still get written."""
    script: dict[str, float | BaseException | None] = dict.fromkeys(_names(pages), 0.5)
    script["SA_0001_small.jpg"] = ValueError("corrupt")
    _ = _install(monkeypatch, script)

    assert upscaler.run(pages) == 1
    assert "1 of 3 page(s) failed" in caplog.text


def test_run_counts_every_page_left_as_failed_when_a_worker_dies(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
) -> None:
    """A dead worker (e.g. out of memory) fails the run, with its cause logged once."""
    script: dict = dict.fromkeys(_names(pages), BrokenProcessPool("killed"))
    _ = _install(monkeypatch, script)

    assert upscaler.run(pages) == 1
    assert caplog.text.count("A worker process died") == 1
    assert "3 of 3 page(s) failed" in caplog.text


def test_ctrl_c_returns_130_and_reports_written_and_cancelled(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
) -> None:
    """Ctrl+C cancels the pages not started, and accounts for every page of the run."""
    first, second, third = _names(pages)
    script: dict[str, float | BaseException | None] = {
        first: 0.5,
        second: ValueError("corrupt"),
        third: PENDING,
    }
    executor = _install(monkeypatch, script)

    def interrupted(*_args: object) -> None:
        """Be interrupted while waiting for the pages.

        Args:
            *_args: `_collect`'s arguments (ignored).

        Raises:
            KeyboardInterrupt: Always.
        """
        raise KeyboardInterrupt

    monkeypatch.setattr(upscaler, "_collect", interrupted)

    assert upscaler.run(pages) == INTERRUPTED_EXIT_CODE
    assert executor.shutdowns[0] == {"wait": True, "cancel_futures": True}
    assert "1 page(s) written, 1 failed, 1 cancelled (of 3)" in caplog.text


def test_unexpected_error_cancels_the_remaining_pages_before_propagating(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """An error no page is expected to raise ends the run now, not after every other page."""
    first, *others = _names(pages)
    executor = _install(monkeypatch, {first: MemoryError()} | dict.fromkeys(others, PENDING))

    with pytest.raises(MemoryError):
        _ = upscaler.run(pages)
    assert executor.shutdowns[0] == {"wait": True, "cancel_futures": True}


def test_log_listener_is_stopped_when_the_run_crashes(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """The listener thread forwarding worker logs never outlives a crashed run."""
    stopped: list[bool] = []

    @final
    class _SpyListener(QueueListener):
        """Records `stop` calls."""

        def stop(self) -> None:
            """Stop, and record it."""
            stopped.append(True)
            super().stop()

    monkeypatch.setattr(upscaler, "QueueListener", _SpyListener)
    _ = _install(monkeypatch, dict.fromkeys(_names(pages), MemoryError()))

    with pytest.raises(MemoryError):
        _ = upscaler.run(pages)
    assert stopped == [True]


def test_real_pool_writes_good_pages_and_fails_on_the_corrupt_one(
    make_config: "ConfigFactory", tmp_path: "pathlib.Path"
) -> None:
    """End to end with real worker processes: what reaches them works (engine, logs, errors)."""
    config = make_config(
        folder_data=str(tmp_path / "data") + "/",
        upscaler={"engine": "onnx", "onnx_model": "realesr-animevideov3-x4"},
    )
    config.dl_dir.mkdir(parents=True)
    image = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    for name in ("SA_0001_small.jpg", "SA_0002_small.jpg"):
        _ = cv2.imwrite(str(config.dl_dir / name), image)
    _ = (config.dl_dir / "SA_0003_small.jpg").write_bytes(b"not an image")

    assert upscaler.run(config) == 1
    assert sorted(path.name for path in config.upscale_dir.iterdir()) == [
        "SA_0001_small.jpg",
        "SA_0002_small.jpg",
    ]


def _names(config: "AppConfig") -> list[str]:
    """List the downloaded pages' names, in processing order.

    Args:
        config: The run's config.

    Returns:
        The names.
    """
    return sorted(path.name for path in config.dl_dir.iterdir())


def test_a_page_written_just_before_a_worker_died_still_counts_as_written(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
) -> None:
    """A broken pool fails only the pages it lost, whatever order they come out in."""
    first, second, third = _names(pages)
    script: dict[str, float | BaseException | None] = {
        first: BrokenProcessPool("killed"),
        second: BrokenProcessPool("killed"),
        third: 0.5,
    }
    executor = _install(monkeypatch, script)
    # The order `as_completed` gives a batch of finished futures in is arbitrary: fix the worst.
    monkeypatch.setattr(
        upscaler, "as_completed", lambda futures: sorted(futures, key=executor.order.index)
    )

    assert upscaler.run(pages) == 1
    assert "2 of 3 page(s) failed" in caplog.text
    assert caplog.text.count("A worker process died") == 1


def test_ctrl_c_while_the_workers_start_cancels_and_reports(
    pages: "AppConfig", monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
) -> None:
    """Ctrl+C during the submits (the first one starts the workers) is still a clean stop."""
    first, *others = _names(pages)
    executor = _install(monkeypatch, {first: PENDING} | dict.fromkeys(others, PENDING))
    submit = executor.submit

    def interrupted_second_submit(*args: object) -> "Future[tuple[str, float]]":
        """Submit the first page, then be interrupted.

        Args:
            *args: `submit`'s arguments.

        Returns:
            The first page's future.

        Raises:
            KeyboardInterrupt: From the second page on.
        """
        if executor.submitted:
            raise KeyboardInterrupt
        return submit(*args)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(executor, "submit", interrupted_second_submit)

    assert upscaler.run(pages) == INTERRUPTED_EXIT_CODE
    assert executor.shutdowns[0] == {"wait": True, "cancel_futures": True}
    assert "0 page(s) written, 0 failed, 3 cancelled (of 3)" in caplog.text


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("offline"), ValueError("bad checksum"), OSError("disk")]
)
def test_a_model_that_cant_be_prepared_fails_cleanly(
    pages: "AppConfig",
    monkeypatch: "pytest.MonkeyPatch",
    caplog: "pytest.LogCaptureFixture",
    error: Exception,
) -> None:
    """A failed model download, checksum or write is one error line and exit 1, no crash."""

    def failing_prepare(_config: "AppConfig") -> None:
        """Fail like `ensure_model` can.

        Args:
            _config: Ignored.

        Raises:
            Exception: `error`.
        """
        raise error

    monkeypatch.setattr(upscaler, "prepare_engine", failing_prepare)

    assert upscaler.run(pages) == 1
    assert "Can't prepare the opencv upscaler's model" in caplog.text
    assert not any(record.exc_info for record in caplog.records)

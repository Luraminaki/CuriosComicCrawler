#!/usr/bin/env python3
"""Upscale downloaded comic pages, then posterise and sharpen the result."""

import dataclasses
import logging
import multiprocessing
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from logging.handlers import QueueHandler, QueueListener
from typing import TYPE_CHECKING

import cv2
import requests

from curios_comic_crawler._atomic import atomic_target
from curios_comic_crawler._exit_codes import INTERRUPTED_EXIT_CODE
from curios_comic_crawler._worker_threads import available_cpus, threads_per_worker
from curios_comic_crawler.image_ops import area_posterise, sharpen_image
from curios_comic_crawler.sr_engine import build_engine, prepare_engine

if TYPE_CHECKING:
    import pathlib
    from concurrent.futures import Future
    from multiprocessing.context import BaseContext
    from multiprocessing.queues import Queue

    import numpy as np

    from curios_comic_crawler.config import AppConfig
    from curios_comic_crawler.models import PosteriseMethod
    from curios_comic_crawler.sr_engine import EngineInit, SREngine

logger = logging.getLogger(__name__)

# One worker unless configured otherwise: each worker loads its own model (peak ~382 MiB for
# the light models, ~4.2 GiB for the RRDBNet ones per full page), and a single worker already uses
# every physical core through ONNX Runtime / OpenCV threads. More workers gain little
# (measured: 2 workers ~13% faster on the light model) for that much more memory.
DEFAULT_WORKERS = 1
# What a single page can fail with: unreadable or unwritable file, undecodable image, engine
# failure (RuntimeError, see sr_engine_onnx), or OpenCV refusing the data.
_PAGE_ERRORS = (OSError, ValueError, RuntimeError, cv2.error)

# Set once per worker process by `_init_worker`, then reused for every page that process is
# given -- reloading the model per page would dwarf the actual upscaling cost.
_worker_engine: "SREngine | None" = None  # pylint: disable=invalid-name  # mutable, not a constant


def _init_worker(
    engine_init: "EngineInit", worker_count: int, log_queue: "Queue[logging.LogRecord]", level: int
) -> None:
    """Prepare a worker process: logging, CPU share, and its super-resolution engine.

    Args:
        engine_init (EngineInit): Picklable data from `prepare_engine`, describing which
            engine/model this worker should build (see `sr_engine.py`).
        worker_count (int): Number of worker processes running concurrently, used to split
            shared resources between them.
        log_queue (Queue): Where the worker sends its log records; the main process writes
            them to its own handlers (log file and console).
        level (int): The main process's log level.
    """
    # Per-process engine, set once by the pool initializer (the pool calls a plain function).
    global _worker_engine  # noqa: PLW0603  # pylint: disable=global-statement

    # Ctrl+C reaches the whole process group: only the main process handles it (cancels the
    # queue, lets running pages finish); a worker interrupted mid-page would break the pool.
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
    root = logging.getLogger()
    root.handlers = [QueueHandler(log_queue)]
    root.setLevel(level)

    # `area_posterise`/`sharpen_image` always run through cv2 in `_process_one`: without this,
    # cv2 uses every CPU for its own parallelism, oversubscribing the CPU once more than one
    # worker runs.
    cv2.setNumThreads(threads_per_worker(worker_count))
    _worker_engine = build_engine(engine_init, worker_count)


def _write_atomically(out_path: "pathlib.Path", image: "np.ndarray") -> None:
    """Write an image so that `out_path` is either absent or complete, never truncated.

    A page cut off mid-write (Ctrl+C, a killed worker) would otherwise exist with its final
    name, and be skipped as "already upscaled" forever.

    Args:
        out_path (pathlib.Path): The page's final path; its suffix picks the image format.
        image (np.ndarray): The image to write.

    Raises:
        ValueError: If OpenCV can't encode the image in that format.
    """
    encoded_ok, encoded = cv2.imencode(out_path.suffix, image)
    if not encoded_ok:
        raise ValueError(f"OpenCV can't encode {out_path.suffix!r} images: {out_path}")
    with atomic_target(out_path) as temporary:
        _ = temporary.write_bytes(encoded)  # the encoded buffer, without copying it


def _process_one(
    img_path: "pathlib.Path",
    upscale_dir: "pathlib.Path",
    gray_values: int,
    posterise_method: "PosteriseMethod",
) -> tuple[str, float]:
    """Upscale, posterise, sharpen, and save a single page. Runs in a worker process.

    Args:
        img_path (pathlib.Path): Path to the downloaded page.
        upscale_dir (pathlib.Path): Directory the processed page is saved to.
        gray_values (int): Number of posterisation clusters to reduce the image to.
        posterise_method (PosteriseMethod): How the posterisation levels are found.

    Returns:
        tuple[str, float]: The page's filename and how long it took to process, in seconds.

    Raises:
        RuntimeError: If the worker has no engine (not started through `_init_worker`).
        ValueError: If the page can't be read as an image.
    """
    if _worker_engine is None:
        raise RuntimeError("Worker process was not initialized with an engine (see _init_worker)")

    start = time.perf_counter()
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image (missing, corrupt, or not an image): {img_path}")

    result = _worker_engine.upscale(img)
    result = area_posterise(result, gray_values, method=posterise_method)
    result = sharpen_image(result)
    _write_atomically(upscale_dir / img_path.name, result)
    return img_path.name, round(time.perf_counter() - start, 2)


def _select_remaining(
    originals: list["pathlib.Path"],
    completed: list["pathlib.Path"],
) -> list["pathlib.Path"]:
    """Decide which downloaded pages still need upscaling.

    Matches by filename rather than by position/count, so a gap in `completed` (a page that
    failed, was skipped, or finished out of order under the process pool) is correctly
    detected as still-remaining instead of being silently treated as done.

    Args:
        originals (list[pathlib.Path]): Downloaded pages, sorted.
        completed (list[pathlib.Path]): Already-upscaled pages.

    Returns:
        list[pathlib.Path]: The subset of `originals` to (re)process.
    """
    completed_names = {path.name for path in completed}
    return [path for path in originals if path.name not in completed_names]


def _resolve_worker_count(configured_workers: int | None, remaining_count: int) -> int:
    """Decide how many worker processes to run, given how many pages are left to process.

    Each worker loads its own full copy of the model, so a `configured_workers` value above the
    available CPUs would spawn more model-loaded processes than can run concurrently: capped
    (with a warning) instead of trusting the config value outright.

    Args:
        configured_workers (int | None): `AppConfig.upscale_workers` (`None` means
            `DEFAULT_WORKERS`).
        remaining_count (int): Number of pages left to process.

    Returns:
        int: Number of worker processes to use, at least 1.
    """
    cpu_count = available_cpus()
    worker_count = configured_workers if configured_workers is not None else DEFAULT_WORKERS
    if worker_count > cpu_count:
        logger.warning(
            "upscale_workers=%s exceeds the %s available CPU(s); capping to %s",
            worker_count,
            cpu_count,
            cpu_count,
        )
        worker_count = cpu_count
    return max(1, min(worker_count, remaining_count))


def _estimate_eta_seconds(avg_per_page: float, remaining_count: int, worker_count: int) -> float:
    """Estimate wall-clock time left, given how many pages are left and how many run at once.

    Dividing by the full `worker_count` is only right while there's a deep enough queue to keep
    every worker busy. Once `remaining_count` drops to or below `worker_count`, every remaining
    page is already running concurrently -- there's nothing left to parallelize further, so the
    wait is about one more `avg_per_page`, not a shrinking fraction of it.

    Args:
        avg_per_page (float): Average processing time per page so far, in seconds.
        remaining_count (int): Number of pages not yet completed.
        worker_count (int): Number of worker processes running pages concurrently.

    Returns:
        float: Estimated seconds until every remaining page is done.
    """
    if remaining_count <= 0:
        return 0.0

    effective_parallelism = min(worker_count, remaining_count)
    return (avg_per_page * remaining_count) / effective_parallelism


def _format_duration(total_seconds: float) -> str:
    """Render a duration as `<d>d - <h>h:<m>m:<s>s` (negative durations read as zero).

    Args:
        total_seconds (float): The duration.

    Returns:
        str: The formatted duration.
    """
    total_seconds = max(total_seconds, 0)
    days = int(total_seconds / (60 * 60 * 24))
    hours = int(total_seconds / (60 * 60)) % 24
    minutes = int(total_seconds / 60) % 60
    seconds = int(total_seconds) % 60
    return f"{days}d - {hours}h:{minutes}m:{seconds}s"


@dataclasses.dataclass(slots=True)
class _Progress:
    """Pages done and failed so far, with the running time average for the ETA.

    Attributes:
        total (int): Pages to process.
        worker_count (int): Pages processed concurrently.
        done (int): Pages written.
        failed (int): Pages that failed or were abandoned.
        elapsed (float): Summed processing time of the pages written, in seconds.
    """

    total: int
    worker_count: int
    done: int = 0
    failed: int = 0
    elapsed: float = 0.0

    @property
    def left(self) -> int:
        """int: Pages neither written nor failed yet."""
        return self.total - self.done - self.failed

    def record(self, name: str, elapsed: float) -> None:
        """Count a written page and log it with the estimated time left.

        Args:
            name (str): The page's filename.
            elapsed (float): How long it took, in seconds.
        """
        self.done += 1
        self.elapsed += elapsed
        eta = _estimate_eta_seconds(self.elapsed / self.done, self.left, self.worker_count)
        logger.info(
            "Image %s processed in %ss (%s/%s done). Estimated remaining time: %s",
            name,
            elapsed,
            self.done,
            self.total,
            _format_duration(eta),
        )


def _collect(futures: "dict[Future[tuple[str, float]], pathlib.Path]", progress: _Progress) -> None:
    """Wait for every page and account for each outcome.

    Args:
        futures (dict[Future, pathlib.Path]): Submitted pages.
        progress (_Progress): Updated as pages complete.
    """
    pool_broken = False
    for future in as_completed(futures):
        page = futures[future]
        try:
            name, elapsed = future.result()
        except BrokenProcessPool as error:
            # Every page not written yet fails the same way, in any order (a page written just
            # before the crash may still come after): count each, report the cause once.
            progress.failed += 1
            if not pool_broken:
                pool_broken = True
                logger.error(
                    "A worker process died (%s): the pages not written yet are lost. With a "
                    + "heavy model or several workers, this is usually a lack of memory (see "
                    + "upscale_workers).",
                    error,
                )
            continue
        except _PAGE_ERRORS:  # pylint: disable=catching-non-exception  # cv2.error is one, pylint can't see it
            progress.failed += 1
            logger.exception("Page %s failed to upscale", page.name)
            continue
        progress.record(name, elapsed)


def _report_interrupt(futures: "dict[Future[tuple[str, float]], pathlib.Path]", total: int) -> None:
    """Log how many pages were written, failed and cancelled, once the pool has stopped.

    Pages already handed to a worker finish after Ctrl+C (their output is written
    atomically), so they count as written; pages not submitted yet count as cancelled.

    Args:
        futures (dict[Future, pathlib.Path]): Every submitted page, all of them settled.
        total (int): Pages the run had to process.
    """
    settled = [future for future in futures if not future.cancelled()]
    failed = sum(1 for future in settled if future.exception() is not None)
    written = len(settled) - failed
    logger.warning(
        "Interrupted: %s page(s) written, %s failed, %s cancelled (of %s)",
        written,
        failed,
        total - written - failed,
        total,
    )


def _pages_to_upscale(config: "AppConfig", *, force: bool) -> list["pathlib.Path"]:
    """List the downloaded pages that still need upscaling (all of them with `force`).

    Args:
        config (AppConfig): Application configuration.
        force (bool): Reprocess every downloaded page.

    Returns:
        list[pathlib.Path]: The pages, sorted.
    """
    config.dl_dir.mkdir(exist_ok=True, parents=True)
    config.upscale_dir.mkdir(exist_ok=True, parents=True)
    originals = sorted(config.dl_dir.glob(f"*{config.ext}"))
    if force:
        logger.info("Force re-upscale requested, reprocessing all %s page(s)", len(originals))
        return originals
    completed = list(config.upscale_dir.glob(f"*{config.ext}"))
    return _select_remaining(originals, completed)


def _pool_context() -> "BaseContext":
    """Choose how worker processes start: a fresh server process, never a fork of this one.

    Forking would copy the running log listener thread's locks (a deadlock risk); Python 3.14
    makes forkserver the default on Linux anyway, so every version behaves the same. The server
    imports `_ignore_sigint` first, so it and every worker ignore Ctrl+C from the start.
    Windows only has spawn (each worker then ignores Ctrl+C from `_init_worker`).

    Returns:
        BaseContext: The multiprocessing context for the pool and its log queue.
    """
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("spawn")
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(["curios_comic_crawler._ignore_sigint", "__main__"])
    return context


def _submit_all(
    executor: ProcessPoolExecutor, pages: list["pathlib.Path"], config: "AppConfig"
) -> "dict[Future[tuple[str, float]], pathlib.Path]":
    """Hand every page to the pool.

    Args:
        executor (ProcessPoolExecutor): The pool.
        pages (list[pathlib.Path]): The pages to process.
        config (AppConfig): Application configuration.

    Returns:
        dict[Future, pathlib.Path]: Each page's future.
    """
    return {
        executor.submit(
            _process_one, page, config.upscale_dir, config.gray_values, config.posterise_method
        ): page
        for page in pages
    }


def _process_pages(
    executor: ProcessPoolExecutor,
    pages: list["pathlib.Path"],
    config: "AppConfig",
    worker_count: int,
) -> int:
    """Hand every page to the pool, then wait for them all.

    Args:
        executor (ProcessPoolExecutor): The started pool.
        pages (list[pathlib.Path]): The pages to process.
        config (AppConfig): Application configuration.
        worker_count (int): The pool's size (for the time estimate).

    Returns:
        int: 0 if every page was written, 1 if any failed, 130 if interrupted.
    """
    progress = _Progress(total=len(pages), worker_count=worker_count)
    futures: dict[Future[tuple[str, float]], pathlib.Path] = {}
    try:
        # Submitted inside the `try`: the first submit starts the workers, which takes a while.
        futures.update(_submit_all(executor, pages, config))
        _collect(futures, progress)
    except KeyboardInterrupt:
        logger.warning("Interrupted: cancelling the pages not started yet...")
        executor.shutdown(wait=True, cancel_futures=True)
        _report_interrupt(futures, len(pages))
        return INTERRUPTED_EXIT_CODE
    except BaseException:  # re-raised: only drops the queued pages before the crash
        # Leaving the `with` would otherwise process every remaining page first.
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    if progress.failed:
        logger.warning("%s of %s page(s) failed to upscale", progress.failed, progress.total)
        return 1
    return 0


def run(config: "AppConfig", *, force: bool = False) -> int:
    """Upscale comic pages that haven't been upscaled yet.

    Pages are processed in worker processes (see `AppConfig.upscale_workers`); every worker
    loads its own copy of the model once and reuses it for every page it's given. Ctrl+C
    cancels the pages not started yet; the running ones finish (each output is written
    atomically, so none is left truncated).

    Args:
        config (AppConfig): Application configuration.
        force (bool, optional): If True, re-upscale every downloaded page, ignoring what has
            already been upscaled (existing files are overwritten). Defaults to False.

    Returns:
        int: 0 on success (including when there is nothing left to process), 1 if the model
        can't be prepared, any page failed or a worker died, 130 if interrupted.
    """
    remaining = _pages_to_upscale(config, force=force)
    if not remaining:
        logger.info("Nothing to upscale in %s", config.dl_dir)
        return 0

    try:
        engine_init = prepare_engine(config)
    except (requests.RequestException, OSError, ValueError) as error:  # download, disk, checksum
        logger.error("Can't prepare the %s upscaler's model -- %s", config.upscaler.engine, error)
        return 1
    worker_count = _resolve_worker_count(config.upscale_workers, len(remaining))
    logger.info(
        "Using upscaler engine=%s, posterise=%s, across %s worker process(es)",
        config.upscaler.engine,
        config.posterise_method,
        worker_count,
    )
    context = _pool_context()
    root = logging.getLogger()
    log_queue: Queue[logging.LogRecord] = context.Queue()
    listener = QueueListener(log_queue, *root.handlers, respect_handler_level=True)
    listener.start()
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_init_worker,
            initargs=(engine_init, worker_count, log_queue, root.level),
        ) as executor:
            return _process_pages(executor, remaining, config, worker_count)
    finally:
        listener.stop()
        log_queue.close()

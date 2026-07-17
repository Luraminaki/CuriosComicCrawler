#!/usr/bin/env python3
"""Upscale downloaded comic pages, then posterise and sharpen the result."""

import logging
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import cv2

from curios_comic_crawler._worker_threads import threads_per_worker
from curios_comic_crawler.config import AppConfig
from curios_comic_crawler.image_ops import area_posterise, sharpen_image
from curios_comic_crawler.sr_engine import EngineInit, SREngine, build_engine, prepare_engine

logger = logging.getLogger(__name__)

# How many batches' worth of futures to keep in flight per worker (see `run`'s `batch_size`) --
# large enough that no worker ever waits idle for the next batch, small enough to bound memory
# for a very large backlog.
_MAX_IN_FLIGHT_BATCHES_PER_WORKER = 4

# Set once per worker process by `_init_worker`, then reused for every page that process is
# given -- reloading the model per page would dwarf the actual upscaling cost.
_worker_engine: SREngine | None = None


def _init_worker(engine_init: EngineInit, worker_count: int) -> None:
    """Build this worker process's super-resolution engine once, for reuse across pages.

    Args:
        engine_init (EngineInit): Picklable data from `prepare_engine`, describing which
            engine/model this worker should build (see `sr_engine.py`).
        worker_count (int): Number of worker processes running concurrently, used to split
            shared resources between them.
    """
    global _worker_engine  # noqa: PLW0603

    # `area_posterise`/`sharpen_image` always run through cv2 in `_process_one`, regardless of
    # which SR engine produced the upscaled image -- without this, cv2 defaults to using every
    # CPU thread for its own internal parallelism (e.g. `cv2.kmeans`), oversubscribing the CPU
    # once more than one worker process is running concurrently.
    cv2.setNumThreads(threads_per_worker(worker_count))

    _worker_engine = build_engine(engine_init, worker_count)


def _process_one(img_path: pathlib.Path, upscale_dir: pathlib.Path, gray_values: int) -> tuple[str, float]:
    """Upscale, posterise, sharpen, and save a single page. Runs in a worker process.

    Args:
        img_path (pathlib.Path): Path to the downloaded page.
        upscale_dir (pathlib.Path): Directory the processed page is saved to.
        gray_values (int): Number of posterisation clusters to reduce the image to.

    Returns:
        tuple[str, float]: The page's filename and how long it took to process, in seconds.
    """
    if _worker_engine is None:
        raise RuntimeError('Worker process was not initialized with an engine (see _init_worker)')

    tic = time.time()

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Failed to read image (missing, corrupt, or not an image): {img_path}')

    result = _worker_engine.upscale(img)
    result = area_posterise(result, gray_values)
    result = sharpen_image(result)

    out_path = upscale_dir / img_path.name
    if not cv2.imwrite(str(out_path), result):
        raise OSError(f'Failed to write upscaled image (unsupported extension, full disk, or permissions): {out_path}')

    return img_path.name, round(time.time() - tic, 2)


def _select_remaining(
    originals: list[pathlib.Path], completed: list[pathlib.Path], *, force: bool,
) -> list[pathlib.Path]:
    """Decide which downloaded pages still need upscaling.

    Matches by filename rather than by position/count, so a gap in `completed` (a page that
    failed, was skipped, or finished out of order under the process pool) is correctly
    detected as still-remaining instead of being silently treated as done.

    Args:
        originals (list[pathlib.Path]): Downloaded pages, sorted.
        completed (list[pathlib.Path]): Already-upscaled pages, sorted.
        force (bool): If True, every downloaded page is reprocessed regardless of `completed`.

    Returns:
        list[pathlib.Path]: The subset of `originals` to (re)process.
    """
    if force:
        return originals
    completed_names = {path.name for path in completed}
    return [path for path in originals if path.name not in completed_names]


def _resolve_worker_count(configured_workers: int | None, remaining_count: int) -> int:
    """Decide how many worker processes to run, given how many pages are left to process.

    Each worker loads its own full copy of the super-resolution model, so a `configured_workers`
    value above the machine's CPU count would spawn more model-loaded processes than can
    actually run concurrently -- capped here (with a warning) instead of trusting the config
    value outright.

    Args:
        configured_workers (int | None): `AppConfig.upscale_workers` (`None` means "one per
            CPU core").
        remaining_count (int): Number of pages left to process.

    Returns:
        int: Number of worker processes to use, at least 1.
    """
    cpu_count = os.cpu_count() or 1
    worker_count = configured_workers if configured_workers is not None else cpu_count

    if worker_count > cpu_count:
        logger.warning(
            'upscale_workers=%s exceeds the %s available CPU core(s); capping to %s to avoid '
            'spawning more model-loaded worker processes than the machine can run concurrently',
            worker_count, cpu_count, cpu_count,
        )
        worker_count = cpu_count

    return min(worker_count, remaining_count)


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
    total_seconds = max(total_seconds, 0)
    days = int(total_seconds / (60 * 60 * 24))
    hours = int(total_seconds / (60 * 60)) % 24
    minutes = int(total_seconds / 60) % 60
    seconds = int(total_seconds) % 60
    return f'{days}d - {hours}h:{minutes}m:{seconds}s'


def run(config: AppConfig, *, force: bool = False) -> int:
    """Upscale comic pages that haven't been upscaled yet.

    Pages are processed in parallel, each in its own worker process (see
    `AppConfig.upscale_workers`); every worker loads its own copy of the model once and reuses
    it for every page it's given.

    Args:
        config (AppConfig): Application configuration.
        force (bool, optional): If True, re-upscale every downloaded page, ignoring what has
            already been upscaled (existing files are overwritten). Defaults to False.

    Returns:
        int: 0 on success (including when there is nothing left to process).
    """
    config.dl_dir.mkdir(exist_ok=True, parents=True)
    config.upscale_dir.mkdir(exist_ok=True, parents=True)

    originals = sorted(config.dl_dir.glob(f'*{config.ext}'))
    completed = [] if force else sorted(config.upscale_dir.glob(f'*{config.ext}'))
    remaining = _select_remaining(originals, completed, force=force)

    if force:
        logger.info('Force re-upscale requested, reprocessing all %s downloaded page(s)', len(originals))

    if not remaining:
        logger.info('Nothing to upscale in %s', config.dl_dir)
        return 0

    engine_init = prepare_engine(config)

    worker_count = _resolve_worker_count(config.upscale_workers, len(remaining))
    logger.info(
        'Using upscaler engine=%s across %s worker process(es)',
        config.upscaler.engine, worker_count,
    )

    completed_count = 0
    failed_count = 0
    total_elapsed = 0.0
    pool_broken = False

    # Pages are submitted in batches rather than all at once: for a very large backlog, queuing
    # every remaining page upfront would hold one `Future` (and its pickled call args) per page
    # in memory at the same time, for no benefit -- only `worker_count` can run at once anyway.
    batch_size = worker_count * _MAX_IN_FLIGHT_BATCHES_PER_WORKER

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(engine_init, worker_count),
    ) as executor:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            futures = [
                executor.submit(_process_one, img_path, config.upscale_dir, config.gray_values)
                for img_path in batch
            ]

            for future in as_completed(futures):
                try:
                    name, elapsed = future.result()
                except BrokenProcessPool:
                    # A worker crashed while initializing its engine (e.g. out of memory loading
                    # a heavy model across many worker processes) -- every other pending future
                    # will raise the same exception, so report the root cause once instead of
                    # once per remaining page.
                    still_pending = len(remaining) - completed_count - failed_count
                    logger.error(
                        'Worker pool crashed while initializing the upscale engine (likely out '
                        'of memory) -- aborting the remaining %s page(s) instead of reporting '
                        'them individually', still_pending,
                    )
                    failed_count += still_pending
                    pool_broken = True
                    break
                except Exception:
                    failed_count += 1
                    logger.exception('A page failed to upscale')
                    continue

                completed_count += 1
                total_elapsed += elapsed

                avg_per_page = total_elapsed / completed_count
                remaining_count = len(remaining) - completed_count - failed_count
                eta_seconds = _estimate_eta_seconds(avg_per_page, remaining_count, worker_count)

                logger.info(
                    'Image %s processed in %ss (%s/%s done). Estimated remaining time: %s',
                    name, elapsed, completed_count, len(remaining), _format_duration(eta_seconds),
                )

            if pool_broken:
                break

    if failed_count:
        logger.warning('%s of %s page(s) failed to upscale', failed_count, len(remaining))
        return 1

    return 0

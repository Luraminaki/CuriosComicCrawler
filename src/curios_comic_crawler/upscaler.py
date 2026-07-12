#!/usr/bin/env python3
"""Upscale downloaded comic pages, then posterise and sharpen the result."""

import logging
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

from .config import AppConfig
from .image_ops import area_posterise, sharpen_image
from .model_registry import MODEL_MANIFEST, ensure_model

logger = logging.getLogger(__name__)

# Set once per worker process by `_init_worker`, then reused for every page that process is
# given -- reloading the model per page would dwarf the actual upscaling cost.
_worker_sup_res = None


def _init_worker(model_path: pathlib.Path, algorithm: str, scale: int) -> None:
    """Load the super-resolution model once per worker process.

    Args:
        model_path (pathlib.Path): Local path to the model weight file.
        algorithm (str): Name passed to `DnnSuperResImpl.setModel`, e.g. `"edsr"`.
        scale (int): Upscaling factor the model was trained for.
    """
    global _worker_sup_res  # noqa: PLW0603

    # Parallelism now comes from running multiple worker *processes*; without this, each
    # process would also spin up its own internal OpenCV thread pool on top of that,
    # oversubscribing the CPU.
    cv2.setNumThreads(1)

    # cv2's type stubs don't cover the dnn_superres contrib module.
    sup_res = cv2.dnn_superres.DnnSuperResImpl_create()  # pyright: ignore[reportAttributeAccessIssue]
    sup_res.readModel(str(model_path))
    sup_res.setModel(algorithm, scale)
    sup_res.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    sup_res.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    _worker_sup_res = sup_res


def _process_one(img_path: pathlib.Path, upscale_dir: pathlib.Path, gray_values: int) -> tuple[str, float]:
    """Upscale, posterise, sharpen, and save a single page. Runs in a worker process.

    Args:
        img_path (pathlib.Path): Path to the downloaded page.
        upscale_dir (pathlib.Path): Directory the processed page is saved to.
        gray_values (int): Number of posterisation clusters to reduce the image to.

    Returns:
        tuple[str, float]: The page's filename and how long it took to process, in seconds.
    """
    if _worker_sup_res is None:
        raise RuntimeError('Worker process was not initialized with a model (see _init_worker)')

    tic = time.time()

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Failed to read image (missing, corrupt, or not an image): {img_path}')

    result = _worker_sup_res.upsample(img)
    result = area_posterise(result, gray_values)
    result = sharpen_image(result)
    cv2.imwrite(str(upscale_dir / img_path.name), result)

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

    model_path = ensure_model(config.models_dir, config.model_name, config.model_scale)
    model_spec = MODEL_MANIFEST[(config.model_name, config.model_scale)]

    worker_count = _resolve_worker_count(config.upscale_workers, len(remaining))
    logger.info(
        'Using model %s with %sx scaling across %s worker process(es)',
        model_spec.name, model_spec.scale, worker_count,
    )

    completed_count = 0
    failed_count = 0
    total_elapsed = 0.0

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(model_path, model_spec.algorithm, model_spec.scale),
    ) as executor:
        futures = [
            executor.submit(_process_one, img_path, config.upscale_dir, config.gray_values)
            for img_path in remaining
        ]

        for future in as_completed(futures):
            try:
                name, elapsed = future.result()
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

    if failed_count:
        logger.warning('%s of %s page(s) failed to upscale', failed_count, len(remaining))
        return 1

    return 0

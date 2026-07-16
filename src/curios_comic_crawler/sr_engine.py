#!/usr/bin/env python3
"""Pluggable super-resolution backends for the upscaler.

`upscaler.py` only ever talks to the `SREngine` protocol below; which concrete implementation
backs it (OpenCV's classic photo-trained models, or `realesrgan-ncnn-py`'s illustration-tuned
ones) is decided by `AppConfig.upscaler`. Adding a third engine means implementing `SREngine`
plus a `prepare()`/`build()` pair in a new module, and adding one member to the
`UpscaleConfig` union in `config.py` -- nothing here or in `upscaler.py` needs to change.

The `prepare()`/`build()` split exists because engine setup that needs network I/O (the OpenCV
engine's model download+integrity check) must happen exactly once, in the main process, before
worker processes are forked -- not once per worker. `prepare_engine()` runs in the main process
and returns a small picklable value; `build_engine()` runs inside each worker (called from
`upscaler._init_worker`) and turns that value into a live engine instance.
"""

from typing import Protocol

import numpy as np

from curios_comic_crawler import sr_engine_ncnn, sr_engine_opencv
from curios_comic_crawler.config import AppConfig, NcnnUpscaleConfig, OpenCVUpscaleConfig

EngineInit = sr_engine_opencv.OpenCVEngineInit | sr_engine_ncnn.NcnnEngineInit


class SREngine(Protocol):
    """A super-resolution backend: takes a BGR image, returns an upscaled BGR image."""

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale `image` (BGR, as read by `cv2.imread`) and return the result.

        Args:
            image (np.ndarray): Image to upscale

        Returns:
            np.ndarray: Upscaled image
        """
        ...


def prepare_engine(config: AppConfig) -> EngineInit:
    """Run the configured engine's main-process setup (e.g. downloading a model).

    Must be called once, before any worker process is forked.

    Args:
        config (AppConfig): Application configuration.

    Returns:
        EngineInit: Picklable data to pass to `build_engine` in each worker process.
    """
    upscaler_config = config.upscaler
    if isinstance(upscaler_config, OpenCVUpscaleConfig):
        return sr_engine_opencv.prepare(upscaler_config, config.models_dir)
    if isinstance(upscaler_config, NcnnUpscaleConfig):
        return sr_engine_ncnn.prepare(upscaler_config)
    raise AssertionError(f'Unhandled upscaler config: {upscaler_config!r}')  # pragma: no cover


def build_engine(engine_init: EngineInit, worker_count: int) -> SREngine:
    """Build the live `SREngine` a worker process will use for every page it's given.

    Args:
        engine_init (EngineInit): This engine's `prepare_engine()` output.
        worker_count (int): Number of worker processes running concurrently (only meaningful
            to engines that need to divide CPU resources between them, e.g. OpenCV's thread
            pool).

    Returns:
        SREngine: A ready-to-use engine instance, local to the calling worker process.
    """
    if isinstance(engine_init, sr_engine_opencv.OpenCVEngineInit):
        return sr_engine_opencv.build(engine_init, worker_count)
    if isinstance(engine_init, sr_engine_ncnn.NcnnEngineInit):
        return sr_engine_ncnn.build(engine_init)
    raise AssertionError(f'Unhandled engine init: {engine_init!r}')  # pragma: no cover

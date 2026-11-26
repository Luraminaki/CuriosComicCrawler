#!/usr/bin/env python3
"""Pluggable super-resolution backends for the upscaler.

`upscaler.py` only ever talks to the `SREngine` protocol below; which concrete implementation
backs it (OpenCV's classic photo-trained models, or the bundled ONNX Runtime model tuned for
illustration/anime art) is decided by `AppConfig.upscaler`. Adding a third engine means
implementing `SREngine` and a `prepare()` function in a new module, adding its config to the
`UpscaleConfig` union in `config.py`, its init type to `EngineInit`, and one case to each
`match` below (basedpyright's exhaustiveness check flags a missing one); `upscaler.py` doesn't
change.

The `prepare_engine()`/`build_engine()` split exists because engine setup that needs network I/O
(the OpenCV engine's model download+integrity check) must happen exactly once, in the main
process, before worker processes are forked -- not once per worker. `prepare_engine()` runs in
the main process and returns a small picklable value; `build_engine()` runs inside each worker
(called from `upscaler._init_worker`) and turns that value into a live engine instance.
"""

from typing import TYPE_CHECKING, Protocol

from curios_comic_crawler import sr_engine_onnx, sr_engine_opencv
from curios_comic_crawler.config import OnnxUpscaleConfig, OpenCVUpscaleConfig

if TYPE_CHECKING:
    import numpy as np

    from curios_comic_crawler.config import AppConfig

EngineInit = sr_engine_opencv.OpenCVEngineInit | sr_engine_onnx.OnnxEngineInit


class SREngine(Protocol):
    """A super-resolution backend: takes a BGR image, returns an upscaled BGR image."""

    def upscale(self, image: "np.ndarray") -> "np.ndarray":
        """Upscale `image` (BGR, as read by `cv2.imread`) and return the result.

        Args:
            image (np.ndarray): Image to upscale

        Returns:
            np.ndarray: Upscaled image
        """
        ...


def prepare_engine(config: "AppConfig") -> EngineInit:
    """Run the configured engine's main-process setup (e.g. downloading a model).

    Must be called once, before any worker process is forked.

    Args:
        config (AppConfig): Application configuration.

    Returns:
        EngineInit: Picklable data to pass to `build_engine` in each worker process.
    """
    match config.upscaler:
        case OpenCVUpscaleConfig() as upscaler_config:
            return sr_engine_opencv.prepare(upscaler_config, config.models_dir)
        case OnnxUpscaleConfig() as upscaler_config:
            return sr_engine_onnx.prepare(upscaler_config)


def build_engine(engine_init: EngineInit, worker_count: int) -> SREngine:
    """Build the live `SREngine` a worker process will use for every page it's given.

    Args:
        engine_init (EngineInit): This engine's `prepare_engine()` output.
        worker_count (int): Number of worker processes running concurrently (only meaningful
            to engines that divide CPU threads between them, e.g. the onnx engine's intra-op
            threads).

    Returns:
        SREngine: A ready-to-use engine instance, local to the calling worker process.
    """
    match engine_init:
        case sr_engine_opencv.OpenCVEngineInit():
            return sr_engine_opencv.OpenCVEngine(engine_init)
        case sr_engine_onnx.OnnxEngineInit():
            return sr_engine_onnx.OnnxEngine(engine_init, worker_count)

#!/usr/bin/env python3
"""SREngine implementation backed by `realesrgan-ncnn-py` (illustration/anime-tuned models).

Runs on CPU only (`gpuid=-1`) -- no Vulkan driver dependency. Models are bundled inside the
`realesrgan-ncnn-py` wheel, so unlike the OpenCV engine there is nothing to download or verify.
Requires the `ncnn` extra: `pip install curios-comic-crawler[ncnn]`.
"""

from typing import NamedTuple

import numpy as np

from curios_comic_crawler.config import NcnnUpscaleConfig
from curios_comic_crawler.models import NcnnModelName

# Index `realesrgan_ncnn_py.Realesrgan(model=...)` expects, per its README.
_MODEL_INDEX: dict[NcnnModelName, int] = {
    'realesr-animevideov3-x2': 0,
    'realesr-animevideov3-x3': 1,
    'realesr-animevideov3-x4': 2,
    'realesrgan-x4plus-anime': 3,
    'realesrgan-x4plus': 4,
}


class NcnnEngineInit(NamedTuple):
    """Picklable data a worker process needs to build an `NcnnEngine`."""

    model: NcnnModelName


def _import_realesrgan() -> type:
    """Import and return `realesrgan_ncnn_py.Realesrgan`.

    Raises:
        RuntimeError: If `realesrgan-ncnn-py` (the `ncnn` extra) isn't installed.
    """
    try:
        from realesrgan_ncnn_py import Realesrgan  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            'The "ncnn" upscale engine requires the optional realesrgan-ncnn-py package. '
            'Install it with: pip install curios-comic-crawler[ncnn]',
        ) from error
    return Realesrgan


class NcnnEngine:
    """Upscales images with `realesrgan_ncnn_py.Realesrgan`."""

    def __init__(self, engine_init: NcnnEngineInit) -> None:
        """Build a CPU-only `Realesrgan` instance for the model described by `engine_init`.

        Args:
            engine_init (NcnnEngineInit): `prepare()`'s output for this model.

        Raises:
            RuntimeError: If `realesrgan-ncnn-py` (the `ncnn` extra) isn't installed.
        """
        realesrgan_cls = _import_realesrgan()
        self._realesrgan = realesrgan_cls(gpuid=-1, model=_MODEL_INDEX[engine_init.model])

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale `image` with the loaded model."""
        return self._realesrgan.process_cv2(image)


def prepare(upscaler_config: NcnnUpscaleConfig) -> NcnnEngineInit:
    """Check the engine is usable and describe the configured model.

    Runs in the main process, before any worker is forked, so a missing `realesrgan-ncnn-py`
    install fails fast with one clear error instead of breaking every worker process's
    `ProcessPoolExecutor` initializer (which surfaces as an opaque `BrokenProcessPool` for
    every single page instead of the actual cause). Nothing to download here otherwise --
    models ship inside the wheel.

    Args:
        upscaler_config (NcnnUpscaleConfig): The `engine: "ncnn"` config section.

    Returns:
        NcnnEngineInit: Picklable data passed to every worker process's `build()` call.

    Raises:
        RuntimeError: If `realesrgan-ncnn-py` (the `ncnn` extra) isn't installed.
    """
    _import_realesrgan()
    return NcnnEngineInit(model=upscaler_config.ncnn_model)


def build(engine_init: NcnnEngineInit) -> NcnnEngine:
    """Build a worker process's `NcnnEngine` from `prepare()`'s output."""
    return NcnnEngine(engine_init)

#!/usr/bin/env python3
"""SREngine implementation backed by ONNX Runtime, using bundled models.

Runs on CPU (`CPUExecutionProvider`) with no extra runtime dependency beyond `onnxruntime`
itself, which is a base dependency of this package (lightweight, actively maintained, wheels
for every current Python version -- unlike the ncnn-based engines this replaces). Each model is
a file bundled directly in this package under `assets/`, converted once from official
xinntao/Real-ESRGAN weights via `tools/convert_onnx_model.py`. Nothing to download, no separate
install extra, no external hosting or sha256-pinning needed for it.
"""

import pathlib
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    RuntimeException,
)

from curios_comic_crawler._worker_threads import threads_per_worker

if TYPE_CHECKING:
    from curios_comic_crawler.config import OnnxUpscaleConfig
    from curios_comic_crawler.models import OnnxModelName

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

_MODEL_FILENAMES: dict["OnnxModelName", str] = {
    "realesr-animevideov3-x4": "realesr-animevideov3.onnx",
    "realesrgan-x4plus-anime-6b": "realesrgan-x4plus-anime-6b.onnx",
    "realesr-general-x4v3": "realesr-general-x4v3.onnx",
    "realesrgan-x4plus": "realesrgan-x4plus.onnx",
}


# ONNX Runtime's own errors derive from `Exception` directly, not `RuntimeError`.
_ORT_ERRORS = (Fail, InvalidArgument, RuntimeException)


class OnnxEngineInit(NamedTuple):
    """Picklable data a worker process needs to build an `OnnxEngine`."""

    model_path: pathlib.Path


class OnnxEngine:
    """Upscales images with an `onnxruntime.InferenceSession`."""

    def __init__(self, engine_init: OnnxEngineInit, worker_count: int) -> None:
        """Load the bundled ONNX model described by `engine_init`.

        Args:
            engine_init (OnnxEngineInit): `prepare()`'s output for this model.
            worker_count (int): Number of worker processes running concurrently. Without this,
                each process's `InferenceSession` defaults to using every CPU thread for its
                own intra-op parallelism, oversubscribing the CPU once more than one worker is
                running -- same issue as `sr_engine_opencv.py`'s `cv2.setNumThreads` call.
        """
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = threads_per_worker(worker_count)
        # The CPU memory arena keeps every buffer it ever allocated for reuse: on full-page
        # inputs it held ~40% more memory than the page needs (measured, see README).
        session_options.enable_cpu_mem_arena = False

        self._session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            str(engine_init.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale `image` (BGR, as read by `cv2.imread`) with the loaded model.

        Follows Real-ESRGAN's convention: the model takes RGB `float32` in `[0, 1]`, as a
        `(1, 3, H, W)` tensor, and returns the same layout. Conversions are done in place
        where possible: on a full page, each extra copy is hundreds of megabytes.

        Args:
            image (np.ndarray): `uint8` BGR image, `(H, W, 3)`.

        Returns:
            np.ndarray: The upscaled `uint8` BGR image.

        Raises:
            RuntimeError: If ONNX Runtime fails (e.g. out of memory).
        """
        tensor = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, swapRB=True)
        try:
            (output,) = self._session.run(None, {"input": tensor})
        except _ORT_ERRORS as error:  # failure inside the runtime: a page-level RuntimeError
            raise RuntimeError(f"ONNX Runtime failed to upscale the image: {error}") from error

        planes = output[0]  # (3, H, W) float32, RGB  # pyright: ignore[reportIndexIssue]
        # The model overshoots [0, 1] a little; clamp first, since cv2's float to uint8
        # conversion wraps far-out values instead of saturating them (and, once clamped, the
        # "abs" of convertScaleAbs is a no-op). Scaling each plane straight to uint8 and merging
        # them in BGR order gives the same pixels as scaling, rounding and casting in float with
        # NumPy, 6x faster and without a float copy of the image (measured on a full page).
        _ = np.clip(planes, 0.0, 1.0, out=planes)
        return cv2.merge(
            [cv2.convertScaleAbs(planes[channel], alpha=255.0) for channel in (2, 1, 0)]
        )


def prepare(upscaler_config: "OnnxUpscaleConfig") -> OnnxEngineInit:
    """Resolve the bundled model's on-disk path. Nothing to download or verify.

    Args:
        upscaler_config (OnnxUpscaleConfig): The `engine: "onnx"` config section.

    Returns:
        OnnxEngineInit: Picklable data passed to every worker process's `build()` call.

    Raises:
        FileNotFoundError: If the bundled model asset is missing (a packaging bug).
    """
    model_path = _ASSETS_DIR / _MODEL_FILENAMES[upscaler_config.onnx_model]
    if not model_path.is_file():
        raise FileNotFoundError(f"Bundled ONNX model not found: {model_path}")
    return OnnxEngineInit(model_path=model_path)

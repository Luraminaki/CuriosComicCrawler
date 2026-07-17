#!/usr/bin/env python3
"""SREngine implementation backed by ONNX Runtime, using bundled models.

Runs on CPU (`CPUExecutionProvider`) with no extra runtime dependency beyond `onnxruntime`
itself, which is a base dependency of this package (lightweight, actively maintained, wheels
for every current Python version -- unlike the ncnn-based engines this replaces). Each model is
a file bundled directly in this package under `assets/`, converted once from official
xinntao/Real-ESRGAN weights via `tools/convert_onnx_model.py`. Nothing to download, no separate
install extra, no external hosting or sha256-pinning needed for it.
"""

import os
import pathlib
from typing import NamedTuple

import numpy as np
import onnxruntime

from curios_comic_crawler.config import OnnxUpscaleConfig
from curios_comic_crawler.models import OnnxModelName

_ASSETS_DIR = pathlib.Path(__file__).parent / 'assets'

_MODEL_FILENAMES: dict[OnnxModelName, str] = {
    'realesr-animevideov3-x4': 'realesr-animevideov3.onnx',
    'realesrgan-x4plus-anime-6b': 'realesrgan-x4plus-anime-6b.onnx',
    'realesr-general-x4v3': 'realesr-general-x4v3.onnx',
    'realesrgan-x4plus': 'realesrgan-x4plus.onnx',
}


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
        cpu_count = os.cpu_count() or 1
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = max(1, cpu_count // worker_count)

        self._session = onnxruntime.InferenceSession(
            str(engine_init.model_path), sess_options=session_options, providers=['CPUExecutionProvider'],
        )

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale `image` (BGR, as read by `cv2.imread`) with the loaded model.

        Follows Real-ESRGAN's own pre/post-processing convention: BGR -> RGB, HWC `uint8`
        `[0, 255]` -> CHW `float32` `[0, 1]` with a batch dimension for input, and the reverse
        for output.
        """
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[np.newaxis, ...])

        (output,) = self._session.run(None, {'input': tensor})

        result = np.clip(output[0], 0.0, 1.0)  # pyright: ignore[reportIndexIssue]
        result = np.transpose(result, (1, 2, 0))
        result = (result * 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(result[:, :, ::-1])


def prepare(upscaler_config: OnnxUpscaleConfig) -> OnnxEngineInit:
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
        raise FileNotFoundError(f'Bundled ONNX model not found: {model_path}')
    return OnnxEngineInit(model_path=model_path)


def build(engine_init: OnnxEngineInit, worker_count: int) -> OnnxEngine:
    """Build a worker process's `OnnxEngine` from `prepare()`'s output."""
    return OnnxEngine(engine_init, worker_count)

#!/usr/bin/env python3
"""SREngine implementation backed by OpenCV's `dnn_superres` module."""

import os
import pathlib
from typing import NamedTuple

import cv2
import numpy as np

from curios_comic_crawler.config import OpenCVUpscaleConfig
from curios_comic_crawler.model_registry import MODEL_MANIFEST, ensure_model


class OpenCVEngineInit(NamedTuple):
    """Picklable data a worker process needs to build an `OpenCVEngine`."""

    model_path: pathlib.Path
    algorithm: str
    scale: int


class OpenCVEngine:
    """Upscales images with a `cv2.dnn_superres.DnnSuperResImpl`."""

    def __init__(self, engine_init: OpenCVEngineInit, worker_count: int) -> None:
        """Load the model described by `engine_init` and configure its thread budget.

        Args:
            engine_init (OpenCVEngineInit): `prepare()`'s output for this model.
            worker_count (int): Number of worker processes running concurrently, used to
                split the available CPU threads between them.
        """
        # Parallelism comes from running multiple worker *processes*; on top of that, each
        # process would also spin up its own internal OpenCV thread pool, oversubscribing the
        # CPU if left uncapped. Split the machine's threads evenly between worker processes
        # instead of always pinning each one to a single thread.
        cpu_count = os.cpu_count() or 1
        cv2.setNumThreads(max(1, cpu_count // worker_count))

        # cv2's type stubs don't cover the dnn_superres contrib module.
        sup_res = cv2.dnn_superres.DnnSuperResImpl_create()  # pyright: ignore[reportAttributeAccessIssue]
        sup_res.readModel(str(engine_init.model_path))
        sup_res.setModel(engine_init.algorithm, engine_init.scale)
        sup_res.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sup_res.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._sup_res = sup_res

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale `image` with the loaded model."""
        return self._sup_res.upsample(image)


def prepare(upscaler_config: OpenCVUpscaleConfig, models_dir: pathlib.Path) -> OpenCVEngineInit:
    """Download/verify the configured model (once, in the main process) and describe it.

    Args:
        upscaler_config (OpenCVUpscaleConfig): The `engine: "opencv"` config section.
        models_dir (pathlib.Path): Directory models are stored in and fetched into.

    Returns:
        OpenCVEngineInit: Picklable data passed to every worker process's `build()` call.
    """
    model_path = ensure_model(models_dir, upscaler_config.model_name, upscaler_config.model_scale)
    spec = MODEL_MANIFEST[(upscaler_config.model_name, upscaler_config.model_scale)]
    return OpenCVEngineInit(model_path=model_path, algorithm=spec.algorithm, scale=spec.scale)


def build(engine_init: OpenCVEngineInit, worker_count: int) -> OpenCVEngine:
    """Build a worker process's `OpenCVEngine` from `prepare()`'s output."""
    return OpenCVEngine(engine_init, worker_count)

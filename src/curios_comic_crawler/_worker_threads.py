#!/usr/bin/env python3
"""Shared CPU thread-budgeting helper for upscale worker processes.

Split out from `sr_engine_onnx.py`/`upscaler.py` (rather than living in either) so both can
import it without a circular dependency: `upscaler.py` depends on `sr_engine.py`, which itself
imports `sr_engine_onnx.py`.
"""

import os


def threads_per_worker(worker_count: int) -> int:
    """Split the machine's CPU count evenly across concurrent worker processes.

    Shared by every engine that needs to cap its own internal thread pool -- `cv2.setNumThreads`
    for the OpenCV-backed steps that run regardless of engine (`upscaler._init_worker`), and
    `onnxruntime.SessionOptions.intra_op_num_threads` for the onnx engine (`sr_engine_onnx.py`)
    -- so that running `worker_count` processes at once doesn't oversubscribe the CPU.

    Args:
        worker_count (int): Number of worker processes running concurrently.

    Returns:
        int: Number of threads this process's own thread pool should use, at least 1.
    """
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // worker_count)

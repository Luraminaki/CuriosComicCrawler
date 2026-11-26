#!/usr/bin/env python3
"""Shared CPU budgeting for upscale worker processes.

Split out from `sr_engine_onnx.py`/`upscaler.py` (rather than living in either) so both can
import it without a circular dependency: `upscaler.py` depends on `sr_engine.py`, which itself
imports `sr_engine_onnx.py`.
"""

import os


def available_cpus() -> int:
    """Count the CPUs this process may actually run on.

    `os.cpu_count()` counts the machine's CPUs, ignoring CPU affinity (`taskset`, container
    limits); the affinity set is what bounds useful parallelism. It isn't available on every
    platform (e.g. macOS, Windows), where the machine's count is the best answer.

    Returns:
        int: At least 1.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def threads_per_worker(worker_count: int) -> int:
    """Split the available CPUs evenly across concurrent worker processes.

    Shared by every engine that needs to cap its own internal thread pool -- `cv2.setNumThreads`
    for the OpenCV-backed steps that run regardless of engine (`upscaler._init_worker`), and
    `onnxruntime.SessionOptions.intra_op_num_threads` for the onnx engine (`sr_engine_onnx.py`)
    -- so that running `worker_count` processes at once doesn't oversubscribe the CPU.

    Args:
        worker_count (int): Number of worker processes running concurrently.

    Returns:
        int: Number of threads this process's own thread pool should use, at least 1.
    """
    return max(1, available_cpus() // worker_count)

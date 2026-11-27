#!/usr/bin/env python3
"""Shared CPU budgeting for upscale worker processes.

Split out from `sr_engine_onnx.py`/`upscaler.py` (rather than living in either) so both can
import it without a circular dependency: `upscaler.py` depends on `sr_engine.py`, which itself
imports `sr_engine_onnx.py`.
"""

import functools
import logging
import os
import pathlib

logger = logging.getLogger(__name__)

_CPU_TOPOLOGY = pathlib.Path("/sys/devices/system/cpu")


def available_cpus() -> int:
    """Count the CPUs this process may actually run on.

    `os.cpu_count()` counts the machine's CPUs, ignoring CPU affinity (`taskset`, a container's
    `--cpuset-cpus`); the affinity set is what bounds useful parallelism. A CPU quota (Docker's
    `--cpus`) doesn't change it, so isn't seen. Affinity isn't available on every platform
    (e.g. macOS, Windows), where the machine's count is the best answer.

    Returns:
        int: At least 1.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


@functools.cache  # the topology doesn't change during a run; each worker asks twice
def physical_cores() -> int:
    """Count the physical cores among the CPUs this process may run on.

    Hyperthreads share a core's arithmetic units: the super-resolution models gain nothing
    from them (measured: same wall time, twice the CPU time). Linux exposes which logical CPUs
    share a core; elsewhere, or if that can't be read, every logical CPU counts as a core.

    Returns:
        int: At least 1.
    """
    if not hasattr(os, "sched_getaffinity") or not _CPU_TOPOLOGY.is_dir():
        return available_cpus()
    cores: set[str] = set()
    for cpu in os.sched_getaffinity(0):
        siblings = _CPU_TOPOLOGY / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        try:
            cores.add(siblings.read_text(encoding="ascii").strip())
        except OSError as error:  # sysfs unreadable (containers, sandboxes)
            logger.debug("Can't read %s (%s): counting logical CPUs", siblings, error)
            return available_cpus()
    return max(1, len(cores))


def threads_per_worker(worker_count: int) -> int:
    """Split the physical cores evenly across concurrent worker processes.

    Shared by every engine that needs to cap its own internal thread pool -- `cv2.setNumThreads`
    for the OpenCV-backed steps that run regardless of engine (`upscaler._init_worker`), and
    `onnxruntime.SessionOptions.intra_op_num_threads` for the onnx engine (`sr_engine_onnx.py`)
    -- so that running `worker_count` processes at once doesn't oversubscribe the CPU.

    Args:
        worker_count (int): Number of worker processes running concurrently.

    Returns:
        int: Number of threads this process's own thread pool should use (its share of the
        physical cores), at least 1.
    """
    return max(1, physical_cores() // worker_count)

"""Tests that the upscaler's workers ignore Ctrl+C from the moment they start."""

# pylint: disable=protected-access  # unit tests exercise the module's private helpers

import multiprocessing
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import pytest

from curios_comic_crawler import upscaler

_FORKSERVER = "forkserver" in multiprocessing.get_all_start_methods()


def test_importing_the_module_ignores_ctrl_c() -> None:
    """The forkserver's first import makes its process ignore SIGINT."""
    probe = (
        "import signal, curios_comic_crawler._ignore_sigint; "
        + "print(signal.getsignal(signal.SIGINT) is signal.SIG_IGN)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "True"


def _ignores_ctrl_c() -> bool:
    """Tell, from inside a worker, whether Ctrl+C is ignored.

    Returns:
        Whether SIGINT's handler is "ignore".
    """
    return signal.getsignal(signal.SIGINT) is signal.SIG_IGN


@pytest.mark.skipif(not _FORKSERVER, reason="no forkserver on this platform (spawn instead)")
def test_workers_ignore_ctrl_c_before_any_initializer_runs() -> None:
    """Forked by the server, a worker already ignores Ctrl+C: no startup window is left open."""
    with ProcessPoolExecutor(max_workers=1, mp_context=upscaler._pool_context()) as executor:
        assert executor.submit(_ignores_ctrl_c).result(timeout=60)

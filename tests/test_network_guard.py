"""Tests for the suite's own network guard (`conftest.fixture_no_network`)."""

import socket
import sys

import pytest


def test_network_connections_are_refused() -> None:
    """A TCP connection fails loudly, before any packet leaves (the address is unroutable)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # were the guard broken, fail fast rather than hang
        with pytest.raises(RuntimeError, match="must not use the network"):
            sock.connect(("192.0.2.1", 80))  # TEST-NET-1: documentation-only, never routed


@pytest.mark.skipif(sys.platform != "linux", reason="autobound abstract sockets are Linux-only")
def test_local_unix_sockets_still_connect() -> None:
    """Unix sockets (multiprocessing's forkserver uses one) are local IPC, left alone."""
    with (
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock,
    ):
        server.bind("")  # an autobound abstract address: no file, no path-length limit
        server.listen(1)
        sock.connect(server.getsockname())

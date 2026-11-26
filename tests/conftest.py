"""Shared pytest fixtures."""

import logging
import socket
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from curios_comic_crawler.config import AppConfig

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator

ConfigFactory = Callable[..., AppConfig]
"""What the `make_config` fixture returns: keyword overrides in, a validated config out."""

# Unix sockets are local IPC, not network: multiprocessing's forkserver talks to its server
# through one, so the real-process pool test needs them. Absent on Windows.
_LOCAL_FAMILIES: tuple[int, ...] = (socket.AF_UNIX,) if hasattr(socket, "AF_UNIX") else ()


@pytest.fixture(name="no_network", autouse=True)
def fixture_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any network connection fail loudly: the tests must never reach the network.

    Local (Unix) sockets still connect.

    Args:
        monkeypatch: Replaces `socket.socket.connect`.
    """
    real_connect = socket.socket.connect

    def _refuse_network(sock: socket.socket, address: "tuple[object, ...] | str | bytes") -> None:
        """Connect a local socket; refuse anything else.

        Args:
            sock: The socket being connected.
            address: Where to.

        Raises:
            RuntimeError: For a network socket (not an `OSError`, which code under test may
                catch and retry: the test must fail, not quietly go on).
        """
        if sock.family in _LOCAL_FAMILIES:
            real_connect(sock, address)
            return
        raise RuntimeError(f"tests must not use the network (tried to connect to {address!r})")

    monkeypatch.setattr(socket.socket, "connect", _refuse_network)


@pytest.fixture(name="no_sleep")
def fixture_no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the pauses asked of `time.sleep` instead of sleeping.

    Args:
        monkeypatch: Replaces `time.sleep`.

    Returns:
        The pauses requested, in seconds, in order.
    """
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    return pauses


@pytest.fixture(name="restore_root_logger")
def fixture_restore_root_logger() -> "Iterator[None]":
    """Put the root logger's handlers and level back after a test that reconfigures logging.

    `logging.basicConfig(force=True)` and `upscaler._init_worker` replace the root handlers
    (pytest's capture handlers included); the handlers they added are closed here.

    Yields:
        Nothing: the test runs in between.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    for handler in root.handlers:
        if handler not in saved_handlers:
            handler.close()
    root.handlers = saved_handlers
    root.setLevel(saved_level)


@pytest.fixture(name="valid_config_dict")
def fixture_valid_config_dict() -> dict:
    """A minimal `config.json`-shaped dict that passes `AppConfig` validation.

    Returns:
        The config, a fresh dict per test (tests may edit it).
    """
    return {
        "root_site": "https://www.collectedcurios.com/",
        "BD_name": "SA",
        "padded": 4,
        "fails": 2,
        "filename_variants": ["small", "small_b"],
        "ext": ".jpg",
        "folder_data": "data/",
        "folder_save_dl": "DL/",
        "folder_save_upscale": "UPSCALE/",
        "folder_models": "models/",
        "upscaler": {"engine": "opencv", "model_name": "edsr", "model_scale": 3},
        "gray_values": 32,
        "headers": {"User-Agent": "Mozilla/5.0"},
    }


@pytest.fixture(name="make_config")
def fixture_make_config(valid_config_dict: dict) -> ConfigFactory:
    """Factory returning a valid `AppConfig`, optionally overridden field-by-field.

    Args:
        valid_config_dict: The base config.

    Returns:
        The factory.
    """

    def _make(**overrides: object) -> AppConfig:
        """Validate the base config with `overrides` applied.

        Args:
            **overrides: Config keys to replace.

        Returns:
            AppConfig: The validated config.
        """
        return AppConfig.model_validate({**valid_config_dict, **overrides})

    return _make


@pytest.fixture(name="config")
def fixture_config(make_config: ConfigFactory, tmp_path: "pathlib.Path") -> AppConfig:
    """A valid `AppConfig` rooted at a fresh `tmp_path`, with its data folders pre-created.

    Args:
        make_config: Builds the config.
        tmp_path: The test's own folder.

    Returns:
        The config.
    """
    app_config = make_config(folder_data=str(tmp_path / "data") + "/")
    app_config.dl_dir.mkdir(parents=True, exist_ok=True)
    app_config.upscale_dir.mkdir(parents=True, exist_ok=True)
    app_config.models_dir.mkdir(parents=True, exist_ok=True)
    return app_config

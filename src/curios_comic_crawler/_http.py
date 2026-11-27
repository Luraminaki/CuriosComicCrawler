#!/usr/bin/env python3
"""Shared download helpers: streaming to disk, hashing, verified atomic downloads.

Used by the page downloader, the model registry and `tools/convert_onnx_model.py`.
"""

import hashlib
from typing import TYPE_CHECKING

import requests

from curios_comic_crawler._atomic import atomic_target

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator

CHUNK_SIZE_BYTES = 256 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30


def stream_to_file(
    response: "requests.Response", path: "pathlib.Path", chunk_size: int = CHUNK_SIZE_BYTES
) -> None:
    """Write a streamed response's body to `path`, chunk by chunk.

    Args:
        response (requests.Response): An already-validated streaming response.
        path (pathlib.Path): Destination file path.
        chunk_size (int, optional): Bytes read per iteration. Defaults to `CHUNK_SIZE_BYTES`.
    """
    with path.open("wb") as target:
        for chunk in response.iter_content(chunk_size):
            _ = target.write(chunk)


def _chunks(path: "pathlib.Path") -> "Iterator[bytes]":
    """Read a file chunk by chunk (bounded memory, whatever its size).

    Args:
        path (pathlib.Path): The file.

    Yields:
        bytes: Successive chunks of at most `CHUNK_SIZE_BYTES`.
    """
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE_BYTES):
            yield chunk


def sha256_of(path: "pathlib.Path") -> str:
    """Hash a file without loading it whole into memory.

    Args:
        path (pathlib.Path): The file.

    Returns:
        str: Its SHA-256, as 64 lowercase hex digits.
    """
    digest = hashlib.sha256()
    for chunk in _chunks(path):
        digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target: "pathlib.Path", sha256: str) -> None:
    """Download a file and move it into place only if its SHA-256 matches.

    The body is streamed to a temporary file next to `target`, checked, then renamed over
    `target` (see `atomic_target`): `target` is either absent, previous, or complete and
    verified.

    Args:
        url (str): Where to download from.
        target (pathlib.Path): Final path; its folder is created if needed.
        sha256 (str): Expected SHA-256 of the content.

    Raises:
        requests.RequestException: If the download fails.
        ValueError: If the content doesn't match `sha256`.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(target) as temporary:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            _ = response.raise_for_status()
            stream_to_file(response, temporary)
        digest = sha256_of(temporary)
        if digest != sha256:
            raise ValueError(
                f"{target.name} failed its integrity check: expected sha256 {sha256}, "
                + f"got {digest}"
            )

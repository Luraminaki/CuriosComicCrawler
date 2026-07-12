#!/usr/bin/env python3
"""Shared HTTP streaming helper used by the downloader and the model registry."""

import pathlib

import requests

CHUNK_SIZE_BYTES = 256 * 1024


def stream_to_file(response: requests.Response, path: pathlib.Path, chunk_size: int = CHUNK_SIZE_BYTES) -> None:
    """Write a streamed response's body to `path`, chunk by chunk.

    Args:
        response (requests.Response): An already-validated streaming response.
        path (pathlib.Path): Destination file path.
        chunk_size (int, optional): Bytes read per iteration. Defaults to `CHUNK_SIZE_BYTES`.
    """
    with path.open('wb') as target:
        for chunk in response.iter_content(chunk_size):
            target.write(chunk)

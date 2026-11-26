"""Tests for `curios_comic_crawler.model_registry`."""

import hashlib
import re
from typing import TYPE_CHECKING, get_args

import pytest
import requests

from curios_comic_crawler import _http, model_registry
from curios_comic_crawler.models import ModelName, ModelSpec

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator


class _FakeStreamResponse:
    """A streaming response with a fixed body, successful unless given an HTTP error."""

    def __init__(self, content: bytes, error: requests.HTTPError | None = None) -> None:
        """Set the body.

        Args:
            content: Body.
            error: What `raise_for_status` raises; `None` for a 200.
        """
        self._content: bytes = content
        self._error: requests.HTTPError | None = error

    def __enter__(self) -> "_FakeStreamResponse":
        """Enter the `with` block.

        Returns:
            The response.
        """
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        """Leave the `with` block, not swallowing errors.

        Args:
            *_exc_info: The exception in flight, if any (ignored).

        Returns:
            False.
        """
        return False

    def raise_for_status(self) -> None:
        """Raise the scripted HTTP error, if any.

        Raises:
            requests.HTTPError: When the response was given one.
        """
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size: int) -> "Iterator[bytes]":
        """Yield the body in one chunk.

        Args:
            chunk_size: Ignored.

        Yields:
            The body.
        """
        del chunk_size
        yield self._content


def test_manifest_keys_match_spec_fields() -> None:
    """Each manifest key matches its spec's model name and scale."""
    for (name, scale), spec in model_registry.MODEL_MANIFEST.items():
        assert spec.name == name
        assert spec.scale == scale


def test_every_model_name_has_at_least_one_manifest_entry() -> None:
    """Every OpenCV model name offered in the config can be downloaded."""
    manifest_names = {name for name, _scale in model_registry.MODEL_MANIFEST}
    assert manifest_names == set(get_args(ModelName))


def test_manifest_filenames_are_unique() -> None:
    """No two models share a file name in the models folder."""
    filenames = [spec.filename for spec in model_registry.MODEL_MANIFEST.values()]
    assert len(filenames) == len(set(filenames))


def test_manifest_sha256_are_valid_hex_digests() -> None:
    """Every manifest checksum is a 64-digit hex SHA-256."""
    for spec in model_registry.MODEL_MANIFEST.values():
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)


def test_manifest_urls_are_https() -> None:
    """Every model is fetched over HTTPS."""
    for spec in model_registry.MODEL_MANIFEST.values():
        assert spec.url.startswith("https://")


def test_ensure_model_downloads_and_verifies_when_missing(
    tmp_path: "pathlib.Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing model is downloaded and its checksum verified."""
    content = b"fake model bytes"
    spec = ModelSpec(
        name="edsr",
        scale=99,
        filename="FAKE_x99.pb",
        url="https://example.com/fake.pb",
        algorithm="edsr",
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ("edsr", 99), spec)
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeStreamResponse(content))

    result = model_registry.ensure_model(tmp_path, "edsr", 99)

    assert result == tmp_path / "FAKE_x99.pb"
    assert result.read_bytes() == content


def test_ensure_model_raises_and_cleans_up_on_hash_mismatch(
    tmp_path: "pathlib.Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download with the wrong checksum is refused and deleted."""
    spec = ModelSpec(
        name="edsr",
        scale=98,
        filename="BAD_x98.pb",
        url="https://example.com/bad.pb",
        algorithm="edsr",
        sha256="0" * 64,
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ("edsr", 98), spec)
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_a, **_k: _FakeStreamResponse(b"not matching content"),
    )

    with pytest.raises(ValueError, match="integrity check"):
        _ = model_registry.ensure_model(tmp_path, "edsr", 98)

    assert list(tmp_path.iterdir()) == []  # neither the model nor a temporary file


def test_ensure_model_redownloads_corrupted_cache(
    tmp_path: "pathlib.Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached model with the wrong checksum is downloaded again."""
    good_content = b"good bytes"
    spec = ModelSpec(
        name="edsr",
        scale=97,
        filename="CACHE_x97.pb",
        url="https://example.com/cache.pb",
        algorithm="edsr",
        sha256=hashlib.sha256(good_content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ("edsr", 97), spec)
    _ = (tmp_path / "CACHE_x97.pb").write_bytes(b"corrupted")
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeStreamResponse(good_content))

    result = model_registry.ensure_model(tmp_path, "edsr", 97)

    assert result.read_bytes() == good_content


def test_ensure_model_reuses_valid_cache_without_network(
    tmp_path: "pathlib.Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached model with the right checksum is used without any download."""
    good_content = b"good bytes"
    spec = ModelSpec(
        name="edsr",
        scale=96,
        filename="VALID_x96.pb",
        url="https://example.com/valid.pb",
        algorithm="edsr",
        sha256=hashlib.sha256(good_content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ("edsr", 96), spec)
    target = tmp_path / "VALID_x96.pb"
    _ = target.write_bytes(good_content)

    def _fail(*_args: object, **_kwargs: object) -> None:
        """Fail the test: a valid cached model must not be downloaded again.

        Args:
            *_args: Ignored.
            **_kwargs: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("network should not be used for a valid cached file")

    monkeypatch.setattr(requests, "get", _fail)

    assert model_registry.ensure_model(tmp_path, "edsr", 96) == target


def test_interrupted_download_leaves_no_temporary_file(
    tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection dropped mid-download leaves nothing behind (it used to leave `.part`)."""

    class _DroppingResponse(_FakeStreamResponse):
        """A response whose connection drops after the first chunk."""

        def iter_content(self, chunk_size: int) -> "Iterator[bytes]":
            """Yield a partial body, then drop the connection.

            Args:
                chunk_size: Ignored.

            Yields:
                The partial body.

            Raises:
                requests.ConnectionError: After the first chunk.
            """
            del chunk_size
            yield b"partial"
            raise requests.ConnectionError("dropped")

    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _DroppingResponse(b""))

    with pytest.raises(requests.ConnectionError):
        _http.download_verified("https://x/m.pb", tmp_path / "m.pb", "0" * 64)
    assert list(tmp_path.iterdir()) == []


def _install_spec(monkeypatch: pytest.MonkeyPatch, scale: int, content: bytes) -> ModelSpec:
    """Add a fake `edsr` model to the manifest, whose expected checksum is `content`'s.

    Args:
        monkeypatch: Edits the manifest for this test only.
        scale: The fake model's scale (its manifest key).
        content: The model file's expected content.

    Returns:
        The added spec.
    """
    spec = ModelSpec(
        name="edsr",
        scale=scale,
        filename=f"FAKE_x{scale}.pb",
        url=f"https://example.com/fake_x{scale}.pb",
        algorithm="edsr",
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ("edsr", scale), spec)
    return spec


def test_zero_byte_cache_is_downloaded_again_without_hashing(
    tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty cached model (an aborted copy) is re-fetched straight away, never hashed."""
    content = b"good bytes"
    spec = _install_spec(monkeypatch, 95, content)
    (tmp_path / spec.filename).touch()

    def _no_cache_hashing(path: "pathlib.Path") -> str:
        """Fail the test: an empty cache file isn't worth hashing.

        Args:
            path: The file that would be hashed.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError(f"{path} should not be hashed")

    monkeypatch.setattr(model_registry, "sha256_of", _no_cache_hashing)
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeStreamResponse(content))

    assert model_registry.ensure_model(tmp_path, "edsr", 95).read_bytes() == content


def test_http_error_keeps_the_previous_file_and_no_temporary(
    tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed download (HTTP 500) leaves the file already in place untouched, and nothing else."""
    target = tmp_path / "m.pb"
    _ = target.write_bytes(b"previous")
    error = requests.HTTPError("500 Server Error")
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeStreamResponse(b"", error))

    with pytest.raises(requests.HTTPError, match="500"):
        _http.download_verified("https://x/m.pb", target, "0" * 64)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_checksum_mismatch_keeps_the_previous_file(
    tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download failing its integrity check never replaces the file already in place."""
    target = tmp_path / "m.pb"
    _ = target.write_bytes(b"previous")
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeStreamResponse(b"tampered"))

    with pytest.raises(ValueError, match="integrity check"):
        _http.download_verified("https://x/m.pb", target, "0" * 64)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]

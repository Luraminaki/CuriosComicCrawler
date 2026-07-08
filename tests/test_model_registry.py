"""Tests for `curios_comic_crawler.model_registry`."""

import hashlib
import pathlib
import re

import pytest

from curios_comic_crawler import model_registry
from curios_comic_crawler.models import ModelSpec


class _FakeStreamResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self) -> '_FakeStreamResponse':
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int) -> object:
        yield self._content


def test_manifest_keys_match_spec_fields() -> None:
    for (name, scale), spec in model_registry.MODEL_MANIFEST.items():
        assert spec.name == name
        assert spec.scale == scale


def test_manifest_filenames_are_unique() -> None:
    filenames = [spec.filename for spec in model_registry.MODEL_MANIFEST.values()]
    assert len(filenames) == len(set(filenames))


def test_manifest_sha256_are_valid_hex_digests() -> None:
    for spec in model_registry.MODEL_MANIFEST.values():
        assert re.fullmatch(r'[0-9a-f]{64}', spec.sha256)


def test_manifest_urls_are_https() -> None:
    for spec in model_registry.MODEL_MANIFEST.values():
        assert spec.url.startswith('https://')


def test_ensure_model_downloads_and_verifies_when_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b'fake model bytes'
    spec = ModelSpec(
        name='edsr', scale=99, filename='FAKE_x99.pb', url='https://example.com/fake.pb',
        algorithm='edsr', sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ('edsr', 99), spec)
    monkeypatch.setattr(model_registry.requests, 'get', lambda *_a, **_k: _FakeStreamResponse(content))

    result = model_registry.ensure_model(tmp_path, 'edsr', 99)

    assert result == tmp_path / 'FAKE_x99.pb'
    assert result.read_bytes() == content


def test_ensure_model_raises_and_cleans_up_on_hash_mismatch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ModelSpec(
        name='edsr', scale=98, filename='BAD_x98.pb', url='https://example.com/bad.pb',
        algorithm='edsr', sha256='0' * 64,
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ('edsr', 98), spec)
    monkeypatch.setattr(
        model_registry.requests, 'get', lambda *_a, **_k: _FakeStreamResponse(b'not matching content'),
    )

    with pytest.raises(ValueError, match='integrity check'):
        model_registry.ensure_model(tmp_path, 'edsr', 98)

    assert not (tmp_path / 'BAD_x98.pb').exists()
    assert not (tmp_path / 'BAD_x98.pb.part').exists()


def test_ensure_model_redownloads_corrupted_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_content = b'good bytes'
    spec = ModelSpec(
        name='edsr', scale=97, filename='CACHE_x97.pb', url='https://example.com/cache.pb',
        algorithm='edsr', sha256=hashlib.sha256(good_content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ('edsr', 97), spec)
    (tmp_path / 'CACHE_x97.pb').write_bytes(b'corrupted')
    monkeypatch.setattr(model_registry.requests, 'get', lambda *_a, **_k: _FakeStreamResponse(good_content))

    result = model_registry.ensure_model(tmp_path, 'edsr', 97)

    assert result.read_bytes() == good_content


def test_ensure_model_reuses_valid_cache_without_network(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_content = b'good bytes'
    spec = ModelSpec(
        name='edsr', scale=96, filename='VALID_x96.pb', url='https://example.com/valid.pb',
        algorithm='edsr', sha256=hashlib.sha256(good_content).hexdigest(),
    )
    monkeypatch.setitem(model_registry.MODEL_MANIFEST, ('edsr', 96), spec)
    target = tmp_path / 'VALID_x96.pb'
    target.write_bytes(good_content)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError('network should not be used for a valid cached file')

    monkeypatch.setattr(model_registry.requests, 'get', _fail)

    assert model_registry.ensure_model(tmp_path, 'edsr', 96) == target

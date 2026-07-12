"""Tests for `curios_comic_crawler.downloader`."""

import pathlib

import pytest
import requests

from curios_comic_crawler import downloader


class _FakeResponse:
    def __init__(self, ok: bool, status_code: int = 200, content: bytes = b'') -> None:
        self.ok = ok
        self.status_code = status_code
        self._content = content

    def __enter__(self) -> '_FakeResponse':
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False

    def iter_content(self, chunk_size: int) -> object:
        yield self._content


def test_get_last_saved_returns_zero_when_empty(config) -> None:
    assert downloader.get_last_saved(config) == 0


def test_get_last_saved_finds_highest_numeric_page(config) -> None:
    for name in ('SA_0001_small.jpg', 'SA_0002_small.jpg', 'SA_0010_small.jpg'):
        (config.dl_dir / name).touch()

    assert downloader.get_last_saved(config) == 10


def test_get_last_saved_skips_non_numeric_stem_segment(config) -> None:
    (config.dl_dir / 'SA_zzzz_small.jpg').touch()
    (config.dl_dir / 'SA_0005_small.jpg').touch()

    assert downloader.get_last_saved(config) == 5


def test_get_last_saved_handles_bd_name_with_underscore(make_config, tmp_path: pathlib.Path) -> None:
    cfg = make_config(BD_name='my_comic', folder_data=str(tmp_path / 'data') + '/')
    cfg.dl_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dl_dir / 'my_comic_0007_small.jpg').touch()

    assert downloader.get_last_saved(cfg) == 7


def test_get_last_saved_not_fooled_by_lexicographic_order(make_config, tmp_path: pathlib.Path) -> None:
    cfg = make_config(padded=3, folder_data=str(tmp_path / 'data') + '/')
    cfg.dl_dir.mkdir(parents=True, exist_ok=True)
    for name in ('SA_999_small.jpg', 'SA_1000_small.jpg'):
        (cfg.dl_dir / name).touch()

    assert downloader.get_last_saved(cfg) == 1000


def test_dl_and_save_img_saves_on_success(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader.requests, 'get', lambda *a, **k: _FakeResponse(ok=True, content=b'hello'))
    save_path = tmp_path / 'page.jpg'

    assert downloader.dl_and_save_img('http://example.com/page.jpg', save_path, {}) is True
    assert save_path.read_bytes() == b'hello'
    assert not save_path.with_suffix('.jpg.part').exists()


def test_dl_and_save_img_returns_false_on_missing_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downloader.requests, 'get', lambda *a, **k: _FakeResponse(ok=False, status_code=404))
    save_path = tmp_path / 'page.jpg'

    assert downloader.dl_and_save_img('http://example.com/page.jpg', save_path, {}) is False
    assert not save_path.exists()


def test_dl_and_save_img_returns_false_on_network_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def _raise(*_args: object, **_kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        raise requests.ConnectionError('boom')

    monkeypatch.setattr(downloader.requests, 'get', _raise)
    monkeypatch.setattr(downloader.time, 'sleep', lambda *_a: None)
    save_path = tmp_path / 'page.jpg'

    assert downloader.dl_and_save_img('http://example.com/page.jpg', save_path, {}) is False
    assert call_count == downloader._MAX_NETWORK_ATTEMPTS


def test_dl_and_save_img_recovers_after_transient_network_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def _flaky_get(*_args: object, **_kwargs: object) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise requests.ConnectionError('boom')
        return _FakeResponse(ok=True, content=b'hello')

    monkeypatch.setattr(downloader.requests, 'get', _flaky_get)
    monkeypatch.setattr(downloader.time, 'sleep', lambda *_a: None)
    save_path = tmp_path / 'page.jpg'

    assert downloader.dl_and_save_img('http://example.com/page.jpg', save_path, {}) is True
    assert save_path.read_bytes() == b'hello'


def test_dl_and_save_img_cleans_up_partial_file_on_stream_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenStreamResponse(_FakeResponse):
        def iter_content(self, chunk_size: int) -> object:
            yield b'partial-data'
            raise requests.ConnectionError('dropped mid-stream')

    monkeypatch.setattr(downloader.requests, 'get', lambda *a, **k: _BrokenStreamResponse(ok=True))
    monkeypatch.setattr(downloader.time, 'sleep', lambda *_a: None)
    save_path = tmp_path / 'page.jpg'

    assert downloader.dl_and_save_img('http://example.com/page.jpg', save_path, {}) is False
    assert not save_path.exists()
    assert not save_path.with_suffix('.jpg.part').exists()


def test_run_stops_one_page_past_the_last_real_page(config, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully-missing page costs `fails` (2) attempts, so the run stops right after it --
    never touching the page after -- matching the pre-refactor script's behavior.
    """
    monkeypatch.setattr(downloader, 'get_last_saved', lambda *_a, **_k: 1283)
    monkeypatch.setattr(downloader.time, 'sleep', lambda *_a: None)

    attempted_pages = []

    def fake_dl(link: str, save_path: pathlib.Path, headers: dict) -> bool:
        page = int(save_path.name.split('_')[1])
        attempted_pages.append(page)
        return page <= 1285

    monkeypatch.setattr(downloader, 'dl_and_save_img', fake_dl)

    downloader.run(config)

    assert max(attempted_pages) == 1286


def test_run_does_not_terminate_permanently_after_one_early_transient_failure(
    config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail counter must reset on success, not accumulate over the whole run."""
    monkeypatch.setattr(downloader, 'get_last_saved', lambda *_a, **_k: 0)
    monkeypatch.setattr(downloader.time, 'sleep', lambda *_a: None)

    call_count = 0

    def fake_dl(link: str, save_path: pathlib.Path, headers: dict) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False  # one transient failure, right at the start
        page = int(save_path.name.split('_')[1])
        return page <= 5

    monkeypatch.setattr(downloader, 'dl_and_save_img', fake_dl)

    downloader.run(config)  # fixture's `fails` is 2

    # Reaching well past page 1 proves the single early failure didn't end the run for good.
    assert call_count > 5


def test_run_force_starts_from_page_one(config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, 'get_last_saved', lambda *_a, **_k: 9999)
    monkeypatch.setattr(downloader.time, 'sleep', lambda *_a: None)

    attempted_pages = []

    def fake_dl(link: str, save_path: pathlib.Path, headers: dict) -> bool:
        page = int(save_path.name.split('_')[1])
        attempted_pages.append(page)
        return page <= 1

    monkeypatch.setattr(downloader, 'dl_and_save_img', fake_dl)

    downloader.run(config, force=True)

    assert min(attempted_pages) == 1

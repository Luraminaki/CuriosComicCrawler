"""Tests for `curios_comic_crawler.downloader`."""

import time
from typing import TYPE_CHECKING, final

import pytest
import requests

from curios_comic_crawler import downloader

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator

    from conftest import ConfigFactory

    from curios_comic_crawler.config import AppConfig


@final
class _FakeResponse:
    """A canned HTTP response, usable as a context manager like requests' own."""

    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
        broken_stream: bool = False,
    ) -> None:
        """Describe the answer.

        Args:
            status_code: HTTP status.
            content: Body.
            headers: Response headers; defaults to an image content type.
            broken_stream: Drop the connection after the first chunk.
        """
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers = {"Content-Type": "image/jpeg"} if headers is None else headers
        self._content = content
        self._broken_stream = broken_stream

    def __enter__(self) -> "_FakeResponse":
        """Enter the `with` block.

        Returns:
            The response.
        """
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        """Leave the `with` block, not swallowing errors.

        Returns:
            False.
        """
        return False

    def iter_content(self, chunk_size: int) -> "Iterator[bytes]":
        """Yield the body, possibly dropping the connection midway.

        Args:
            chunk_size: Ignored.

        Yields:
            The body.

        Raises:
            requests.ConnectionError: When the stream is broken.
        """
        del chunk_size
        yield self._content
        if self._broken_stream:
            raise requests.ConnectionError("dropped mid-stream")


@final
class _FakeSession:
    """Serves scripted answers (responses, or exceptions to raise) in order."""

    def __init__(self, *answers: "_FakeResponse | Exception") -> None:
        """Script the answers.

        Args:
            *answers: One per request.
        """
        self.answers = list(answers)
        self.requests = 0

    def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        """Answer the next request.

        Returns:
            The scripted response.

        Raises:
            Exception: The scripted exception.
        """
        self.requests += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture(name="no_sleep")
def fixture_no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the downloader's pauses instead of sleeping.

    Args:
        monkeypatch: Replaces `time.sleep`.

    Returns:
        The pauses requested, in seconds.
    """
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    return pauses


def fetch(session: _FakeSession, save_path: "pathlib.Path") -> downloader.Fetch:
    """Fetch one image through a fake session.

    Args:
        session: The scripted session.
        save_path: Destination.

    Returns:
        The outcome.
    """
    return downloader.dl_and_save_img(session, "http://x/p.jpg", save_path)  # pyright: ignore[reportArgumentType]


def test_get_last_saved_returns_zero_when_empty(config: "AppConfig") -> None:
    """With no page downloaded yet, the last saved page is 0."""
    assert downloader.get_last_saved(config) == 0


def test_get_last_saved_finds_highest_numeric_page(config: "AppConfig") -> None:
    """The last saved page is the highest page number on disk."""
    for name in ("SA_0001_small.jpg", "SA_0002_small.jpg", "SA_0010_small.jpg"):
        (config.dl_dir / name).touch()

    assert downloader.get_last_saved(config) == 10


def test_get_last_saved_skips_non_numeric_stem_segment(config: "AppConfig") -> None:
    """Files whose page segment isn't a number are ignored."""
    (config.dl_dir / "SA_zzzz_small.jpg").touch()
    (config.dl_dir / "SA_0005_small.jpg").touch()

    assert downloader.get_last_saved(config) == 5


def test_get_last_saved_handles_bd_name_with_underscore(
    make_config: "ConfigFactory", tmp_path: "pathlib.Path"
) -> None:
    """A comic name containing `_` doesn't shift the page segment."""
    cfg = make_config(BD_name="my_comic", folder_data=str(tmp_path / "data") + "/")
    cfg.dl_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dl_dir / "my_comic_0007_small.jpg").touch()

    assert downloader.get_last_saved(cfg) == 7


def test_get_last_saved_not_fooled_by_lexicographic_order(
    make_config: "ConfigFactory", tmp_path: "pathlib.Path"
) -> None:
    """Page 1000 beats page 999: numbers are compared, not names."""
    cfg = make_config(padded=3, folder_data=str(tmp_path / "data") + "/")
    cfg.dl_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SA_999_small.jpg", "SA_1000_small.jpg"):
        (cfg.dl_dir / name).touch()

    assert downloader.get_last_saved(cfg) == 1000


def test_image_is_saved(tmp_path: "pathlib.Path", no_sleep: list[float]) -> None:
    """A 200 image answer is written to the target file."""
    save_path = tmp_path / "page.jpg"

    assert fetch(_FakeSession(_FakeResponse(content=b"hello")), save_path) is downloader.Fetch.SAVED
    assert save_path.read_bytes() == b"hello"
    assert no_sleep == []


@pytest.mark.parametrize("status", [404, 410])
@pytest.mark.usefixtures("no_sleep")
def test_only_404_and_410_mean_missing(tmp_path: "pathlib.Path", status: int) -> None:
    """The site's "not found" answers mean the page doesn't exist, without retrying."""
    session = _FakeSession(_FakeResponse(status_code=status))

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.MISSING
    assert session.requests == 1


def test_busy_server_is_retried_honouring_retry_after(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """A 503 used to end the crawl as "missing"; it is retried after the asked delay."""
    session = _FakeSession(
        _FakeResponse(status_code=503, headers={"Retry-After": "7"}),
        _FakeResponse(content=b"hello"),
    )

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.SAVED
    assert no_sleep == [7.0]


def test_persistent_errors_fail_after_backing_off(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """Network errors are retried with a growing delay, then reported as a failure."""
    session = _FakeSession(*[requests.ConnectionError("boom")] * 3)

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert no_sleep == [2.0, 4.0]


@pytest.mark.usefixtures("no_sleep")
def test_dropped_download_is_retried_and_leaves_no_partial_file(tmp_path: "pathlib.Path") -> None:
    """A connection dropped mid-download is retried; no `.part` file is left behind."""
    save_path = tmp_path / "page.jpg"
    session = _FakeSession(
        _FakeResponse(content=b"partial", broken_stream=True), _FakeResponse(content=b"full")
    )

    assert fetch(session, save_path) is downloader.Fetch.SAVED
    assert save_path.read_bytes() == b"full"
    assert list(tmp_path.iterdir()) == [save_path]


@pytest.mark.usefixtures("no_sleep")
def test_html_instead_of_an_image_is_a_failure(tmp_path: "pathlib.Path") -> None:
    """An HTML page (error or challenge) is not saved as a .jpg the upscaler would choke on."""
    session = _FakeSession(_FakeResponse(content=b"<html>", headers={"Content-Type": "text/html"}))

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("no_sleep")
def test_run_stops_with_an_error_when_the_site_fails(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure ends the run with exit code 1, not 0 like a normal end of comic."""
    monkeypatch.setattr(downloader, "dl_and_save_img", lambda *_a, **_k: downloader.Fetch.FAILED)

    assert downloader.run(config) == 1


def test_run_stops_one_page_past_the_last_real_page(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """The run stops right after a fully-missing page, never trying the next one.

    A missing page costs `fails` (2) attempts, matching the pre-refactor script's behavior.
    """
    monkeypatch.setattr(downloader, "get_last_saved", lambda *_a, **_k: 1283)
    monkeypatch.setattr(time, "sleep", lambda *_a: None)

    attempted_pages = []

    def fake_dl(_session: object, _link: str, save_path: "pathlib.Path") -> downloader.Fetch:
        """Record the page and answer as a site ending at the test's last page.

        Returns:
            The scripted outcome.
        """
        page = int(save_path.name.split("_")[1])
        attempted_pages.append(page)
        return downloader.Fetch.SAVED if page <= 1285 else downloader.Fetch.MISSING

    monkeypatch.setattr(downloader, "dl_and_save_img", fake_dl)

    _ = downloader.run(config)

    assert max(attempted_pages) == 1286


def test_run_does_not_terminate_permanently_after_one_early_transient_failure(
    config: "AppConfig",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """The fail counter must reset on success, not accumulate over the whole run."""
    monkeypatch.setattr(downloader, "get_last_saved", lambda *_a, **_k: 0)
    monkeypatch.setattr(time, "sleep", lambda *_a: None)

    call_count = 0

    def fake_dl(_session: object, _link: str, save_path: "pathlib.Path") -> downloader.Fetch:
        """Answer one missing variant first, then as a site ending at page 5.

        Returns:
            The scripted outcome.
        """
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return downloader.Fetch.MISSING  # one missing variant, right at the start
        page = int(save_path.name.split("_")[1])
        return downloader.Fetch.SAVED if page <= 5 else downloader.Fetch.MISSING

    monkeypatch.setattr(downloader, "dl_and_save_img", fake_dl)

    _ = downloader.run(config)  # fixture's `fails` is 2

    # Reaching well past page 1 proves the single early failure didn't end the run for good.
    assert call_count > 5


def test_run_force_starts_from_page_one(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """`--force` downloads from page 1 whatever is already saved."""
    monkeypatch.setattr(downloader, "get_last_saved", lambda *_a, **_k: 9999)
    monkeypatch.setattr(time, "sleep", lambda *_a: None)

    attempted_pages = []

    def fake_dl(_session: object, _link: str, save_path: "pathlib.Path") -> downloader.Fetch:
        """Record the page and answer as a site ending at the test's last page.

        Returns:
            The scripted outcome.
        """
        page = int(save_path.name.split("_")[1])
        attempted_pages.append(page)
        return downloader.Fetch.SAVED if page <= 1 else downloader.Fetch.MISSING

    monkeypatch.setattr(downloader, "dl_and_save_img", fake_dl)

    _ = downloader.run(config, force=True)

    assert min(attempted_pages) == 1

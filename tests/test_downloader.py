"""Tests for `curios_comic_crawler.downloader`'s single-image fetch: retries, failures, saving.

The page loop (`run`, `_fetch_page`) and `get_last_saved` are in `test_downloader_run.py`.
"""

from typing import TYPE_CHECKING, final

import pytest
import requests

from curios_comic_crawler import downloader

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator


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
        self.status_code: int = status_code
        self.ok: bool = status_code < 400
        self.headers: dict[str, str] = (
            {"Content-Type": "image/jpeg"} if headers is None else headers
        )
        self._content: bytes = content
        self._broken_stream: bool = broken_stream

    def __enter__(self) -> "_FakeResponse":
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
        self.answers: list[_FakeResponse | Exception] = list(answers)
        self.requests: int = 0

    def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        """Answer the next request.

        Args:
            *_args: The request's positional arguments (ignored).
            **_kwargs: The request's keyword arguments (ignored).

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


def fetch(session: _FakeSession, save_path: "pathlib.Path") -> downloader.Fetch:
    """Fetch one image through a fake session.

    Args:
        session: The scripted session.
        save_path: Destination.

    Returns:
        The outcome.
    """
    return downloader.dl_and_save_img(session, "http://x/p.jpg", save_path)  # pyright: ignore[reportArgumentType]


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


def test_retry_after_is_capped_at_sixty_seconds(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """A server asking to wait an hour is waited on for 60 s at most (it is effectively down)."""
    session = _FakeSession(
        _FakeResponse(status_code=503, headers={"Retry-After": "3600"}),
        _FakeResponse(content=b"hello"),
    )

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.SAVED
    assert no_sleep == [60.0]


def test_http_date_retry_after_falls_back_to_the_backoff(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """The rare date form of `Retry-After` isn't parsed: the first backoff step (2 s) is used."""
    session = _FakeSession(
        _FakeResponse(status_code=503, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        _FakeResponse(content=b"hello"),
    )

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.SAVED
    assert no_sleep == [2.0]


def test_three_busy_answers_fail_after_backing_off(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """Three 503s in a row give up: 3 requests, two growing pauses, no file."""
    session = _FakeSession(*[_FakeResponse(status_code=503) for _ in range(3)])

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert session.requests == 3
    assert no_sleep == [2.0, 4.0]
    assert list(tmp_path.iterdir()) == []


def test_persistent_errors_fail_after_backing_off(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """Network errors are retried with a growing delay, then reported as a failure."""
    session = _FakeSession(*[requests.ConnectionError("boom")] * 3)

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert no_sleep == [2.0, 4.0]


def test_forbidden_fails_at_once_without_retrying(
    tmp_path: "pathlib.Path", no_sleep: list[float]
) -> None:
    """A 403 won't change on retry: one request, no pause, no file."""
    session = _FakeSession(_FakeResponse(status_code=403, content=b"denied"))

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert session.requests == 1
    assert no_sleep == []
    assert list(tmp_path.iterdir()) == []


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
def test_stream_dropped_on_every_attempt_fails_and_leaves_nothing(
    tmp_path: "pathlib.Path",
) -> None:
    """A connection that always drops mid-download ends as a failure with an empty folder."""
    session = _FakeSession(
        *[_FakeResponse(content=b"partial", broken_stream=True) for _ in range(3)]
    )

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert session.requests == 3
    assert list(tmp_path.iterdir()) == []


def test_disk_error_while_saving_propagates_without_retry(
    tmp_path: "pathlib.Path", monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """A local write failure isn't the site's fault: it isn't retried, and no `.part` is left."""

    def write_then_fail(_response: object, path: "pathlib.Path", _chunk_size: int) -> None:
        """Write part of the file, then fail like a full disk.

        Args:
            _response: The response being saved (ignored).
            path: The temporary file being written.
            _chunk_size: Ignored.

        Raises:
            OSError: Always.
        """
        _ = path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(downloader, "stream_to_file", write_then_fail)
    session = _FakeSession(_FakeResponse(content=b"hello"))

    with pytest.raises(OSError, match="disk full"):
        _ = fetch(session, tmp_path / "page.jpg")
    assert session.requests == 1
    assert no_sleep == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("no_sleep")
def test_html_instead_of_an_image_is_a_failure(tmp_path: "pathlib.Path") -> None:
    """An HTML page (error or challenge) is not saved as a .jpg the upscaler would choke on."""
    session = _FakeSession(_FakeResponse(content=b"<html>", headers={"Content-Type": "text/html"}))

    assert fetch(session, tmp_path / "page.jpg") is downloader.Fetch.FAILED
    assert list(tmp_path.iterdir()) == []

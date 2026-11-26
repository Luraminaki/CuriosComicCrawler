"""Tests for `curios_comic_crawler.downloader`'s page loop: resuming, variants, stopping.

The single-image fetch (`dl_and_save_img`: retries, failures, saving) is in
`test_downloader.py`.
"""

# pylint: disable=protected-access  # unit tests exercise the module's private helpers

from typing import TYPE_CHECKING, final

import pytest

from curios_comic_crawler import downloader

if TYPE_CHECKING:
    import pathlib

    from conftest import ConfigFactory

    from curios_comic_crawler.config import AppConfig


@final
class _Site:
    """Stands in for `dl_and_save_img`: a site whose pages end at `last_page`."""

    def __init__(self, last_page: int, missing: tuple[str, ...] = ()) -> None:
        """Describe the site.

        Args:
            last_page: The last page that exists.
            missing: Filenames answered as missing even though their page exists.
        """
        self.last_page: int = last_page
        self.missing: tuple[str, ...] = missing
        self.links: list[str] = []

    def __call__(self, _session: object, link: str, save_path: "pathlib.Path") -> downloader.Fetch:
        """Record the request and answer it.

        Args:
            _session: The run's HTTP session (ignored).
            link: The image URL.
            save_path: Where the image would be saved.

        Returns:
            SAVED for an existing page, MISSING otherwise.
        """
        self.links.append(link)
        page = int(save_path.name.split("_")[1])
        if page > self.last_page or save_path.name in self.missing:
            return downloader.Fetch.MISSING
        return downloader.Fetch.SAVED

    @property
    def names(self) -> list[str]:
        """The requested filenames, in order."""
        return [link.rsplit("/", 1)[-1] for link in self.links]


def _install_site(monkeypatch: pytest.MonkeyPatch, site: _Site, last_saved: int = 0) -> _Site:
    """Serve downloads from `site`, as if `last_saved` pages were already on disk.

    Args:
        monkeypatch: Replaces the downloader's fetch and resume point.
        site: The fake site.
        last_saved: What `get_last_saved` reports.

    Returns:
        `site`, for chaining.
    """
    monkeypatch.setattr(downloader, "dl_and_save_img", site)
    monkeypatch.setattr(downloader, "get_last_saved", lambda _config: last_saved)
    return site


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


def test_get_last_saved_ignores_other_comics_and_partial_files(config: "AppConfig") -> None:
    """Another comic sharing the prefix (`SA2_`) and unfinished `.part` downloads don't count."""
    (config.dl_dir / "SA_0003_small.jpg").touch()
    (config.dl_dir / "SA2_0099_small.jpg").touch()
    (config.dl_dir / "SA_0050_small.jpg.part").touch()

    assert downloader.get_last_saved(config) == 3


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


def test_variants_are_requested_in_order_from_the_root_site(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """Each variant's URL is `root_site` + its filename, tried in the configured order."""
    site = _install_site(monkeypatch, _Site(last_page=0))

    assert downloader.run(config) == 0
    assert site.links == [
        "https://www.collectedcurios.com/SA_0001_small.jpg",
        "https://www.collectedcurios.com/SA_0001_small_b.jpg",
    ]
    assert no_sleep == [1]


def test_throttle_comes_between_variants_not_before_the_first(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch
) -> None:
    """The politeness pause separates two requests; the first one goes out at once."""
    events: list[str] = []

    def record_fetch(_session: object, link: str, _save_path: "pathlib.Path") -> downloader.Fetch:
        """Record a request; every variant is missing.

        Args:
            _session: The run's HTTP session (ignored).
            link: The image URL.
            _save_path: Ignored.

        Returns:
            MISSING.
        """
        events.append(link.rsplit("/", 1)[-1])
        return downloader.Fetch.MISSING

    monkeypatch.setattr(downloader, "dl_and_save_img", record_fetch)
    monkeypatch.setattr("time.sleep", lambda seconds: events.append(f"sleep {seconds}"))

    outcomes = downloader._fetch_page(object(), config, 1)  # pyright: ignore[reportArgumentType]

    assert outcomes == [downloader.Fetch.MISSING, downloader.Fetch.MISSING]
    assert events == ["SA_0001_small.jpg", "sleep 1", "SA_0001_small_b.jpg"]


@pytest.mark.usefixtures("no_sleep")
def test_run_stops_with_an_error_when_the_site_fails(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure ends the run with exit code 1, not 0 like a normal end of comic."""
    monkeypatch.setattr(downloader, "dl_and_save_img", lambda *_a, **_k: downloader.Fetch.FAILED)

    assert downloader.run(config) == 1


@pytest.mark.usefixtures("no_sleep")
def test_run_stops_one_page_past_the_last_real_page(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run resumes after the last saved page and stops at the first fully-missing one.

    A missing page costs one fail per variant (2), which is the fixture's `fails`.
    """
    site = _install_site(monkeypatch, _Site(last_page=1285), last_saved=1283)

    assert downloader.run(config) == 0
    assert site.names == [
        "SA_1284_small.jpg",
        "SA_1285_small.jpg",
        "SA_1286_small.jpg",
        "SA_1286_small_b.jpg",
    ]


@pytest.mark.usefixtures("no_sleep")
def test_run_resets_the_fail_count_after_a_saved_page(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing first variant followed by a saved one doesn't count towards stopping later."""
    site = _install_site(monkeypatch, _Site(last_page=5, missing=("SA_0001_small.jpg",)))

    assert downloader.run(config) == 0
    assert site.names == [
        "SA_0001_small.jpg",
        "SA_0001_small_b.jpg",
        "SA_0002_small.jpg",
        "SA_0003_small.jpg",
        "SA_0004_small.jpg",
        "SA_0005_small.jpg",
        "SA_0006_small.jpg",
        "SA_0006_small_b.jpg",
    ]


@pytest.mark.usefixtures("no_sleep")
def test_run_force_starts_from_page_one(
    config: "AppConfig", monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--force` downloads from page 1 whatever is already saved."""
    site = _install_site(monkeypatch, _Site(last_page=1), last_saved=9999)

    assert downloader.run(config, force=True) == 0
    assert site.names == ["SA_0001_small.jpg", "SA_0002_small.jpg", "SA_0002_small_b.jpg"]

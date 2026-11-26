#!/usr/bin/env python3
"""Download comic pages listed on the configured site into `config.dl_dir`."""

import enum
import logging
import re
import time
from typing import TYPE_CHECKING

import requests

from curios_comic_crawler._http import CHUNK_SIZE_BYTES, stream_to_file

if TYPE_CHECKING:
    import pathlib
    from re import Pattern

    from curios_comic_crawler.config import AppConfig

logger = logging.getLogger(__name__)

_MAX_NETWORK_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2
_MAX_RETRY_DELAY_SECONDS = 60.0  # a server asking for longer is effectively down
_THROTTLE_SECONDS = 1
_TIMEOUT_SECONDS = 10
_MISSING_STATUSES = frozenset({404, 410})
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _page_prefix(config: "AppConfig", page: int) -> str:
    """Build the `bd_name` + zero-padded page number segment shared by every filename variant.

    Args:
        config (AppConfig): Application configuration.
        page (int): Page number.

    Returns:
        str: E.g. `SA_0042`.
    """
    return f"{config.bd_name}_{str(page).zfill(config.padded)}"


def _page_image_names(config: "AppConfig", page: int) -> list[str]:
    """Build every filename variant tried for `page`, in order.

    Args:
        config (AppConfig): Application configuration.
        page (int): Page number.

    Returns:
        list[str]: One filename per entry in `config.filename_variants`.
    """
    prefix = _page_prefix(config, page)
    return [f"{prefix}_{variant}{config.ext}" for variant in config.filename_variants]


def _page_stem_pattern(config: "AppConfig") -> "Pattern[str]":
    """Build the regex matching a saved page's filename stem, capturing its page number.

    Mirrors `_page_prefix`'s `bd_name_<digits>` segment followed by one of
    `config.filename_variants`, anchored on the exact `bd_name` so it can't misparse a page
    number out of a `bd_name` that itself contains an underscore.

    Args:
        config (AppConfig): Application configuration.

    Returns:
        Pattern[str]: The compiled pattern; group 1 is the page number.
    """
    variants = "|".join(re.escape(variant) for variant in config.filename_variants)
    return re.compile(rf"^{re.escape(config.bd_name)}_(\d+)_(?:{variants})$")


def get_last_saved(config: "AppConfig") -> int:
    """Retrieve the page number of the last saved file.

    Args:
        config (AppConfig): Application configuration.

    Returns:
        int: Last saved page number, or 0 if nothing has been saved yet.
    """
    pattern = _page_stem_pattern(config)
    highest = 0
    found = False

    for candidate in config.dl_dir.glob(f"*{config.ext}"):
        match = pattern.match(candidate.stem)
        if match:
            found = True
            highest = max(highest, int(match.group(1)))

    if not found:
        logger.info("No prior save")

    return highest


class Fetch(enum.Enum):
    """Outcome of fetching one page image."""

    SAVED = "saved"
    MISSING = "missing"  # the site says the page doesn't exist (404/410)
    FAILED = "failed"  # anything else: network trouble, server error, not an image


def _retry_delay(response: "requests.Response | None", attempt: int) -> float:
    """How long to wait before the next attempt: the server's `Retry-After`, else a backoff.

    Args:
        response (requests.Response | None): The failed response, if one came back.
        attempt (int): Attempts made so far (1-based).

    Returns:
        float: Seconds to wait (2, 4, 8... unless the server asks for a specific delay).
    """
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    if retry_after.isdecimal():  # the date form is rare; fall back to the backoff for it
        return min(float(retry_after), _MAX_RETRY_DELAY_SECONDS)
    return float(_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))


def _save_image(response: "requests.Response", save_path: "pathlib.Path") -> None:
    """Stream an image response to `save_path` through a temporary file.

    Args:
        response (requests.Response): A successful response carrying an image.
        save_path (pathlib.Path): Destination path; only ever complete.

    Raises:
        requests.RequestException: If the connection drops mid-download.
        OSError: If the file can't be written.
    """
    partial_path = save_path.with_name(f"{save_path.name}.part")
    try:
        stream_to_file(response, partial_path, CHUNK_SIZE_BYTES)
        _ = partial_path.replace(save_path)
    finally:
        partial_path.unlink(missing_ok=True)  # left only if the download failed


def _attempt(
    session: "requests.Session", link: str, save_path: "pathlib.Path"
) -> "tuple[Fetch | None, requests.Response | None]":
    """Make one attempt at fetching a page image.

    Args:
        session (requests.Session): The run's HTTP session.
        link (str): Image URL.
        save_path (pathlib.Path): Destination path.

    Returns:
        tuple: The outcome (`None` if the attempt may be retried), and the response if any.
    """
    try:
        with session.get(link, stream=True, timeout=_TIMEOUT_SECONDS) as response:
            if response.status_code in _MISSING_STATUSES:
                logger.info("Image not found -- HTTP_CODE: %s", response.status_code)
                return Fetch.MISSING, response
            if response.status_code in _RETRYABLE_STATUSES:
                logger.warning("Server busy or failing -- HTTP_CODE: %s", response.status_code)
                return None, response
            if not response.ok:
                logger.error(
                    "Unexpected answer for %s -- HTTP_CODE: %s", link, response.status_code
                )
                return Fetch.FAILED, response
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                logger.error("%s is not an image (Content-Type %r)", link, content_type)
                return Fetch.FAILED, response
            logger.info("Saving: %s", save_path.name)
            _save_image(response, save_path)
            return Fetch.SAVED, response
    except requests.RequestException as error:  # network trouble, possibly mid-download
        logger.warning("Download of %s failed: %r", link, error)
        return None, None


def dl_and_save_img(session: "requests.Session", link: str, save_path: "pathlib.Path") -> Fetch:
    """Fetch a page image from `link` and save it to `save_path`.

    Network errors (including a connection dropping mid-download), 429 and 5xx answers are
    retried with a growing delay; only 404/410 mean the page doesn't exist.

    Args:
        session (requests.Session): The run's HTTP session (shared connection and headers).
        link (str): Image URL.
        save_path (pathlib.Path): Destination path.

    Returns:
        Fetch: What happened.
    """
    for attempt in range(1, _MAX_NETWORK_ATTEMPTS + 1):
        outcome, response = _attempt(session, link, save_path)
        if outcome is not None:
            return outcome
        if attempt < _MAX_NETWORK_ATTEMPTS:
            time.sleep(_retry_delay(response, attempt))

    logger.error("Giving up on %s after %s attempts", link, _MAX_NETWORK_ATTEMPTS)
    return Fetch.FAILED


def _fetch_page(session: "requests.Session", config: "AppConfig", page: int) -> list[Fetch]:
    """Try each filename variant of a page until one is saved or one fails.

    Args:
        session (requests.Session): The run's HTTP session.
        config (AppConfig): Application configuration.
        page (int): Page number.

    Returns:
        list[Fetch]: The outcome of each variant tried, in order.
    """
    outcomes: list[Fetch] = []
    for index, image_name in enumerate(_page_image_names(config, page)):
        if index:
            time.sleep(_THROTTLE_SECONDS)  # politeness between requests to the site
        logger.info("Downloading: %s", image_name)
        outcome = dl_and_save_img(
            session, config.root_site + image_name, config.dl_dir / image_name
        )
        outcomes.append(outcome)
        if outcome is not Fetch.MISSING:
            break
    return outcomes


def run(config: "AppConfig", *, force: bool = False) -> int:
    """Download comic pages that haven't been downloaded yet.

    Pages are fetched in order until `config.fails` consecutive filename variants are missing
    (404). A fully-missing page costs one fail per filename variant; any saved page resets the
    count.

    Args:
        config (AppConfig): Application configuration.
        force (bool, optional): If True, ignore what has already been downloaded and start
            over from the first page (existing files are overwritten). Defaults to False.

    Returns:
        int: 0 when the run ends on missing pages (the end of the comic), 1 when it stopped on
        a failure (network, server error, unexpected answer).
    """
    config.dl_dir.mkdir(exist_ok=True, parents=True)
    if force:
        logger.info("Force re-download requested, starting over from page 1")
        next_page = 1
    else:
        last_page_available = get_last_saved(config)
        logger.info("Last downloaded: %s", last_page_available)
        next_page = last_page_available + 1

    consecutive_fails = 0
    with requests.Session() as session:
        session.headers.update(config.headers)
        while True:
            outcomes = _fetch_page(session, config, next_page)
            if Fetch.FAILED in outcomes:
                logger.error(
                    "Stopping at page %s: the site couldn't be read (see above)", next_page
                )
                return 1
            consecutive_fails = 0 if Fetch.SAVED in outcomes else consecutive_fails + len(outcomes)
            if consecutive_fails >= config.fails:
                break
            time.sleep(_THROTTLE_SECONDS)
            next_page += 1

    logger.info("Nothing more to download. Process ending.")
    return 0

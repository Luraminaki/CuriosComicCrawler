#!/usr/bin/env python3
"""Download comic pages listed on the configured site into `config.dl_dir`."""

import logging
import pathlib
import re
import time
from re import Pattern

import requests

from ._http import CHUNK_SIZE_BYTES, stream_to_file
from .config import AppConfig

logger = logging.getLogger(__name__)

_MAX_NETWORK_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2
_THROTTLE_SECONDS = 1


def _page_prefix(config: AppConfig, page: int) -> str:
    """The `bd_name` + zero-padded page number segment shared by every filename variant."""
    return f'{config.bd_name}_{str(page).zfill(config.padded)}'


def _page_image_names(config: AppConfig, page: int) -> list[str]:
    """Build every filename variant tried for `page`, in order.

    Args:
        config (AppConfig): Application configuration.
        page (int): Page number.

    Returns:
        list[str]: One filename per entry in `config.filename_variants`.
    """
    prefix = _page_prefix(config, page)
    return [f'{prefix}_{variant}{config.ext}' for variant in config.filename_variants]


def _page_stem_pattern(config: AppConfig) -> Pattern[str]:
    """Regex matching a saved page's filename stem, capturing its page number.

    Mirrors `_page_prefix`'s `bd_name_<digits>` segment followed by one of
    `config.filename_variants`, anchored on the exact `bd_name` so it can't misparse a page
    number out of a `bd_name` that itself contains an underscore.
    """
    variants = '|'.join(re.escape(variant) for variant in config.filename_variants)
    return re.compile(rf'^{re.escape(config.bd_name)}_(\d+)_(?:{variants})$')


def get_last_saved(config: AppConfig) -> int:
    """Retrieve the page number of the last saved file.

    Args:
        config (AppConfig): Application configuration.

    Returns:
        int: Last saved page number, or 0 if nothing has been saved yet.
    """
    pattern = _page_stem_pattern(config)
    highest = 0
    found = False

    for candidate in config.dl_dir.glob(f'*{config.ext}'):
        match = pattern.match(candidate.stem)
        if match:
            found = True
            highest = max(highest, int(match.group(1)))

    if not found:
        logger.info('No prior save')

    return highest


def dl_and_save_img(link: str, save_path: pathlib.Path, headers: dict[str, str]) -> bool:
    """Fetch an image from `link` and save it to `save_path`.

    Transient network errors (timeouts, connection resets) are retried a few times before
    giving up, since a couple of blips shouldn't be indistinguishable from the page genuinely
    not existing.

    Args:
        link (str): Image source link.
        save_path (pathlib.Path): Destination path.
        headers (dict[str, str]): Headers for the request.

    Returns:
        bool: True if the download succeeded, False otherwise.
    """
    for attempt in range(1, _MAX_NETWORK_ATTEMPTS + 1):
        try:
            resource = requests.get(link, headers=headers, stream=True, timeout=5, verify=True)
        except requests.RequestException as error:
            logger.warning(
                'Distant resource %s unreachable (attempt %s/%s): %r',
                link, attempt, _MAX_NETWORK_ATTEMPTS, error,
            )
            if attempt < _MAX_NETWORK_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        with resource:
            if not resource.ok:
                logger.info('Image not found -- HTTP_CODE: %s', resource.status_code)
                return False

            logger.info('Saving: %s', save_path.name)
            partial_path = save_path.with_suffix(f'{save_path.suffix}.part')
            try:
                stream_to_file(resource, partial_path, CHUNK_SIZE_BYTES)
            except requests.RequestException as error:
                logger.warning('Download of %s interrupted: %r', link, error)
                partial_path.unlink(missing_ok=True)
                return False

        partial_path.replace(save_path)
        return True

    logger.warning('Giving up on %s after %s network error(s)', link, _MAX_NETWORK_ATTEMPTS)
    return False


def run(config: AppConfig, *, force: bool = False) -> int:
    """Download comic pages that haven't been downloaded yet.

    Args:
        config (AppConfig): Application configuration.
        force (bool, optional): If True, ignore what has already been downloaded and start
            over from the first page (existing files are overwritten). Defaults to False.

    Returns:
        int: 0 on success (this function only returns once downloading stops).
    """
    config.dl_dir.mkdir(exist_ok=True, parents=True)

    if force:
        logger.info('Force re-download requested, starting over from page 1')
        next_page = 1
    else:
        last_page_available = get_last_saved(config)
        logger.info('Last downloaded: %s', last_page_available)
        next_page = last_page_available + 1

    consecutive_fails = 0

    while consecutive_fails < config.fails:
        possible_image_names = _page_image_names(config, next_page)

        # Every filename variant is still tried for the current page even once the fail count
        # has technically reached `config.fails` -- a later variant succeeding (e.g. the site
        # used the alternate suffix for this page) resets the streak, so a single fully-missing
        # page costs `len(possible_image_names)` fails, not 1. This keeps `fails` counting
        # "consecutive missing pages" while resetting on any success, not accumulating over the
        # whole run.
        for image_name in possible_image_names:
            logger.info('Downloading: %s', image_name)

            if dl_and_save_img(config.root_site + image_name, config.dl_dir / image_name, config.headers):
                consecutive_fails = 0
                break

            logger.info('Failed with %s', image_name)
            consecutive_fails += 1
            time.sleep(_THROTTLE_SECONDS)

        time.sleep(_THROTTLE_SECONDS)
        next_page += 1

    logger.info('Nothing more to download. Process ending.')

    return 0

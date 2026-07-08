#!/usr/bin/env python3
"""Download comic pages listed on the configured site into `config.dl_dir`."""

import logging
import pathlib
import time

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)


def get_last_saved(save_path: pathlib.Path, ext: str) -> int:
    """Retrieve the page number of the last saved file.

    Args:
        save_path (pathlib.Path): Directory downloaded pages are saved to.
        ext (str): Image file extension, including the leading dot.

    Returns:
        int: Last saved page number, or 0 if nothing has been saved yet.
    """
    if not (available := sorted(save_path.glob(f'*{ext}'), reverse=True)):
        logger.info('No prior save')
        return 0

    for latest in available:
        last_nbr_available = latest.stem.split('_', maxsplit=2)[1]

        if last_nbr_available.isdigit():
            return int(last_nbr_available)

    return 0


def dl_and_save_img(link: str, save_path: pathlib.Path, headers: dict[str, str]) -> bool:
    """Fetch an image from `link` and save it to `save_path`.

    Args:
        link (str): Image source link.
        save_path (pathlib.Path): Destination path.
        headers (dict[str, str]): Headers for the request.

    Returns:
        bool: True if the download succeeded, False otherwise.
    """
    try:
        resource = requests.get(link, headers=headers, stream=True, timeout=5, verify=True)
    except requests.RequestException as error:
        logger.warning('Distant resource %s unreachable: %r', link, error)
        return False

    if resource.ok:
        logger.info('Saving: %s', save_path.name)
        with save_path.open('wb') as img:
            for chunk in resource.iter_content(1024):
                img.write(chunk)
        return True

    logger.info('Image not found -- HTTP_CODE: %s', resource.status_code)
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
        last_page_available = get_last_saved(config.dl_dir, config.ext)
        logger.info('Last downloaded: %s', last_page_available)
        next_page = last_page_available + 1

    consecutive_fails = 0

    while consecutive_fails < config.fails:
        base_image_name = f'{config.bd_name}_{str(next_page).zfill(config.padded)}_{config.ends}'
        possible_image_names = [
            f'{base_image_name}{config.ext}',
            f'{base_image_name}_{config.extra}{config.ext}',
        ]

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
            time.sleep(1)

        time.sleep(1)
        next_page += 1

    logger.info('Nothing more to download. Process ending.')

    return 0

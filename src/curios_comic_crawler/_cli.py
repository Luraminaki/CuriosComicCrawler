#!/usr/bin/env python3
"""Shared CLI plumbing for the download and upscale entry points."""

import argparse
import logging
import pathlib
import time
from collections.abc import Callable

from . import __version__
from .config import AppConfig, load_config
from .logging_utils import configure_launcher_logging

logger = logging.getLogger(__name__)


def main(
    script_name: str,
    run: Callable[[AppConfig, argparse.Namespace], int],
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> int:
    """Parse arguments, load configuration, then run `run` with consistent logging/timing.

    Args:
        script_name (str): Name used for the log file stem and startup banner.
        run (Callable[[AppConfig, argparse.Namespace], int]): Entry point that performs the
            actual work.
        configure_parser (Callable[[argparse.ArgumentParser], None] | None, optional): Callback
            to add entry-point-specific arguments beyond the shared `-c/--configuration`.
            Defaults to None.

    Returns:
        int: Process exit code (0 on success, 1 on configuration error or crash).
    """
    # Parsed before logging is configured, so `--help`/bad arguments exit cleanly without
    # creating a log file.
    parser = argparse.ArgumentParser(prog=script_name)
    parser.add_argument('-c', '--configuration', help='Configuration file location')
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args()

    configure_launcher_logging(script_name)
    logger.info('Version %s', __version__)

    config_file = pathlib.Path(args.configuration) if args.configuration else pathlib.Path.cwd() / 'config.json'

    try:
        config = load_config(config_file)
    except Exception:
        logger.exception('Loading %s failed', config_file)
        return 1

    logger.info('%s acquired', config_file)

    tic = time.perf_counter()
    exit_code = 0
    try:
        exit_code = run(config, args)
    except Exception:
        logger.exception('%s crashed', script_name)
        exit_code = 1
    finally:
        logger.info('Elapsed time: %ss', round(time.perf_counter() - tic, 3))

    return exit_code

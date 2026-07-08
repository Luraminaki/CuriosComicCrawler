#!/usr/bin/env python3
"""Menu-driven launcher that runs the downloader and/or upscaler.

Runs interactively (asks what to do) when invoked with no flags, or fully non-interactively
when `--mode` is given -- the latter is meant for cron jobs / scripted re-runs.
"""

import argparse
import logging

from . import downloader, upscaler
from .config import AppConfig

logger = logging.getLogger(__name__)

_MODE_LABELS = {'d': 'download', 'u': 'upscale', 'b': 'both'}


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add launcher-specific arguments to `parser`.

    Args:
        parser (argparse.ArgumentParser): Parser to extend.
    """
    _ = parser.add_argument(
        '-m', '--mode', choices=sorted(_MODE_LABELS.values()),
        help='Skip the interactive menu and run this mode directly.',
    )
    _ = parser.add_argument(
        '--force-download', action='store_true',
        help='Re-download every page, ignoring what is already saved.',
    )
    _ = parser.add_argument(
        '--force-upscale', action='store_true',
        help='Re-upscale every downloaded page, ignoring what is already upscaled.',
    )


def _prompt_mode() -> str:
    while True:
        answer = input(
            'What do you want to run?\n'
            '  d) Download\n'
            '  u) Upscale\n'
            '  b) Both (download then upscale)\n'
            '> ',
        ).strip().lower()
        if answer in _MODE_LABELS:
            return _MODE_LABELS[answer]
        print(f"{answer!r} is not a valid choice.")


def _prompt_yes_no(prompt: str) -> bool:
    return input(f'{prompt} [y/N] ').strip().lower() in ('y', 'yes')


def run(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the downloader and/or upscaler, interactively or per `args`.

    Args:
        config (AppConfig): Application configuration.
        args (argparse.Namespace): Parsed launcher arguments (`mode`, `force_download`,
            `force_upscale` -- see `configure_parser`).

    Returns:
        int: 0 on success, non-zero if any step fails.
    """
    mode = args.mode
    force_download = args.force_download
    force_upscale = args.force_upscale

    if mode is None:
        mode = _prompt_mode()
        if mode in ('download', 'both'):
            force_download = _prompt_yes_no('Force re-download of every page?')
        if mode in ('upscale', 'both'):
            force_upscale = _prompt_yes_no('Force re-upscale of every downloaded page?')

    exit_code = 0

    if mode in ('download', 'both'):
        logger.info('Running downloader (force=%s)', force_download)
        exit_code = max(exit_code, downloader.run(config, force=force_download))

    if mode in ('upscale', 'both'):
        logger.info('Running upscaler (force=%s)', force_upscale)
        exit_code = max(exit_code, upscaler.run(config, force=force_upscale))

    return exit_code

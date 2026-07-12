#!/usr/bin/env python3
"""Console-script entry point: download comic pages."""

import argparse
import sys

from . import downloader
from ._cli import main


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download every page, ignoring what is already saved.',
    )


def cli() -> None:
    """Run the downloader and exit with its return code."""
    sys.exit(main(
        'comiccrawler-download',
        lambda config, args: downloader.run(config, force=args.force),
        configure_parser=_configure_parser,
    ))


if __name__ == '__main__':
    cli()

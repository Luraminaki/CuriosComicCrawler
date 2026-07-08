#!/usr/bin/env python3
"""Console-script entry point: download comic pages."""

import sys

from . import downloader
from ._cli import main


def cli() -> None:
    """Run the downloader and exit with its return code."""
    sys.exit(main('comiccrawler-download', lambda config, _args: downloader.run(config)))


if __name__ == '__main__':
    cli()

#!/usr/bin/env python3
"""Console-script entry point: upscale downloaded comic pages."""

import sys

from . import upscaler
from ._cli import main


def cli() -> None:
    """Run the upscaler and exit with its return code."""
    sys.exit(main('comiccrawler-upscale', lambda config, _args: upscaler.run(config)))


if __name__ == '__main__':
    cli()

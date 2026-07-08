#!/usr/bin/env python3
"""Console-script entry point: menu-driven launcher for download + upscale."""

import sys

from . import launcher
from ._cli import main


def cli() -> None:
    """Run the launcher and exit with its return code."""
    sys.exit(main('comiccrawler', launcher.run, configure_parser=launcher.configure_parser))


if __name__ == '__main__':
    cli()

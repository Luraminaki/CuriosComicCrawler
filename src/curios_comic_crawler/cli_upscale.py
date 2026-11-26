#!/usr/bin/env python3
"""Console-script entry point: upscale downloaded comic pages."""

import sys
from typing import TYPE_CHECKING

from curios_comic_crawler import upscaler
from curios_comic_crawler._cli import main

if TYPE_CHECKING:
    import argparse


def _configure_parser(parser: "argparse.ArgumentParser") -> None:
    """Add the upscale command's own options.

    Args:
        parser (argparse.ArgumentParser): The command's parser.
    """
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upscale every downloaded page, ignoring what is already upscaled.",
    )


def cli() -> None:
    """Run the upscaler and exit with its return code."""
    sys.exit(
        main(
            "comiccrawler-upscale",
            lambda config, args: upscaler.run(config, force=args.force),
            configure_parser=_configure_parser,
        )
    )


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""Menu-driven launcher that runs the downloader and/or upscaler.

Runs interactively (asks what to do) when invoked with no flags, or fully non-interactively
when `--mode` is given -- the latter is meant for cron jobs / scripted re-runs.
"""

import logging
import sys
from typing import TYPE_CHECKING

from curios_comic_crawler import downloader
from curios_comic_crawler._exit_codes import INTERRUPTED_EXIT_CODE

if TYPE_CHECKING:
    import argparse

    from curios_comic_crawler.config import AppConfig

logger = logging.getLogger(__name__)

_MODE_LABELS = {"d": "download", "u": "upscale", "b": "both"}


def configure_parser(parser: "argparse.ArgumentParser") -> None:
    """Add launcher-specific arguments to `parser`.

    Args:
        parser (argparse.ArgumentParser): Parser to extend.
    """
    _ = parser.add_argument(
        "-m",
        "--mode",
        choices=sorted(_MODE_LABELS.values()),
        help="Skip the interactive menu and run this mode directly.",
    )
    _ = parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download every page, ignoring what is already saved.",
    )
    _ = parser.add_argument(
        "--force-upscale",
        action="store_true",
        help="Re-upscale every downloaded page, ignoring what is already upscaled.",
    )


def _prompt_mode() -> str:
    """Ask which steps to run until a valid choice is typed.

    Returns:
        str: `"download"`, `"upscale"` or `"both"`.
    """
    while True:
        answer = (
            input(
                "What do you want to run?\n"
                + "  d) Download\n"
                + "  u) Upscale\n"
                + "  b) Both (download then upscale)\n"
                + "> ",
            )
            .strip()
            .lower()
        )
        if answer in _MODE_LABELS:
            return _MODE_LABELS[answer]
        print(f"{answer!r} is not a valid choice.")


def _prompt_yes_no(prompt: str) -> bool:
    """Ask a yes/no question (no by default).

    Args:
        prompt (str): The question.

    Returns:
        bool: True for "y" or "yes".
    """
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _ask(force_download: bool, force_upscale: bool) -> tuple[str, bool, bool]:
    """Ask what to run, and whether to force each step a `--force-*` flag didn't already set.

    Args:
        force_download (bool): `--force-download` (no question asked if given).
        force_upscale (bool): `--force-upscale` (no question asked if given).

    Returns:
        tuple[str, bool, bool]: The mode, then the download and upscale force choices.

    Raises:
        EOFError: If the input ends (Ctrl+D) before every question is answered.
    """
    mode = _prompt_mode()
    if mode in ("download", "both") and not force_download:
        force_download = _prompt_yes_no("Force re-download of every page?")
    if mode in ("upscale", "both") and not force_upscale:
        force_upscale = _prompt_yes_no("Force re-upscale of every downloaded page?")
    return mode, force_download, force_upscale


def run(config: "AppConfig", args: "argparse.Namespace") -> int:
    """Run the downloader and/or upscaler, interactively or per `args`.

    Args:
        config (AppConfig): Application configuration.
        args (argparse.Namespace): Parsed launcher arguments (`mode`, `force_download`,
            `force_upscale` -- see `configure_parser`).

    Returns:
        int: 0 on success, the failing step's code otherwise (1, or 130 if interrupted), 2 if
        `--mode` is missing and there is no terminal to ask from, 130 if the questions are
        left unanswered (Ctrl+D).
    """
    mode = args.mode
    force_download = args.force_download
    force_upscale = args.force_upscale

    if mode is None:
        if not sys.stdin.isatty():  # cron, pipes: nobody to answer the questions
            logger.error("No --mode given and no terminal to ask from: pass --mode")
            return 2
        try:
            mode, force_download, force_upscale = _ask(force_download, force_upscale)
        except EOFError:  # Ctrl+D: the terminal closed its input instead of answering
            logger.warning("No answer: nothing run")
            return INTERRUPTED_EXIT_CODE

    exit_code = 0

    if mode in ("download", "both"):
        logger.info("Running downloader (force=%s)", force_download)
        exit_code = max(exit_code, downloader.run(config, force=force_download))

    if mode in ("upscale", "both"):
        # Imported here: cv2 and ONNX Runtime cost ~94 MiB that a download-only run never uses.
        from curios_comic_crawler import upscaler  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        logger.info("Running upscaler (force=%s)", force_upscale)
        exit_code = max(exit_code, upscaler.run(config, force=force_upscale))

    return exit_code

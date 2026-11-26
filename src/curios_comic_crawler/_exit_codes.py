#!/usr/bin/env python3
"""Exit codes shared by the commands, kept apart so the CLI needn't import the upscaler."""

INTERRUPTED_EXIT_CODE = 130  # the shell convention for "stopped by SIGINT" (Ctrl+C)

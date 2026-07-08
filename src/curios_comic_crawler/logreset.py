#!/usr/bin/env python3
"""Reset logging configuration to a clean slate.

Useful when multiple processes or modules configure logging differently, to avoid
conflicting handlers and ensure a consistent logging configuration across the project.
"""

import logging
import logging.config


def reset_logging(conf: dict | None = None) -> None:
    """Resets logging.

    Removes any configured handlers and filters, then applies `conf` (if provided).

    Args:
        conf (dict | None, optional): A `logging.config.dictConfig`-compatible mapping (this
            stays a plain `dict`, not a pydantic model, since it is passed straight through to
            that stdlib API). Defaults to None.
    """
    root = logging.getLogger()
    _ = list(map(root.removeHandler, root.handlers[:]))
    _ = list(map(root.removeFilter, root.filters[:]))

    if conf is not None:
        logging.config.dictConfig(conf)

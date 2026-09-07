"""Consistent logging setup for scripts and the CLI."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, honouring the ``LOG_LEVEL`` env var."""
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

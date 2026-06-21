"""Structured logging setup for the ps14 package.

A single :func:`get_logger` returns a configured logger with a consistent format.
Idempotent: repeated calls do not add duplicate handlers.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_CONFIGURED: set[str] = set()


def get_logger(name: str = "ps14", level: str | int = "INFO") -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name (usually ``__name__`` or ``"ps14"``).
    level:
        Logging level name or integer.

    Returns
    -------
    logging.Logger
        A logger writing to stderr with the package format; handlers are added once.
    """
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]

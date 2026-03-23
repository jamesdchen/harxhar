"""Centralized logging configuration."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a stderr StreamHandler and standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

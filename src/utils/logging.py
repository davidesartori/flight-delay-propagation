"""Logging configuration for the application."""
import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configure global logging for the application.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if root.handlers:
        root.handlers.clear()

    root.addHandler(console_handler)

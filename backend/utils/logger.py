"""
backend/utils/logger.py
Centralized logging system for AutoEval-C.
Ensures every major agent action and error is recorded for audit.
"""

import logging
import sys
from pathlib import Path

from backend.config.config import LOG_FILE_PATH


def _build_formatter() -> logging.Formatter:
    """
    Returns the shared log formatter used by all handlers.
    """
    return logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _has_file_handler(logger: logging.Logger, log_path: Path) -> bool:
    """
    Checks whether the logger already has a file handler for the target path.
    """
    target_path = str(log_path.resolve())

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(handler.baseFilename).resolve()
            if str(handler_path) == target_path:
                return True

    return False


def _has_stdout_stream_handler(logger: logging.Logger) -> bool:
    """
    Checks whether the logger already has a stdout stream handler.
    """
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and getattr(handler, "stream", None) is sys.stdout
        ):
            return True

    return False


def _clear_existing_handlers(logger: logging.Logger) -> None:
    """
    Removes and closes all existing handlers for safe reconfiguration.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            continue


def setup_logger(
    name: str = "autoeval_engine",
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Configures and returns a logger instance that outputs to both
    the console and a persistent log file.

    force_reconfigure=True is useful for tests or controlled resets.
    """
    if not name or not name.strip():
        raise ValueError("Logger name cannot be empty.")

    logger = logging.getLogger(name.strip())
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if force_reconfigure:
        _clear_existing_handlers(logger)

    formatter = _build_formatter()
    log_path = Path(LOG_FILE_PATH)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create log directory: {log_path.parent}") from exc

    if not _has_file_handler(logger, log_path):
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to open log file: {log_path}") from exc

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not _has_stdout_stream_handler(logger):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# Global engine logger — imported by all agents and utilities
engine_logger = setup_logger()
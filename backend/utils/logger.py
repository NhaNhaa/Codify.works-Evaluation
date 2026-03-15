"""
backend/utils/logger.py
Centralized logging system for AutoEval-C.
Ensures every major agent action and error is recorded for audit.
"""

import logging
import sys
from pathlib import Path
from backend.config.config import LOG_FILE_PATH


def setup_logger(name: str = "autoeval_engine") -> logging.Logger:
    """
    Configures and returns a logger instance that outputs to both
    the console and a persistent log file.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate loggers if setup is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # ── FORMATTING ─────────────────────────────────────────────────
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── FILE HANDLER (Persistent logs) ─────────────────────────────
    log_path = Path(LOG_FILE_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ── STREAM HANDLER (VS Code Console) ───────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# Global engine logger — imported by all agents and utilities
engine_logger = setup_logger()
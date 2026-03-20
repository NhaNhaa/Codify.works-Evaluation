import logging
from pathlib import Path

import pytest

from backend.utils import logger as logger_module


def _cleanup_logger(logger_name: str) -> None:
    logger_instance = logging.getLogger(logger_name)
    for handler in list(logger_instance.handlers):
        logger_instance.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            continue


def test_setup_logger_creates_file_and_stream_handlers(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "engine.log"
    monkeypatch.setattr(logger_module, "LOG_FILE_PATH", str(log_file))

    logger_name = "test_logger_create_handlers"
    _cleanup_logger(logger_name)

    logger_instance = logger_module.setup_logger(
        name=logger_name,
        force_reconfigure=True,
    )

    assert logger_instance.name == logger_name
    assert logger_instance.level == logging.INFO
    assert logger_instance.propagate is False
    assert log_file.parent.exists()
    assert len(logger_instance.handlers) == 2
    assert any(isinstance(handler, logging.FileHandler) for handler in logger_instance.handlers)
    assert any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger_instance.handlers
    )

    _cleanup_logger(logger_name)


def test_setup_logger_does_not_duplicate_handlers(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "engine.log"
    monkeypatch.setattr(logger_module, "LOG_FILE_PATH", str(log_file))

    logger_name = "test_logger_no_duplicates"
    _cleanup_logger(logger_name)

    logger_first = logger_module.setup_logger(name=logger_name, force_reconfigure=True)
    logger_second = logger_module.setup_logger(name=logger_name)

    assert logger_first is logger_second
    assert len(logger_second.handlers) == 2

    _cleanup_logger(logger_name)


def test_setup_logger_force_reconfigure_keeps_clean_handler_count(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "engine.log"
    monkeypatch.setattr(logger_module, "LOG_FILE_PATH", str(log_file))

    logger_name = "test_logger_force_reconfigure"
    _cleanup_logger(logger_name)

    logger_instance = logger_module.setup_logger(name=logger_name, force_reconfigure=True)
    assert len(logger_instance.handlers) == 2

    logger_instance = logger_module.setup_logger(name=logger_name, force_reconfigure=True)
    assert len(logger_instance.handlers) == 2

    _cleanup_logger(logger_name)


def test_setup_logger_raises_for_empty_name():
    with pytest.raises(ValueError, match="Logger name cannot be empty"):
        logger_module.setup_logger(name="   ", force_reconfigure=True)
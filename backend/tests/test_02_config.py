"""
backend/tests/test_config.py
Pytest tests for configuration module.
Run with: pytest backend/tests/test_config.py -v
"""

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.config import config


# ----------------------------------------------------------------------
# Fixture to ensure a clean environment before each test
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_env_and_mock_dotenv(monkeypatch):
    """Remove all AutoEval-C related env vars and mock load_dotenv."""
    # List of all env vars used by config
    env_vars = [
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "MISTRAL_API_KEY",
        "ZHIPU_API_KEY",
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_BASE_URL",
        "AGENT1_MODEL",
        "AGENT2_MODEL",
        "AGENT3_MODEL",
        "EMBEDDING_MODEL_NAME",
        "CHROMA_DB_PATH",
        "CHROMA_SKILLS_COLLECTION",
        "CHROMA_REFERENCES_COLLECTION",
        "DATA_INPUTS_PATH",
        "DATA_OUTPUTS_PATH",
        "LOG_FILE_PATH",
        "API_HOST",
        "API_PORT",
        "API_RELOAD",
        "API_LOG_LEVEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "LLM_RESPONSE_FORMAT",
        "LLM_RETRY_DELAY_SECONDS",
        "LLM_MAX_RETRIES",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    # Mock load_dotenv to do nothing – prevents .env file interference
    with patch("dotenv.load_dotenv", return_value=None):
        yield


# ----------------------------------------------------------------------
# Helper function tests (pure logic, no env)
# ----------------------------------------------------------------------
def test_get_env_stripped(monkeypatch):
    from backend.config.config import _get_env_stripped

    monkeypatch.setenv("TEST_KEY", "  value  ")
    assert _get_env_stripped("TEST_KEY") == "value"
    assert _get_env_stripped("MISSING_KEY", "default") == "default"
    monkeypatch.setenv("TEST_EMPTY", "")
    assert _get_env_stripped("TEST_EMPTY", "default") == "default"
    monkeypatch.setenv("TEST_BLANK", "   ")
    assert _get_env_stripped("TEST_BLANK", "default") == "default"


def test_get_env_int(monkeypatch):
    from backend.config.config import _get_env_int

    monkeypatch.setenv("TEST_INT", "42")
    assert _get_env_int("TEST_INT", 0) == 42
    monkeypatch.setenv("TEST_INT", "not_int")
    assert _get_env_int("TEST_INT", 10) == 10
    monkeypatch.setenv("TEST_INT", "-5")
    assert _get_env_int("TEST_INT", 10, minimum=0) == 10  # below minimum
    monkeypatch.setenv("TEST_INT", "5")
    assert _get_env_int("TEST_INT", 10, minimum=0) == 5


def test_get_env_float(monkeypatch):
    from backend.config.config import _get_env_float

    monkeypatch.setenv("TEST_FLOAT", "0.1")
    assert _get_env_float("TEST_FLOAT", 0.5) == 0.1
    monkeypatch.setenv("TEST_FLOAT", "invalid")
    assert _get_env_float("TEST_FLOAT", 0.5) == 0.5
    monkeypatch.setenv("TEST_FLOAT", "1.5")
    assert _get_env_float("TEST_FLOAT", 0.5, minimum=0.0, maximum=1.0) == 0.5  # above max
    monkeypatch.setenv("TEST_FLOAT", "-0.5")
    assert _get_env_float("TEST_FLOAT", 0.5, minimum=0.0) == 0.5  # below min


def test_get_env_bool(monkeypatch):
    from backend.config.config import _get_env_bool

    for true_val in ["true", "1", "yes", "on", "TRUE", "ON"]:
        monkeypatch.setenv("TEST_BOOL", true_val)
        assert _get_env_bool("TEST_BOOL", False) is True
    for false_val in ["false", "0", "no", "off", "FALSE", "OFF"]:
        monkeypatch.setenv("TEST_BOOL", false_val)
        assert _get_env_bool("TEST_BOOL", True) is False
    monkeypatch.setenv("TEST_BOOL", "maybe")
    assert _get_env_bool("TEST_BOOL", True) is True  # invalid -> default


def test_get_normalized_path(monkeypatch):
    from backend.config.config import _get_normalized_path

    monkeypatch.setenv("TEST_PATH", "data//inputs//")
    assert _get_normalized_path("TEST_PATH", "default") == os.path.normpath("data/inputs")
    monkeypatch.delenv("TEST_PATH", raising=False)
    assert _get_normalized_path("TEST_PATH", "default/path") == os.path.normpath("default/path")


# ----------------------------------------------------------------------
# Integration tests with actual config values (using monkeypatch to simulate .env)
# ----------------------------------------------------------------------
def test_default_values(monkeypatch):
    """Test that defaults are used when env vars are missing."""
    importlib.reload(config)

    assert config.LLM_PROVIDER == "groq"
    assert config.AGENT_MODELS["agent1_skill_extractor"] == "moonshotai/kimi-k2-instruct"
    assert config.LLM_TEMPERATURE == 0.1
    assert config.LLM_MAX_TOKENS == 4096
    assert config.LLM_RESPONSE_FORMAT == "json_object"


def test_env_override(monkeypatch):
    """Test that env vars override defaults."""
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("AGENT1_MODEL", "mistral-large-latest")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "text")

    importlib.reload(config)

    assert config.LLM_PROVIDER == "mistral"
    assert config.AGENT_MODELS["agent1_skill_extractor"] == "mistral-large-latest"
    assert config.LLM_TEMPERATURE == 0.2
    assert config.LLM_MAX_TOKENS == 2048
    assert config.LLM_RESPONSE_FORMAT == "text"


def test_provider_config_valid(monkeypatch):
    """Test get_provider_config returns correct dict for a known provider."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    importlib.reload(config)

    provider = config.get_provider_config()
    assert provider["base_url"] == "https://api.groq.com/openai/v1"
    assert provider["api_key"] == "test-key-123"
    assert provider["mode"] == "realtime"


def test_provider_config_invalid(monkeypatch):
    """Test get_provider_config raises for unknown provider."""
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    importlib.reload(config)
    with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
        config.get_provider_config()


def test_provider_config_missing_key(monkeypatch):
    """Test get_provider_config raises when API key missing."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    # Do NOT set GROQ_API_KEY – it remains missing
    importlib.reload(config)
    with pytest.raises(ValueError, match="Missing API key"):
        config.get_provider_config()


def test_get_model_valid(monkeypatch):
    monkeypatch.setenv("AGENT1_MODEL", "custom-model")
    importlib.reload(config)
    assert config.get_model("agent1_skill_extractor") == "custom-model"


def test_get_model_invalid_key():
    with pytest.raises(ValueError, match="Invalid agent key"):
        config.get_model("wrong_key")


def test_get_llm_settings(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.15")
    monkeypatch.setenv("LLM_MAX_TOKENS", "5000")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_schema")
    importlib.reload(config)

    settings = config.get_llm_settings()
    assert settings["temperature"] == 0.15
    assert settings["max_tokens"] == 5000
    assert settings["response_format"] == "json_schema"


def test_get_rate_limit_settings(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_DELAY_SECONDS", "30")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    importlib.reload(config)

    settings = config.get_rate_limit_settings()
    assert settings["retry_delay"] == 30
    assert settings["max_retries"] == 5


def test_system_paths(monkeypatch, tmp_path):
    """Test path normalization."""
    test_path = tmp_path / "custom_chroma"
    monkeypatch.setenv("CHROMA_DB_PATH", str(test_path))
    importlib.reload(config)

    paths = config.get_system_paths()
    assert Path(paths["chroma_persist_directory"]) == test_path
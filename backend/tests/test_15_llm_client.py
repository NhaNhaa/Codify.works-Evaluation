"""
backend/tests/test_llm_client.py
Tests for the shared LLM client wrapper.
Uses mocking to avoid real API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.config import config
from backend.utils.llm_client import call_llm_with_retry


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture
def mock_openai_client():
    """Returns a mock OpenAI client."""
    client = MagicMock()
    # Default: successful response with a JSON object
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"key": "value"}'
    client.chat.completions.create.return_value = mock_response
    return client


def create_mock_response(content: str):
    """Helper to create a mock response with given content."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_call_llm_success(mock_openai_client):
    """Test basic successful call returns parsed content."""
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result == '{"key": "value"}'


def test_call_llm_with_response_format(mock_openai_client):
    """Test that response_format is passed when provided."""
    call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format={"type": "json_object"},
        agent_label="TEST",
    )
    # Check that create was called with response_format
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs.get("response_format") == {"type": "json_object"}


def test_call_llm_auto_json_from_env(mock_openai_client, monkeypatch):
    """Test that if prompt expects JSON and env has json_object, response_format is auto-added."""
    monkeypatch.setattr(config, "LLM_RESPONSE_FORMAT", "json_object")
    call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Return only JSON"}],
        agent_label="TEST",
    )
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs.get("response_format") == {"type": "json_object"}


def test_call_llm_retry_on_rate_limit(mock_openai_client):
    """Test retry on rate limit error."""
    # First call raises rate limit, second succeeds
    mock_openai_client.chat.completions.create.side_effect = [
        Exception("429 Too Many Requests"),
        create_mock_response('{"success": true}'),
    ]
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result == '{"success": true}'
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_call_llm_retry_on_transient_error(mock_openai_client):
    """Test retry on transient error."""
    mock_openai_client.chat.completions.create.side_effect = [
        Exception("timeout"),
        create_mock_response('{"success": true}'),
    ]
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result == '{"success": true}'
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_call_llm_max_retries_exceeded(mock_openai_client, monkeypatch):
    """Test that after max retries, returns None."""
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    mock_openai_client.chat.completions.create.side_effect = Exception("429 Rate Limit")
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result is None
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_call_llm_unsupported_parameter_fallback(mock_openai_client):
    """Test fallback to simpler request when optional parameter is rejected."""
    # First variant fails with unsupported parameter, second succeeds
    def side_effect(**kwargs):
        if "response_format" in kwargs:
            raise Exception("Unsupported parameter: response_format")
        return create_mock_response('{"fallback": true}')

    mock_openai_client.chat.completions.create.side_effect = side_effect
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format={"type": "json_object"},
        agent_label="TEST",
    )
    assert result == '{"fallback": true}'
    # Should be called twice: first with response_format, second without
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_call_llm_temperature_normalization(mock_openai_client):
    """Test temperature is normalized (clamped between 0 and 2)."""
    call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=-0.5,  # Should be clamped to 0.0
        agent_label="TEST",
    )
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.0

    call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=2.5,  # Should be clamped to 2.0
        agent_label="TEST",
    )
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 2.0


def test_call_llm_json_extraction(mock_openai_client):
    """Test JSON extraction from noisy response."""
    noisy = "Here is the JSON:\n```json\n{\"key\": \"value\"}\n```\nHope that helps."
    mock_openai_client.chat.completions.create.return_value = create_mock_response(noisy)
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Return JSON"}],
        agent_label="TEST",
    )
    # Should extract the JSON block
    assert result == '{"key": "value"}'
    # Also test when prompt expects JSON but extraction not needed (clean response)
    mock_openai_client.chat.completions.create.return_value = create_mock_response('{"clean": true}')
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Return JSON"}],
        agent_label="TEST",
    )
    assert result == '{"clean": true}'


def test_call_llm_empty_content(mock_openai_client):
    """Test handling of empty content."""
    mock_openai_client.chat.completions.create.return_value = create_mock_response("   ")
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result is None


def test_call_llm_invalid_client():
    """Test when client is None."""
    result = call_llm_with_retry(
        client=None,
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result is None


def test_call_llm_invalid_model(mock_openai_client):
    """Test when model is missing."""
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="   ",
        messages=[{"role": "user", "content": "Hello"}],
        agent_label="TEST",
    )
    assert result is None


def test_call_llm_invalid_messages(mock_openai_client):
    """Test invalid messages (empty list, missing role, etc.)."""
    # Empty list
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test",
        messages=[],
        agent_label="TEST",
    )
    assert result is None

    # Missing role
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test",
        messages=[{"content": "hello"}],
        agent_label="TEST",
    )
    assert result is None

    # Missing content
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test",
        messages=[{"role": "user"}],
        agent_label="TEST",
    )
    assert result is None

    # Blank content
    result = call_llm_with_retry(
        client=mock_openai_client,
        model="test",
        messages=[{"role": "user", "content": "   "}],
        agent_label="TEST",
    )
    assert result is None
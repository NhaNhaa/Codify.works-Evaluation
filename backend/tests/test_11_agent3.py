"""
backend/tests/test_11_agent3.py
Tests for Agent 3: Feedback Writer.
Mocks LLM calls and verifies feedback generation, self‑check, and sentence limits.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.agent3_feedback import FeedbackWriter, get_agent3
from backend.config.constants import (
    FEEDBACK_PASS_MAX_SENTENCES,
    FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION,
    MAX_SELF_CHECK_ATTEMPTS,
    STATUS_PASS,
    STATUS_FAIL,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock call_llm_with_retry."""
    mock_call = MagicMock()
    monkeypatch.setattr("backend.agents.agent3_feedback.call_llm_with_retry", mock_call)
    return mock_call


@pytest.fixture
def mock_config(monkeypatch):
    """Mock config.LLM_TEMPERATURE."""
    monkeypatch.setattr("backend.agents.agent3_feedback.config", MagicMock(LLM_TEMPERATURE=0.1))


@pytest.fixture
def mock_provider_config(monkeypatch):
    """Mock get_provider_config and get_model."""
    monkeypatch.setattr("backend.agents.agent3_feedback.get_provider_config",
                        lambda: {"base_url": "http://test", "api_key": "key", "mode": "realtime"})
    monkeypatch.setattr("backend.agents.agent3_feedback.get_model",
                        lambda x: "test-model")


@pytest.fixture
def mock_formatter(monkeypatch):
    """Mock build_output to return a dict with the input preserved under 'json'."""
    def fake_build_output(final_result):
        return {"json": final_result, "markdown": "md"}
    mock_build = MagicMock(side_effect=fake_build_output)
    monkeypatch.setattr("backend.agents.agent3_feedback.build_output", mock_build)
    return mock_build


@pytest.fixture
def feedback_writer(mock_provider_config, mock_config):
    """Return a FeedbackWriter instance with mocked dependencies."""
    return FeedbackWriter()


@pytest.fixture
def sample_skills():
    """Return a list of sample skill verdicts from Agent 2, using non‑deterministic skill texts."""
    return [
        {
            "skill": "Some generic skill A",  # not in deterministic families
            "rank": 1,
            "weight": 4,
            "status": "PASS",
            "line_start": 10,
            "line_end": 12,
            "student_snippet": "code",
            "recommended_fix": "",
            "feedback": "Technical evaluation completed.",
            "verified": True,
        },
        {
            "skill": "Some generic skill B",
            "rank": 2,
            "weight": 3,
            "status": "FAIL",
            "line_start": 10,
            "line_end": 12,
            "student_snippet": "code",
            "recommended_fix": "fix",
            "feedback": "Technical evaluation completed.",
            "verified": True,
        },
    ]


@pytest.fixture
def evaluation_result(sample_skills):
    """Full evaluation result dict as passed from Agent 2."""
    return {
        "student_id": "student_01",
        "assignment_id": "lab_01",
        "skills": sample_skills,
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_init(feedback_writer):
    """Test initialization."""
    assert feedback_writer.client is not None
    assert feedback_writer.model == "test-model"


def test_run_success(feedback_writer, evaluation_result, mock_llm_client, mock_formatter):
    """Test run returns enriched skills and calls formatter."""
    # Mock LLM to return some feedback and self‑check approvals
    mock_llm_client.side_effect = [
        "Well done! You correctly did it.",
        '{"approved": true}',
        "You forgot to do something.",
        '{"approved": true}',
    ]
    result = feedback_writer.run(evaluation_result)
    assert result is not None
    # result is the output from build_output, which has keys "json" and "markdown"
    assert "json" in result
    assert "markdown" in result
    assert result["json"]["student_id"] == "student_01"
    assert len(result["json"]["skills"]) == 2
    # Ensure feedback fields are populated
    for skill in result["json"]["skills"]:
        assert "feedback" in skill
        assert skill["feedback"] != "Technical evaluation completed."
    mock_formatter.assert_called_once()


def test_run_invalid_input(feedback_writer):
    """Test run with non‑dict or missing skills."""
    assert feedback_writer.run(None) is None
    assert feedback_writer.run({}) is None  # empty dict, no skills


def test_run_no_skills(feedback_writer):
    """Test run with empty skills list."""
    result = feedback_writer.run({"student_id": "s1", "assignment_id": "a1", "skills": []})
    assert result is None


def test_write_and_selfcheck_pass(feedback_writer, mock_llm_client):
    """Test _write_and_selfcheck_feedback for a PASS skill (non‑deterministic)."""
    skill = {
        "skill": "Generic skill",  # not in deterministic families
        "status": "PASS",
        "weight": 2,
        "student_snippet": "code",
        "recommended_fix": "",
        "feedback": "",
        "line_start": 5,
        "line_end": 7,
    }
    mock_llm_client.side_effect = [
        "You did it correctly.",
        '{"approved": true}',
    ]
    result = feedback_writer._write_and_selfcheck_feedback(skill)
    assert result["feedback"] == "You did it correctly."
    assert result["status"] == "PASS"
    assert mock_llm_client.call_count == 2


def test_write_and_selfcheck_fail_high_weight(feedback_writer, mock_llm_client):
    """Test FAIL with weight >= HIGH_WEIGHT_THRESHOLD (non‑deterministic)."""
    skill = {
        "skill": "Generic skill",
        "status": "FAIL",
        "weight": 4,
        "student_snippet": "code",
        "recommended_fix": "fix",
        "feedback": "",
        "line_start": 1,
        "line_end": 3,
    }
    mock_llm_client.side_effect = [
        "You made a mistake.",
        '{"approved": true}',
    ]
    result = feedback_writer._write_and_selfcheck_feedback(skill)
    assert "mistake" in result["feedback"].lower()
    assert result["status"] == "FAIL"


def test_write_and_selfcheck_self_check_retry(feedback_writer, mock_llm_client):
    """Test self‑check rejects first attempt, second is accepted (non‑deterministic)."""
    skill = {
        "skill": "Generic skill",
        "status": "PASS",
        "weight": 2,
        "student_snippet": "code",
        "recommended_fix": "",
        "feedback": "",
    }
    mock_llm_client.side_effect = [
        "First attempt feedback.",
        '{"approved": false, "reason": "too long"}',
        "Second attempt feedback.",
        '{"approved": true}',
    ]
    result = feedback_writer._write_and_selfcheck_feedback(skill)
    assert result["feedback"] == "Second attempt feedback."
    assert mock_llm_client.call_count == 4


def test_write_and_selfcheck_max_attempts(feedback_writer, mock_llm_client):
    """Test self‑check always rejects, uses last feedback after max attempts (non‑deterministic)."""
    skill = {
        "skill": "Generic skill",
        "status": "PASS",
        "weight": 2,
        "student_snippet": "code",
        "recommended_fix": "",
        "feedback": "",
    }
    responses = []
    for i in range(MAX_SELF_CHECK_ATTEMPTS):
        responses.append(f"Feedback attempt {i+1}")
        responses.append('{"approved": false}')
    mock_llm_client.side_effect = responses
    result = feedback_writer._write_and_selfcheck_feedback(skill)
    assert result["feedback"] == f"Feedback attempt {MAX_SELF_CHECK_ATTEMPTS}"
    assert mock_llm_client.call_count == MAX_SELF_CHECK_ATTEMPTS * 2


def test_generate_feedback_deterministic(feedback_writer):
    """Test that deterministic feedback is used when applicable."""
    skill = {
        "skill": "shift elements",
        "status": "PASS",
        "weight": 3,
        "student_snippet": "code",
        "recommended_fix": "",
        "feedback": "",
    }
    result = feedback_writer._generate_feedback(skill)
    # Should return deterministic shift_transform PASS feedback
    assert "left-shift copy pattern" in result


def test_generate_feedback_llm_call(feedback_writer, mock_llm_client):
    """Test that LLM is called for non‑deterministic skills."""
    skill = {
        "skill": "Some other skill",
        "status": "PASS",
        "weight": 2,
        "student_snippet": "code",
        "recommended_fix": "",
        "feedback": "",
        "line_start": 1,
        "line_end": 2,
    }
    mock_llm_client.return_value = "Generated feedback."
    result = feedback_writer._generate_feedback(skill)
    assert result == "Generated feedback."
    mock_llm_client.assert_called_once()
    # Ensure correct temperature and no response_format (plain text)
    args, kwargs = mock_llm_client.call_args
    assert kwargs["temperature"] == 0.1
    assert "response_format" not in kwargs


def test_generate_feedback_with_previous(feedback_writer, mock_llm_client):
    """Test that previous attempt is included when provided."""
    skill = {
        "skill": "Some skill",
        "status": "FAIL",
        "weight": 3,
        "student_snippet": "code",
        "recommended_fix": "fix",
        "feedback": "",
        "line_start": 1,
        "line_end": 2,
    }
    previous = "Old feedback."
    mock_llm_client.return_value = "New feedback."
    feedback_writer._generate_feedback(skill, previous)
    args, kwargs = mock_llm_client.call_args
    # The prompt should contain the previous attempt
    assert previous in kwargs["messages"][1]["content"]


def test_generate_feedback_llm_failure(feedback_writer, mock_llm_client):
    """Test fallback when LLM returns None."""
    skill = {
        "skill": "Some skill",
        "status": "PASS",
        "weight": 2,
        "student_snippet": "code",
        "recommended_fix": "",
        "feedback": "",
    }
    mock_llm_client.return_value = None
    result = feedback_writer._generate_feedback(skill)
    # Should fall back to deterministic if possible, else fallback string
    # Since skill is not recognized, fallback is used.
    assert result == "Your code correctly demonstrates this skill."


def test_build_deterministic_feedback(feedback_writer):
    """Test all deterministic feedback branches."""
    # Shift transform PASS
    skill = {"skill": "shift elements", "status": "PASS"}
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # Shift transform FAIL
    skill["status"] = "FAIL"
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # Shift temp safety PASS
    skill = {"skill": "temporary variable", "status": "PASS"}
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # Shift temp safety FAIL
    skill["status"] = "FAIL"
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # scanf PASS
    skill = {"skill": "scanf input", "status": "PASS"}
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # scanf FAIL
    skill["status"] = "FAIL"
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # array index PASS
    skill = {"skill": "arr[i] access", "status": "PASS"}
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # array index FAIL
    skill["status"] = "FAIL"
    assert feedback_writer._build_deterministic_feedback(skill) is not None
    # Other skill
    skill = {"skill": "other", "status": "PASS"}
    assert feedback_writer._build_deterministic_feedback(skill) is None


def test_build_fallback_feedback(feedback_writer):
    """Test fallback feedback strings."""
    skill = {"status": "PASS"}
    assert "correctly demonstrates" in feedback_writer._build_fallback_feedback(skill)
    skill["status"] = "FAIL"
    assert "does not fully demonstrate" in feedback_writer._build_fallback_feedback(skill)


def test_selfcheck_feedback_approved(feedback_writer, mock_llm_client):
    """Test self‑check returns True when approved."""
    mock_llm_client.return_value = '{"approved": true}'
    result = feedback_writer._selfcheck_feedback("skill", "PASS", 2, "feedback")
    assert result is True
    args, kwargs = mock_llm_client.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_selfcheck_feedback_not_approved(feedback_writer, mock_llm_client):
    """Test self‑check returns False when not approved."""
    mock_llm_client.return_value = '{"approved": false, "reason": "too long"}'
    result = feedback_writer._selfcheck_feedback("skill", "PASS", 2, "feedback")
    assert result is False


def test_selfcheck_feedback_llm_failure(feedback_writer, mock_llm_client):
    """Test self‑check returns True on LLM failure (conservative)."""
    mock_llm_client.return_value = None
    result = feedback_writer._selfcheck_feedback("skill", "PASS", 2, "feedback")
    assert result is True


def test_selfcheck_feedback_invalid_json(feedback_writer, mock_llm_client):
    """Test self‑check returns True on parse error."""
    mock_llm_client.return_value = "not json"
    result = feedback_writer._selfcheck_feedback("skill", "PASS", 2, "feedback")
    assert result is True


def test_enforce_sentence_limits_pass(feedback_writer):
    """Test sentence limit for PASS."""
    long_feedback = "First sentence. Second sentence. Third sentence. Fourth sentence."
    limited = feedback_writer._enforce_sentence_limits(long_feedback, "PASS")
    # Should be truncated to FEEDBACK_PASS_MAX_SENTENCES (3)
    assert len(limited.split(". ")) == FEEDBACK_PASS_MAX_SENTENCES


def test_enforce_sentence_limits_fail(feedback_writer):
    """Test sentence limit for FAIL."""
    long_feedback = "First. Second. Third. Fourth."
    limited = feedback_writer._enforce_sentence_limits(long_feedback, "FAIL")
    assert len(limited.split(". ")) == FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION


def test_sanitize_feedback_precision_risky_phrases(feedback_writer):
    """Test risky phrases are replaced with deterministic feedback."""
    skill = {"skill": "shift elements", "status": "PASS"}
    feedback = "The whole program works correctly."
    sanitized = feedback_writer._sanitize_feedback_precision(feedback, skill)
    # Should use deterministic feedback
    assert "left-shift copy pattern" in sanitized


def test_sanitize_feedback_precision_front_replacement(feedback_writer):
    """Test 'front' is replaced with 'last position' when appropriate."""
    skill = {"skill": "shift elements", "recommended_fix": "arr[4] = temp;"}
    feedback = "The value should be placed at the front."
    sanitized = feedback_writer._sanitize_feedback_precision(feedback, skill)
    assert "front" not in sanitized
    assert "last position" in sanitized


def test_sanitize_feedback_precision_no_change(feedback_writer):
    """Test safe feedback is unchanged."""
    skill = {"skill": "other", "recommended_fix": ""}
    feedback = "This is safe feedback."
    sanitized = feedback_writer._sanitize_feedback_precision(feedback, skill)
    assert sanitized == "This is safe feedback."


def test_truncate_to_sentences(feedback_writer):
    """Test sentence truncation utility."""
    text = "First. Second? Third! Fourth."
    truncated = feedback_writer._truncate_to_sentences(text, 2)
    assert truncated == "First. Second?"

    # Already within limit
    truncated = feedback_writer._truncate_to_sentences(text, 5)
    assert truncated == text


def test_detect_skill_family(feedback_writer):
    """Test skill family detection."""
    assert feedback_writer._detect_skill_family("shift elements") == "shift_transform"
    assert feedback_writer._detect_skill_family("use temporary variable") == "shift_temp_safety"
    assert feedback_writer._detect_skill_family("scanf input") == "scanf_input"
    assert feedback_writer._detect_skill_family("arr[i] index") == "array_index"
    assert feedback_writer._detect_skill_family("other") == "other"


def test_clean_json(feedback_writer):
    """Test cleaning of JSON strings."""
    content = "```json\n{\"key\": \"value\"}\n```"
    assert feedback_writer._clean_json(content) == '{"key": "value"}'
    content2 = "```\n[1,2,3]\n```"
    assert feedback_writer._clean_json(content2) == "[1,2,3]"
    content3 = "   plain text   "
    assert feedback_writer._clean_json(content3) == "plain text"


def test_safe_int(feedback_writer):
    """Test safe integer conversion."""
    assert feedback_writer._safe_int("5", 0) == 5
    assert feedback_writer._safe_int("abc", 10) == 10
    assert feedback_writer._safe_int(None, 10) == 10


def test_get_agent3_singleton(mock_provider_config, mock_config):
    """Test that get_agent3 returns the same instance."""
    a1 = get_agent3()
    a2 = get_agent3()
    assert a1 is a2
"""
backend/tests/test_10_agent2.py
Tests for Agent 2: Evaluator.
Mocks all external dependencies (LLM, RAG, file I/O).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.agent2_evaluator import Evaluator, get_agent2
from backend.config.constants import MAX_REVERIFICATION_ATTEMPTS, MAX_SNIPPET_LINES, STATUS_PASS, STATUS_FAIL


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock call_llm_with_retry. No default side_effect – tests must set it."""
    mock_call = MagicMock()
    monkeypatch.setattr("backend.agents.agent2_evaluator.call_llm_with_retry", mock_call)
    return mock_call


@pytest.fixture
def mock_rag(monkeypatch):
    """Mock rag_pipeline methods."""
    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_micro_skills.return_value = [
        {"text": "Skill 1", "rank": 1, "weight": 5},
        {"text": "Skill 2", "rank": 2, "weight": 3},
        {"text": "Skill 3", "rank": 3, "weight": 2},
    ]
    # Default: teacher ref available for any rank
    def retrieve_teacher_ref(assignment_id, skill_rank):
        return {
            "rank": skill_rank,
            "snippet": f"teacher code for rank {skill_rank}",
            "line_start": 1,
            "line_end": 3,
            "status": "PASS",
        }
    mock_pipeline.retrieve_teacher_reference.side_effect = retrieve_teacher_ref
    monkeypatch.setattr("backend.agents.agent2_evaluator.rag_pipeline", mock_pipeline)
    return mock_pipeline


@pytest.fixture
def mock_security(monkeypatch):
    """Mock safe_read_file to return dummy student code."""
    mock_safe = MagicMock()
    mock_safe.return_value = "int main() { return 0; }"
    monkeypatch.setattr("backend.agents.agent2_evaluator.safe_read_file", mock_safe)
    return mock_safe


@pytest.fixture
def mock_config(monkeypatch):
    """Mock config.LLM_TEMPERATURE."""
    monkeypatch.setattr("backend.agents.agent2_evaluator.config", MagicMock(LLM_TEMPERATURE=0.1))


@pytest.fixture
def mock_provider_config(monkeypatch):
    """Mock get_provider_config and get_model."""
    monkeypatch.setattr("backend.agents.agent2_evaluator.get_provider_config",
                        lambda: {"base_url": "http://test", "api_key": "key", "mode": "realtime"})
    monkeypatch.setattr("backend.agents.agent2_evaluator.get_model",
                        lambda x: "test-model")


@pytest.fixture
def evaluator(mock_provider_config, mock_config):
    """Return an Evaluator instance with mocked dependencies."""
    return Evaluator()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_init(evaluator):
    """Test initialization."""
    assert evaluator.client is not None
    assert evaluator.model == "test-model"


def test_run_success(evaluator, mock_rag, mock_security, mock_llm_client):
    """Test successful run with all skills evaluated."""
    # Prepare mock: for each skill, evaluation returns PASS, verification returns true
    # 3 skills * 2 calls = 6 items
    mock_llm_client.side_effect = [
        '{"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Good."}',
        '{"confirmed": true}',
        '{"status": "PASS", "line_start": 20, "line_end": 22, "student_snippet": "code2", "recommended_fix": null, "feedback": "Good2."}',
        '{"confirmed": true}',
        '{"status": "PASS", "line_start": 30, "line_end": 32, "student_snippet": "code3", "recommended_fix": null, "feedback": "Good3."}',
        '{"confirmed": true}',
    ]
    result = evaluator.run("lab1", "student1", "path/to/student.c")
    assert result is not None
    assert result["student_id"] == "student1"
    assert result["assignment_id"] == "lab1"
    assert len(result["skills"]) == 3
    assert mock_llm_client.call_count == 6


def test_run_missing_student_file(evaluator, mock_security, mock_rag):
    """Test when student file cannot be read."""
    mock_security.return_value = None
    result = evaluator.run("lab1", "student1", "bad_path")
    assert result is None


def test_run_no_skills(evaluator, mock_rag, mock_security):
    """Test when no micro skills are found."""
    mock_rag.retrieve_micro_skills.return_value = []
    result = evaluator.run("lab1", "student1", "path")
    assert result is None


def test_run_invalid_identifier(evaluator):
    """Test with empty assignment_id or student_id."""
    assert evaluator.run("", "student1", "path") is None
    assert evaluator.run("lab1", "", "path") is None


def test_evaluate_skill_with_verification_success(evaluator, mock_rag, mock_llm_client):
    """Test evaluation of one skill with successful verification."""
    mock_llm_client.side_effect = [
        '{"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Good job."}',
        '{"confirmed": true}',
    ]
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    student_code = "int main() {}"
    verdict = evaluator._evaluate_skill_with_verification(skill, student_code, "lab1")
    assert verdict["rank"] == 1
    assert verdict["verified"] is True


def test_evaluate_skill_no_teacher_ref(evaluator, mock_rag, mock_llm_client):
    """Test evaluation when teacher reference is missing."""
    # Override rag to return no teacher ref
    mock_rag.retrieve_teacher_reference.side_effect = None
    mock_rag.retrieve_teacher_reference.return_value = None
    # Only one evaluation call needed
    mock_llm_client.side_effect = [
        '{"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Good."}',
    ]
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    verdict = evaluator._evaluate_skill_with_verification(skill, "code", "lab1")
    assert verdict["rank"] == 1
    assert verdict["verified"] is False
    mock_llm_client.assert_called_once()


def test_evaluate_skill_verification_retry(evaluator, mock_rag, mock_llm_client):
    """Test verification loop retries on mismatch."""
    mock_llm_client.side_effect = [
        '{"status": "FAIL", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Wrong."}',
        '{"confirmed": false}',
        '{"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Correct."}',
        '{"confirmed": true}',
    ]
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    verdict = evaluator._evaluate_skill_with_verification(skill, "code", "lab1")
    assert verdict["status"] == "PASS"
    assert verdict["verified"] is True
    assert mock_llm_client.call_count == 4


def test_evaluate_skill_verification_max_attempts(evaluator, mock_rag, mock_llm_client):
    """Test verification loop stops after max attempts and keeps latest verdict."""
    mock_llm_client.side_effect = [
        '{"status": "FAIL", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Wrong."}',
        '{"confirmed": false}',
        '{"status": "FAIL", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Still wrong."}',
        '{"confirmed": false}',
        '{"status": "FAIL", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Still wrong."}',
        '{"confirmed": false}',
    ]
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    verdict = evaluator._evaluate_skill_with_verification(skill, "code", "lab1")
    assert verdict["status"] == "FAIL"
    assert verdict["verified"] is False
    # 1 evaluation + 3 verifies + 2 re-evaluations = 6
    assert mock_llm_client.call_count == 6


def test_evaluate_student_code_success(evaluator, mock_llm_client):
    """Test _evaluate_student_code returns normalized verdict."""
    mock_llm_client.return_value = '{"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Good job."}'
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    verdict = evaluator._evaluate_student_code(skill, "code")
    assert verdict["skill"] == "Skill 1"
    assert verdict["rank"] == 1
    assert verdict["weight"] == 5
    assert verdict["status"] == "PASS"
    assert verdict["line_start"] == 10
    mock_llm_client.assert_called_once()
    # Check that response_format was passed
    args, kwargs = mock_llm_client.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_evaluate_student_code_with_teacher_ref(evaluator, mock_llm_client):
    """Test evaluation with teacher reference provided."""
    mock_llm_client.return_value = '{"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": null, "feedback": "Good job."}'
    teacher_ref = {"snippet": "teacher code", "line_start": 1, "line_end": 3}
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    evaluator._evaluate_student_code(skill, "code", teacher_ref)
    mock_llm_client.assert_called_once()


def test_evaluate_student_code_llm_failure(evaluator, mock_llm_client):
    """Test when LLM returns None."""
    mock_llm_client.return_value = None
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    verdict = evaluator._evaluate_student_code(skill, "code")
    assert verdict["status"] == "FAIL"
    assert "failed due to an internal error" in verdict["feedback"]


def test_evaluate_student_code_json_parse_error(evaluator, mock_llm_client):
    """Test when LLM returns invalid JSON."""
    mock_llm_client.return_value = "not json"
    skill = {"text": "Skill 1", "rank": 1, "weight": 5}
    verdict = evaluator._evaluate_student_code(skill, "code")
    assert verdict["status"] == "FAIL"


def test_verify_verdict_confirmed(evaluator, mock_llm_client):
    """Test verification returns True when confirmed."""
    mock_llm_client.return_value = '{"confirmed": true}'
    verdict = {"status": "PASS", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": "", "feedback": ""}
    teacher_ref = {"snippet": "teacher", "line_start": 1, "line_end": 3}
    result = evaluator._verify_verdict(verdict, teacher_ref, "fullcode", "skill")
    assert result is True


def test_verify_verdict_not_confirmed(evaluator, mock_llm_client):
    """Test verification returns False when not confirmed."""
    mock_llm_client.return_value = '{"confirmed": false, "reason": "wrong lines"}'
    verdict = {"status": "FAIL", "line_start": 10, "line_end": 12, "student_snippet": "code", "recommended_fix": "", "feedback": ""}
    teacher_ref = {"snippet": "teacher", "line_start": 1, "line_end": 3}
    result = evaluator._verify_verdict(verdict, teacher_ref, "fullcode", "skill")
    assert result is False


def test_verify_verdict_llm_failure(evaluator, mock_llm_client):
    """Test verification returns False when LLM fails."""
    mock_llm_client.return_value = None
    result = evaluator._verify_verdict({}, {"snippet": ""}, "", "")
    assert result is False


def test_force_teacher_fix_policy_pass(evaluator):
    """Test that for PASS, recommended_fix is cleared."""
    verdict = {"status": "PASS", "recommended_fix": "some fix"}
    teacher_ref = {"snippet": "teacher code"}
    result = evaluator._force_teacher_fix_policy(verdict, teacher_ref)
    assert result["recommended_fix"] == ""


def test_force_teacher_fix_policy_fail_with_ref(evaluator):
    """Test that for FAIL with teacher ref, recommended_fix is set to snippet."""
    verdict = {"status": "FAIL", "recommended_fix": "llm fix", "rank": 1}
    teacher_ref = {"snippet": "teacher code"}
    result = evaluator._force_teacher_fix_policy(verdict, teacher_ref)
    assert result["recommended_fix"] == "teacher code"


def test_force_teacher_fix_policy_fail_no_ref(evaluator):
    """Test that for FAIL without teacher ref, recommended_fix is cleared."""
    verdict = {"status": "FAIL", "recommended_fix": "llm fix", "rank": 1}
    result = evaluator._force_teacher_fix_policy(verdict, None)
    assert result["recommended_fix"] == ""


def test_normalize_verdict(evaluator):
    """Test _normalize_verdict cleans up fields."""
    parsed = {
        "status": "PASS",
        "line_start": 10,
        "line_end": 12,
        "student_snippet": "  code  ",
        "recommended_fix": "  fix  ",
        "feedback": "  feedback  ",
    }
    skill = {"text": "Skill", "rank": 1, "weight": 5}
    verdict = evaluator._normalize_verdict(parsed, skill)
    assert verdict["status"] == "PASS"
    assert verdict["line_start"] == 10
    assert verdict["line_end"] == 12
    assert verdict["student_snippet"] == "code"
    assert verdict["recommended_fix"] == "fix"
    assert verdict["feedback"] == "feedback"

    # Test missing fields
    verdict = evaluator._normalize_verdict({}, skill)
    assert verdict["status"] == "FAIL"
    assert verdict["line_start"] is None


def test_enforce_snippet_limits_within_limit(evaluator):
    """Test snippet within limit is not truncated."""
    verdict = {"student_snippet": "line1\nline2", "rank": 1}
    result = evaluator._enforce_snippet_limits(verdict)
    assert result["student_snippet"] == "line1\nline2"


def test_enforce_snippet_limits_exceeds_limit(evaluator):
    """Test snippet exceeding limit is truncated."""
    lines = [f"line{i}" for i in range(MAX_SNIPPET_LINES + 2)]
    verdict = {"student_snippet": "\n".join(lines), "rank": 1, "line_start": 5}
    result = evaluator._enforce_snippet_limits(verdict)
    assert len(result["student_snippet"].split("\n")) == MAX_SNIPPET_LINES
    assert result["line_end"] == 5 + MAX_SNIPPET_LINES - 1


def test_detect_skill_family(evaluator):
    assert evaluator._detect_skill_family("shift elements") == "shift_transform"
    assert evaluator._detect_skill_family("use temporary variable") == "shift_temp_safety"
    assert evaluator._detect_skill_family("scanf input") == "scanf_input"
    assert evaluator._detect_skill_family("arr[i] access") == "array_index"
    assert evaluator._detect_skill_family("other") == "other"


def test_number_lines(evaluator):
    code = "a\nb\nc"
    numbered = evaluator._number_lines(code)
    assert numbered == "1: a\n2: b\n3: c"


def test_strip_line_prefixes(evaluator):
    snippet = "1: code\n2:   indented"
    cleaned = evaluator._strip_line_prefixes(snippet)
    assert "1:" not in cleaned
    assert "code" in cleaned
    assert "indented" in cleaned


def test_clean_json(evaluator):
    content = "```json\n{\"key\": \"value\"}\n```"
    assert evaluator._clean_json(content) == '{"key": "value"}'


def test_is_valid_teacher_reference(evaluator):
    ref = {"rank": 1, "snippet": "code", "line_start": 1, "line_end": 3}
    assert evaluator._is_valid_teacher_reference(ref, 1) is True
    assert evaluator._is_valid_teacher_reference(ref, 2) is False  # wrong rank
    ref["rank"] = 1
    ref["snippet"] = ""
    assert evaluator._is_valid_teacher_reference(ref, 1) is False
    ref["snippet"] = "code"
    ref["line_start"] = 0
    assert evaluator._is_valid_teacher_reference(ref, 1) is False
    ref["line_start"] = 3
    ref["line_end"] = 1
    assert evaluator._is_valid_teacher_reference(ref, 1) is False


def test_get_agent2_singleton(mock_provider_config, mock_config):
    """Test that get_agent2 returns the same instance."""
    a1 = get_agent2()
    a2 = get_agent2()
    assert a1 is a2
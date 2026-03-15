"""
backend/tests/test_10_agent2.py
Tests for Agent 2 — Evaluator.
Run with: python -m pytest backend/tests/test_10_agent2.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from backend.config.constants import (
    MAX_REVERIFICATION_ATTEMPTS,
    DEFAULT_WEIGHT_DISTRIBUTIONS
)


# ── MOCK HELPERS ───────────────────────────────────────────────────
def make_mock_response(content: str):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


def make_agent():
    from backend.agents.agent2_evaluator import Evaluator
    agent = Evaluator.__new__(Evaluator)
    agent.model  = "test-model"
    agent.client = MagicMock()
    return agent


# ── FIXTURES ───────────────────────────────────────────────────────
SAMPLE_SKILL = {
    "text":   "Avoid losing data using temp variable",
    "rank":   1,
    "weight": 4
}

SAMPLE_TEACHER_REF = {
    "snippet":    "int temp = arr[0];",
    "line_start": 10,
    "line_end":   10,
    "status":     "PASS"
}

PASS_VERDICT = {
    "skill":           "Avoid losing data using temp variable",
    "rank":            1,
    "weight":          4,
    "status":          "PASS",
    "line_start":      8,
    "line_end":        8,
    "student_snippet": "int temp = arr[0];",
    "recommended_fix": None,
    "feedback":        "Correct use of temp variable."
}

FAIL_VERDICT = {
    "skill":           "Avoid losing data using temp variable",
    "rank":            1,
    "weight":          4,
    "status":          "FAIL",
    "line_start":      8,
    "line_end":        8,
    "student_snippet": "arr[0] = arr[1];",
    "recommended_fix": "int temp = arr[0];",
    "feedback":        "Data lost — no temp variable used."
}


# ══════════════════════════════════════════════════════════════════
# 1. _clean_json
# ══════════════════════════════════════════════════════════════════

def test_clean_json_strips_fences():
    from backend.agents.agent2_evaluator import Evaluator
    raw     = '```json\n{"status": "PASS"}\n```'
    cleaned = Evaluator._clean_json(raw)
    assert json.loads(cleaned) == {"status": "PASS"}


# ══════════════════════════════════════════════════════════════════
# 2. _evaluate_student_code
# ══════════════════════════════════════════════════════════════════

def test_evaluate_student_code_pass():
    """Should return PASS verdict when LLM returns PASS."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(PASS_VERDICT)
    )
    result = agent._evaluate_student_code(
        skill=SAMPLE_SKILL,
        student_code="int temp = arr[0];",
        teacher_ref=None
    )
    assert result["status"] == "PASS"
    assert result["rank"]   == 1


def test_evaluate_student_code_fail():
    """Should return FAIL verdict when LLM returns FAIL."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(FAIL_VERDICT)
    )
    result = agent._evaluate_student_code(
        skill=SAMPLE_SKILL,
        student_code="arr[0] = arr[1];",
        teacher_ref=None
    )
    assert result["status"] == "FAIL"


def test_evaluate_student_code_with_teacher_ref():
    """Should include teacher context when teacher_ref provided."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(PASS_VERDICT)
    )
    result = agent._evaluate_student_code(
        skill=SAMPLE_SKILL,
        student_code="int temp = arr[0];",
        teacher_ref=SAMPLE_TEACHER_REF
    )
    assert result["status"] == "PASS"


def test_evaluate_student_code_llm_exception_returns_fallback():
    """Should return safe FAIL fallback on LLM exception."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._evaluate_student_code(
        skill=SAMPLE_SKILL,
        student_code="code",
        teacher_ref=None
    )
    assert result["status"]   == "FAIL"
    assert result["verified"] == False


def test_evaluate_student_code_bad_json_returns_fallback():
    """Should return safe FAIL fallback on bad JSON from LLM."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        "not valid json"
    )
    result = agent._evaluate_student_code(
        skill=SAMPLE_SKILL,
        student_code="code",
        teacher_ref=None
    )
    assert result["status"] == "FAIL"


# ══════════════════════════════════════════════════════════════════
# 3. _verify_verdict
# ══════════════════════════════════════════════════════════════════

def test_verify_verdict_confirmed():
    """Should return True when LLM confirms verdict."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps({"confirmed": True})
    )
    result = agent._verify_verdict(
        verdict=PASS_VERDICT,
        teacher_ref=SAMPLE_TEACHER_REF,
        student_code="int temp = arr[0];",
        skill_text=SAMPLE_SKILL["text"]
    )
    assert result is True


def test_verify_verdict_not_confirmed():
    """Should return False when LLM rejects verdict."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps({"confirmed": False, "reason": "snippet mismatch"})
    )
    result = agent._verify_verdict(
        verdict=FAIL_VERDICT,
        teacher_ref=SAMPLE_TEACHER_REF,
        student_code="arr[0] = arr[1];",
        skill_text=SAMPLE_SKILL["text"]
    )
    assert result is False


def test_verify_verdict_llm_exception_returns_false():
    """Should return False on LLM exception — triggers re-evaluation."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._verify_verdict(
        verdict=PASS_VERDICT,
        teacher_ref=SAMPLE_TEACHER_REF,
        student_code="code",
        skill_text=SAMPLE_SKILL["text"]
    )
    assert result is False


# ══════════════════════════════════════════════════════════════════
# 4. _evaluate_skill_with_verification
# ══════════════════════════════════════════════════════════════════

def test_verification_passes_on_first_attempt():
    """Should confirm verdict on first attempt and set verified=True."""
    agent = make_agent()

    with patch("backend.agents.agent2_evaluator.rag_pipeline") as mock_rag:
        mock_rag.retrieve_teacher_reference.return_value = SAMPLE_TEACHER_REF

        # First call: evaluate → PASS verdict
        # Second call: verify → confirmed
        agent.client.chat.completions.create.side_effect = [
            make_mock_response(json.dumps(PASS_VERDICT)),
            make_mock_response(json.dumps({"confirmed": True}))
        ]

        result = agent._evaluate_skill_with_verification(
            skill=SAMPLE_SKILL,
            student_code="int temp = arr[0];",
            assignment_id="test_assignment"
        )

    assert result["verified"] is True
    assert result["status"]   == "PASS"


def test_verification_retries_on_mismatch():
    """Should retry up to MAX_REVERIFICATION_ATTEMPTS on mismatch."""
    agent = make_agent()

    with patch("backend.agents.agent2_evaluator.rag_pipeline") as mock_rag:
        mock_rag.retrieve_teacher_reference.return_value = SAMPLE_TEACHER_REF

        # Pattern: evaluate → not confirmed → re-evaluate → not confirmed → re-evaluate → confirmed
        agent.client.chat.completions.create.side_effect = [
            make_mock_response(json.dumps(FAIL_VERDICT)),          # initial eval
            make_mock_response(json.dumps({"confirmed": False})),  # verify attempt 1
            make_mock_response(json.dumps(FAIL_VERDICT)),          # re-eval attempt 1
            make_mock_response(json.dumps({"confirmed": False})),  # verify attempt 2
            make_mock_response(json.dumps(FAIL_VERDICT)),          # re-eval attempt 2
            make_mock_response(json.dumps({"confirmed": True})),   # verify attempt 3
        ]

        result = agent._evaluate_skill_with_verification(
            skill=SAMPLE_SKILL,
            student_code="arr[0] = arr[1];",
            assignment_id="test_assignment"
        )

    assert result["verified"] is True


def test_verification_accepts_after_max_attempts():
    """Should accept best verdict after MAX_REVERIFICATION_ATTEMPTS."""
    agent = make_agent()

    with patch("backend.agents.agent2_evaluator.rag_pipeline") as mock_rag:
        mock_rag.retrieve_teacher_reference.return_value = SAMPLE_TEACHER_REF

        # Always fails verification — never confirmed
        responses = []
        for _ in range(MAX_REVERIFICATION_ATTEMPTS):
            responses.append(make_mock_response(json.dumps(FAIL_VERDICT)))
            responses.append(make_mock_response(json.dumps({"confirmed": False})))

        agent.client.chat.completions.create.side_effect = responses

        result = agent._evaluate_skill_with_verification(
            skill=SAMPLE_SKILL,
            student_code="arr[0] = arr[1];",
            assignment_id="test_assignment"
        )

    # verified=False but verdict still returned — not None
    assert result is not None
    assert "status" in result


def test_no_teacher_ref_accepts_verdict_immediately():
    """Should accept verdict immediately when no teacher reference exists."""
    agent = make_agent()

    with patch("backend.agents.agent2_evaluator.rag_pipeline") as mock_rag:
        mock_rag.retrieve_teacher_reference.return_value = None

        agent.client.chat.completions.create.return_value = make_mock_response(
            json.dumps(PASS_VERDICT)
        )

        result = agent._evaluate_skill_with_verification(
            skill=SAMPLE_SKILL,
            student_code="int temp = arr[0];",
            assignment_id="test_assignment"
        )

    assert result["verified"] is True


# ── MAIN RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
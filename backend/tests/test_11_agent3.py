"""
backend/tests/test_11_agent3.py
Tests for Agent 3 — FeedbackWriter.
Run with: python -m pytest backend/tests/test_11_agent3.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from backend.config.constants import (
    HIGH_WEIGHT_THRESHOLD,
    MAX_SELF_CHECK_ATTEMPTS,
    DEFAULT_WEIGHT_DISTRIBUTIONS
)


# ── MOCK HELPERS ───────────────────────────────────────────────────
def make_mock_response(content: str):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


def make_agent():
    from backend.agents.agent3_feedback import FeedbackWriter
    agent = FeedbackWriter.__new__(FeedbackWriter)
    agent.model  = "test-model"
    agent.client = MagicMock()
    return agent


# ── FIXTURES ───────────────────────────────────────────────────────
PASS_SKILL = {
    "skill":           "Avoid losing data using temp variable",
    "rank":            1,
    "weight":          4,
    "status":          "PASS",
    "line_start":      8,
    "line_end":        8,
    "student_snippet": "int temp = arr[0];",
    "recommended_fix": None,
    "feedback":        "",
    "verified":        True
}

FAIL_HIGH_SKILL = {
    "skill":           "Shift elements using arr[i] = arr[i+1]",
    "rank":            2,
    "weight":          3,
    "status":          "FAIL",
    "line_start":      10,
    "line_end":        12,
    "student_snippet": "arr[i-1] = arr[i];",
    "recommended_fix": "arr[i] = arr[i + 1];",
    "feedback":        "",
    "verified":        True
}

FAIL_LOW_SKILL = {
    "skill":           "Use scanf to read integers",
    "rank":            4,
    "weight":          1,
    "status":          "FAIL",
    "line_start":      5,
    "line_end":        5,
    "student_snippet": "scanf('%d', arr[i]);",
    "recommended_fix": "scanf('%d', &arr[i]);",
    "feedback":        "",
    "verified":        True
}

SAMPLE_EVALUATION = {
    "student_id":    "student_01",
    "assignment_id": "lab_01",
    "skills":        [PASS_SKILL, FAIL_HIGH_SKILL, FAIL_LOW_SKILL]
}


# ══════════════════════════════════════════════════════════════════
# 1. _clean_json
# ══════════════════════════════════════════════════════════════════

def test_clean_json_strips_fences():
    from backend.agents.agent3_feedback import FeedbackWriter
    raw     = '```json\n{"approved": true}\n```'
    cleaned = FeedbackWriter._clean_json(raw)
    assert json.loads(cleaned) == {"approved": True}


# ══════════════════════════════════════════════════════════════════
# 2. _generate_feedback
# ══════════════════════════════════════════════════════════════════

def test_generate_feedback_pass_skill():
    """Should return non-empty feedback string for PASS skill."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        "Well done! Your use of a temp variable is correct."
    )
    result = agent._generate_feedback(PASS_SKILL)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_feedback_fail_high_weight():
    """Should return non-empty feedback for high-weight FAIL skill."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        "WHAT YOU DID: You used arr[i-1]... WHY IT IS WRONG: ... HOW TO FIX IT: ..."
    )
    result = agent._generate_feedback(FAIL_HIGH_SKILL)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_feedback_fail_low_weight():
    """Should return non-empty feedback for low-weight FAIL skill."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        "WHAT YOU DID: Missing & operator. HOW TO FIX IT: Use &arr[i]."
    )
    result = agent._generate_feedback(FAIL_LOW_SKILL)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_feedback_with_previous():
    """Should include previous feedback context when provided."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        "Improved feedback after self-check rejection."
    )
    result = agent._generate_feedback(FAIL_HIGH_SKILL, previous="Old feedback.")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_feedback_llm_exception_returns_fallback():
    """Should return fallback string on LLM exception."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._generate_feedback(PASS_SKILL)
    assert "could not be generated" in result.lower()


# ══════════════════════════════════════════════════════════════════
# 3. _selfcheck_feedback
# ══════════════════════════════════════════════════════════════════

def test_selfcheck_approved():
    """Should return True when LLM approves feedback."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps({"approved": True})
    )
    result = agent._selfcheck_feedback(
        skill_text="Avoid losing data using temp variable",
        status="PASS",
        weight=4,
        feedback="Great job using a temp variable!"
    )
    assert result is True


def test_selfcheck_rejected():
    """Should return False when LLM rejects feedback."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps({"approved": False, "reason": "Too vague"})
    )
    result = agent._selfcheck_feedback(
        skill_text="Shift elements",
        status="FAIL",
        weight=3,
        feedback="You did something wrong."
    )
    assert result is False


def test_selfcheck_llm_exception_returns_true():
    """Should return True on exception — avoids infinite loop."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._selfcheck_feedback(
        skill_text="Some skill",
        status="FAIL",
        weight=2,
        feedback="Some feedback"
    )
    assert result is True


# ══════════════════════════════════════════════════════════════════
# 4. _write_and_selfcheck_feedback
# ══════════════════════════════════════════════════════════════════

def test_write_and_selfcheck_approved_first_attempt():
    """Should return enriched skill with feedback on first approved attempt."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = [
        make_mock_response("Great job using a temp variable!"),   # generate
        make_mock_response(json.dumps({"approved": True}))        # selfcheck
    ]
    result = agent._write_and_selfcheck_feedback(PASS_SKILL.copy())
    assert "feedback" in result
    assert len(result["feedback"]) > 0


def test_write_and_selfcheck_rewrites_on_rejection():
    """Should rewrite feedback when self-check rejects first attempt."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = [
        make_mock_response("Vague feedback."),                              # generate
        make_mock_response(json.dumps({"approved": False, "reason": "vague"})),  # selfcheck fail
        make_mock_response("Improved specific feedback."),                  # regenerate
        make_mock_response(json.dumps({"approved": True}))                 # selfcheck pass
    ]
    result = agent._write_and_selfcheck_feedback(FAIL_HIGH_SKILL.copy())
    assert "feedback" in result
    assert len(result["feedback"]) > 0


def test_write_and_selfcheck_accepts_after_max_attempts():
    """Should accept best feedback after MAX_SELF_CHECK_ATTEMPTS."""
    agent = make_agent()
    responses = []
    for _ in range(MAX_SELF_CHECK_ATTEMPTS):
        responses.append(make_mock_response("Some feedback."))
        responses.append(make_mock_response(json.dumps({"approved": False, "reason": "still vague"})))
    agent.client.chat.completions.create.side_effect = responses

    result = agent._write_and_selfcheck_feedback(FAIL_LOW_SKILL.copy())
    assert "feedback" in result
    assert result["feedback"] is not None


# ══════════════════════════════════════════════════════════════════
# 5. run — full Phase 3 pipeline
# ══════════════════════════════════════════════════════════════════

def test_run_returns_json_and_markdown():
    """Should return dict with both json and markdown keys."""
    agent = make_agent()

    # Each skill needs: generate feedback + selfcheck approval
    responses = []
    for skill in SAMPLE_EVALUATION["skills"]:
        responses.append(make_mock_response("Good feedback text."))
        responses.append(make_mock_response(json.dumps({"approved": True})))

    agent.client.chat.completions.create.side_effect = responses

    result = agent.run(SAMPLE_EVALUATION)
    assert result is not None
    assert "json"     in result
    assert "markdown" in result
    assert isinstance(result["markdown"], str)
    assert len(result["markdown"]) > 0


def test_run_returns_none_on_empty_skills():
    """Should return None when no skills passed."""
    agent  = make_agent()
    result = agent.run({
        "student_id":    "student_01",
        "assignment_id": "lab_01",
        "skills":        []
    })
    assert result is None


def test_run_sorts_skills_by_weight_descending():
    """Skills in output should be sorted highest weight first."""
    agent = make_agent()

    responses = []
    for skill in SAMPLE_EVALUATION["skills"]:
        responses.append(make_mock_response("Feedback text."))
        responses.append(make_mock_response(json.dumps({"approved": True})))

    agent.client.chat.completions.create.side_effect = responses

    result  = agent.run(SAMPLE_EVALUATION)
    skills  = result["json"]["skills"]
    weights = [s["weight"] for s in skills]
    assert weights == sorted(weights, reverse=True)


# ── MAIN RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
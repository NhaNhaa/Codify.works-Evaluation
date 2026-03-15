"""
backend/tests/test_09_agent1.py
Tests for Agent 1 — SkillExtractor.
Run with: python -m pytest backend/tests/test_09_agent1.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from backend.config.constants import (
    MIN_MICRO_SKILLS,
    MAX_MICRO_SKILLS,
    TOTAL_WEIGHT_TARGET,
    DEFAULT_WEIGHT_DISTRIBUTIONS
)


# ── MOCK LLM RESPONSE HELPER ───────────────────────────────────────
def make_mock_response(content: str):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


# ── SHARED FIXTURES ────────────────────────────────────────────────
SAMPLE_SKILLS_4 = [
    "Use scanf to read integers into array",
    "Access array elements using arr[i]",
    "Avoid losing data using temp variable",
    "Shift elements using arr[i] = arr[i+1]"
]

SAMPLE_SKILLS_DICTS_4 = [
    {"text": s, "rank": i + 1, "weight": DEFAULT_WEIGHT_DISTRIBUTIONS[4][i]}
    for i, s in enumerate(SAMPLE_SKILLS_4)
]


# ── HELPER: Build agent
def make_agent():
    from backend.agents.agent1_extractor import SkillExtractor
    agent = SkillExtractor.__new__(SkillExtractor)
    agent.model  = "test-model"
    agent.client = MagicMock()
    return agent


# ══════════════════════════════════════════════════════════════════
# 1. CONSTANTS SANITY
# ══════════════════════════════════════════════════════════════════

def test_weight_distributions_sum_to_10():
    """Every distribution in the table must sum to exactly 10."""
    for count, weights in DEFAULT_WEIGHT_DISTRIBUTIONS.items():
        assert sum(weights) == TOTAL_WEIGHT_TARGET, \
            f"Weight sum for {count} skills = {sum(weights)}, expected {TOTAL_WEIGHT_TARGET}"


def test_skill_count_within_bounds():
    """Every key in distribution table must be within [MIN, MAX]."""
    for count in DEFAULT_WEIGHT_DISTRIBUTIONS:
        assert MIN_MICRO_SKILLS <= count <= MAX_MICRO_SKILLS


# ══════════════════════════════════════════════════════════════════
# 2. _clean_json
# ══════════════════════════════════════════════════════════════════

def test_clean_json_strips_fences():
    from backend.agents.agent1_extractor import SkillExtractor
    raw     = '```json\n["skill one", "skill two"]\n```'
    cleaned = SkillExtractor._clean_json(raw)
    parsed  = json.loads(cleaned)
    assert parsed == ["skill one", "skill two"]


def test_clean_json_no_fences_unchanged():
    from backend.agents.agent1_extractor import SkillExtractor
    raw    = '["skill one", "skill two"]'
    result = SkillExtractor._clean_json(raw)
    assert json.loads(result) == ["skill one", "skill two"]


# ══════════════════════════════════════════════════════════════════
# 3. _call_llm_for_list
# ══════════════════════════════════════════════════════════════════

def test_call_llm_for_list_valid():
    """Should return list when LLM returns valid JSON list."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(SAMPLE_SKILLS_4)
    )
    result = agent._call_llm_for_list("test prompt", "test step")
    assert isinstance(result, list)
    assert len(result) == 4


def test_call_llm_for_list_bad_json():
    """Should return empty list on invalid JSON."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        "this is not json"
    )
    result = agent._call_llm_for_list("test prompt", "test step")
    assert result == []


def test_call_llm_for_list_returns_dict_not_list():
    """Should return empty list when LLM returns dict instead of list."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps({"skill": "something"})
    )
    result = agent._call_llm_for_list("test prompt", "test step")
    assert result == []


def test_call_llm_for_list_llm_exception():
    """Should return empty list when LLM call raises exception."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._call_llm_for_list("test prompt", "test step")
    assert result == []


# ══════════════════════════════════════════════════════════════════
# 4. _rank_and_assign_weights
# ══════════════════════════════════════════════════════════════════

def test_rank_and_assign_weights_valid():
    """Should return correctly ranked and weighted skills."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(SAMPLE_SKILLS_4)
    )
    result = agent._rank_and_assign_weights(SAMPLE_SKILLS_4)

    assert len(result) >= MIN_MICRO_SKILLS
    assert len(result) <= MAX_MICRO_SKILLS
    assert sum(s["weight"] for s in result) == TOTAL_WEIGHT_TARGET
    assert result[0]["rank"] == 1
    assert result[-1]["rank"] == len(result)


def test_rank_and_assign_weights_below_min():
    """Should return empty list when LLM returns fewer than MIN skills."""
    agent = make_agent()
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(["only one skill"])
    )
    result = agent._rank_and_assign_weights(["only one skill"])
    assert result == []


def test_rank_and_assign_weights_truncates_to_max():
    """Should truncate to MAX_MICRO_SKILLS if LLM returns too many."""
    agent = make_agent()
    too_many = [f"skill {i}" for i in range(10)]
    agent.client.chat.completions.create.return_value = make_mock_response(
        json.dumps(too_many)
    )
    result = agent._rank_and_assign_weights(too_many)
    assert len(result) <= MAX_MICRO_SKILLS
    assert sum(s["weight"] for s in result) == TOTAL_WEIGHT_TARGET


# ══════════════════════════════════════════════════════════════════
# 5. _final_validation_loop
# ══════════════════════════════════════════════════════════════════

def test_final_validation_loop_all_valid():
    """Should return unchanged skills when all 3 files confirm valid."""
    agent = make_agent()
    valid_response = json.dumps({
        "valid": True,
        "skills": SAMPLE_SKILLS_4
    })
    agent.client.chat.completions.create.return_value = make_mock_response(
        valid_response
    )
    result = agent._final_validation_loop(
        skills=SAMPLE_SKILLS_DICTS_4,
        instructions="instructions content",
        starter="starter content",
        teacher="teacher content"
    )
    assert isinstance(result, list)
    assert len(result) >= MIN_MICRO_SKILLS
    assert sum(s["weight"] for s in result) == TOTAL_WEIGHT_TARGET


def test_final_validation_loop_revises_skills():
    """Should revise and return updated skills when LLM flags invalid."""
    agent = make_agent()
    revised_skills = [
        "Use scanf to read integers",
        "Access array elements using arr[i]",
        "Shift elements correctly"
    ]
    invalid_then_valid = [
        make_mock_response(json.dumps({"valid": False, "skills": revised_skills})),
        make_mock_response(json.dumps({"valid": True,  "skills": revised_skills})),
        make_mock_response(json.dumps({"valid": True,  "skills": revised_skills})),
    ]
    agent.client.chat.completions.create.side_effect = invalid_then_valid

    result = agent._final_validation_loop(
        skills=SAMPLE_SKILLS_DICTS_4,
        instructions="instructions content",
        starter="starter content",
        teacher="teacher content"
    )
    assert isinstance(result, list)
    assert sum(s["weight"] for s in result) == TOTAL_WEIGHT_TARGET


def test_final_validation_loop_llm_exception_returns_empty():
    """Should return empty list when LLM raises exception."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._final_validation_loop(
        skills=SAMPLE_SKILLS_DICTS_4,
        instructions="instructions content",
        starter="starter content",
        teacher="teacher content"
    )
    assert result == []


# ══════════════════════════════════════════════════════════════════
# 6. _generate_teacher_references
# ══════════════════════════════════════════════════════════════════

def test_generate_teacher_references_valid():
    """Should return one reference dict per skill."""
    agent = make_agent()
    ref_response = json.dumps({
        "snippet":    "arr[i] = arr[i + 1];",
        "line_start": 10,
        "line_end":   12
    })
    agent.client.chat.completions.create.return_value = make_mock_response(
        ref_response
    )
    result = agent._generate_teacher_references(
        skills=SAMPLE_SKILLS_DICTS_4,
        teacher="teacher code here"
    )
    assert len(result) == len(SAMPLE_SKILLS_DICTS_4)
    for ref in result:
        assert "snippet"    in ref
        assert "line_start" in ref
        assert "line_end"   in ref
        assert "rank"       in ref


def test_generate_teacher_references_llm_exception_returns_empty():
    """Should return empty list if any LLM call fails."""
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = Exception("API error")
    result = agent._generate_teacher_references(
        skills=SAMPLE_SKILLS_DICTS_4,
        teacher="teacher code here"
    )
    assert result == []


# ── MAIN RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
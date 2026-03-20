"""
backend/tests/test_09_agent1_extractor.py
Tests for Agent 1: Skill Extractor.
Mocks all external dependencies (LLM, RAG, file I/O, validators).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.agent1_extractor import SkillExtractor, get_agent1
from backend.config.constants import MIN_MICRO_SKILLS, MAX_MICRO_SKILLS


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_validators(monkeypatch):
    """Mock all validator functions to return predictable results."""
    mock_v = MagicMock()
    mock_v.validate_and_fix_skills.side_effect = lambda x: x  # identity
    mock_v.deduplicate_skills.side_effect = lambda x: x
    mock_v.reject_generic_skills.side_effect = lambda x: x
    mock_v.filter_skills_by_assignment_context.side_effect = lambda skills, **kwargs: skills  # accept kwargs
    mock_v.python_rank_and_weight.return_value = [
        {"text": "Skill 1", "rank": 1, "weight": 5},
        {"text": "Skill 2", "rank": 2, "weight": 3},
        {"text": "Skill 3", "rank": 3, "weight": 2},
    ]
    mock_v.SKILL_FORMULA_PROMPT = "FORMULA"
    mock_v.VALID_SKILL_VERBS = ["Be able to", "Use", "Avoid"]
    mock_v.MIN_SKILL_WORDS = 6
    mock_v.MAX_SKILL_WORDS = 15
    mock_v.number_lines.side_effect = lambda x: x
    mock_v.strip_line_prefixes.side_effect = lambda x: x
    mock_v.clean_json.side_effect = lambda x: x
    monkeypatch.setattr("backend.agents.agent1_extractor.validators", mock_v)
    return mock_v


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock call_llm_with_retry to return controlled JSON."""
    mock_call = MagicMock()
    mock_call.return_value = '["skill from llm"]'
    monkeypatch.setattr("backend.agents.agent1_extractor.call_llm_with_retry", mock_call)
    return mock_call


@pytest.fixture
def mock_rag(monkeypatch):
    """Mock rag_pipeline methods."""
    mock_pipeline = MagicMock()
    mock_pipeline.assignment_exists.return_value = False
    mock_pipeline.store_micro_skills.return_value = True
    mock_pipeline.store_teacher_references.return_value = True
    mock_pipeline.clear_assignment.return_value = True
    monkeypatch.setattr("backend.agents.agent1_extractor.rag_pipeline", mock_pipeline)
    return mock_pipeline


@pytest.fixture
def mock_security(monkeypatch):
    """Mock safe_read_file to return dummy content."""
    mock_safe = MagicMock()
    mock_safe.side_effect = lambda path, ext: f"content of {path}"
    monkeypatch.setattr("backend.agents.agent1_extractor.safe_read_file", mock_safe)
    return mock_safe


@pytest.fixture
def mock_skill_parser(monkeypatch):
    """Mock extract_skills to simulate parser results."""
    mock_extract = MagicMock()
    mock_extract.return_value = ["parsed skill 1", "parsed skill 2"]
    monkeypatch.setattr("backend.agents.agent1_extractor.extract_skills", mock_extract)
    return mock_extract


@pytest.fixture
def mock_config(monkeypatch):
    """Mock config functions."""
    monkeypatch.setattr("backend.agents.agent1_extractor.get_provider_config",
                        lambda: {"base_url": "http://test", "api_key": "key", "mode": "realtime"})
    monkeypatch.setattr("backend.agents.agent1_extractor.get_model",
                        lambda x: "test-model")


@pytest.fixture
def extractor(mock_config):
    """Return a SkillExtractor instance with mocked config."""
    return SkillExtractor()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_init(extractor):
    """Test initialization."""
    assert extractor.client is not None
    assert extractor.model == "test-model"


def test_run_assignment_exists_skip(extractor, mock_rag, mock_security, mock_skill_parser):
    """Test run when assignment exists and force_regenerate=False."""
    mock_rag.assignment_exists.return_value = True
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c", force_regenerate=False)
    assert result is True
    mock_rag.store_micro_skills.assert_not_called()
    mock_security.assert_not_called()  # files not read


def test_run_assignment_exists_force(extractor, mock_rag, mock_security, mock_skill_parser, mock_llm_client, mock_validators):
    """Test run when assignment exists and force_regenerate=True."""
    mock_rag.assignment_exists.return_value = True
    # Override is_same_concept to avoid duplicate rejection
    mock_validators.is_same_concept.return_value = False
    # Make LLM return a list for missing skills, and valid objects for teacher references
    mock_llm_client.side_effect = [
        '["new skill"]',                           # for missing skills
        '{"snippet": "code1", "line_start": 1, "line_end": 2}',  # for skill1
        '{"snippet": "code2", "line_start": 3, "line_end": 4}',  # for skill2
        '{"snippet": "code3", "line_start": 5, "line_end": 6}',  # for skill3
    ]
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c", force_regenerate=True)
    assert result is True
    mock_security.assert_called()  # files read
    # Agent 1 does not call clear_assignment directly; rag_pipeline.store_micro_skills handles it.
    # Instead, verify that store_micro_skills was called with force_regenerate=True.
    mock_rag.store_micro_skills.assert_called_once()
    args, kwargs = mock_rag.store_micro_skills.call_args
    assert kwargs["force_regenerate"] is True


def test_run_parser_skills_sufficient(extractor, mock_rag, mock_security, mock_skill_parser, mock_llm_client, mock_validators):
    """Test when parser returns >= MIN_MICRO_SKILLS -> no LLM cross-check."""
    mock_skill_parser.return_value = ["skill1", "skill2", "skill3"]
    # Mock teacher reference generation: need one JSON object per skill
    mock_llm_client.side_effect = [
        '{"snippet": "code1", "line_start": 1, "line_end": 2}',
        '{"snippet": "code2", "line_start": 3, "line_end": 4}',
        '{"snippet": "code3", "line_start": 5, "line_end": 6}',
    ]
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c")
    assert result is True
    # LLM should not be called for missing skills, only for teacher references
    assert mock_llm_client.call_count == 3
    mock_validators.python_rank_and_weight.assert_called_once_with(["skill1", "skill2", "skill3"])


def test_run_parser_skills_insufficient(extractor, mock_rag, mock_security, mock_skill_parser, mock_llm_client, mock_validators):
    """Test when parser returns < MIN_MICRO_SKILLS -> LLM cross-check."""
    mock_skill_parser.return_value = ["skill1"]  # less than min
    # Mock LLM responses: first for missing skills (return two new skills), then teacher references for all three skills.
    mock_llm_client.side_effect = [
        '["new skill 1", "new skill 2"]',          # missing skills (returns 2)
        '{"snippet": "code1", "line_start": 1, "line_end": 2}',  # skill1
        '{"snippet": "code2", "line_start": 3, "line_end": 4}',  # new skill 1
        '{"snippet": "code3", "line_start": 5, "line_end": 6}',  # new skill 2
    ]
    mock_validators.validate_and_fix_skills.return_value = ["new skill 1", "new skill 2"]
    mock_validators.is_same_concept.return_value = False  # avoid duplicate rejection
    # filter_skills_by_assignment_context already handles kwargs via fixture
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c")
    assert result is True
    # LLM should be called: 1 (missing) + 3 (teacher references) = 4 calls
    assert mock_llm_client.call_count == 4
    # Final skills should include original + two new (total 3)
    mock_validators.python_rank_and_weight.assert_called_once()
    args, _ = mock_validators.python_rank_and_weight.call_args
    assert "skill1" in args[0]
    assert "new skill 1" in args[0]
    assert "new skill 2" in args[0]
    assert len(args[0]) == 3


def test_run_parser_returns_none(extractor, mock_rag, mock_security, mock_skill_parser, mock_llm_client):
    """Test when extract_skills returns None (invalid file)."""
    mock_skill_parser.return_value = None
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c")
    assert result is False


def test_run_no_skills_after_all_processing(extractor, mock_rag, mock_security, mock_skill_parser, mock_llm_client, mock_validators):
    """Test when after all steps we still have < MIN_MICRO_SKILLS."""
    mock_skill_parser.return_value = []  # no parser skills
    mock_llm_client.return_value = '[]'  # LLM returns empty
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c")
    assert result is False


def test_rank_and_weight_failure(extractor, mock_rag, mock_security, mock_skill_parser, mock_validators):
    """Test when python_rank_and_weight returns empty (fails validation)."""
    mock_skill_parser.return_value = ["skill1", "skill2", "skill3"]
    mock_validators.python_rank_and_weight.return_value = []
    result = extractor.run("lab1", "inst.md", "starter.c", "teacher.c")
    assert result is False


def test_teacher_reference_generation_success(extractor, mock_llm_client, mock_validators):
    """Test _generate_teacher_references returns valid list."""
    skills = [{"rank": 1, "text": "skill1"}, {"rank": 2, "text": "skill2"}]
    teacher_code = "line1\nline2"
    mock_llm_client.side_effect = [
        '{"snippet": "code", "line_start": 1, "line_end": 2}',
        '{"snippet": "code2", "line_start": 3, "line_end": 4}',
    ]
    mock_validators.strip_line_prefixes.side_effect = lambda x: x
    result = extractor._generate_teacher_references(skills, teacher_code)
    assert len(result) == 2
    assert result[0]["rank"] == 1
    assert result[0]["snippet"] == "code"
    assert result[0]["line_start"] == 1
    assert result[0]["line_end"] == 2
    assert result[1]["rank"] == 2
    assert result[1]["snippet"] == "code2"
    assert result[1]["line_start"] == 3
    assert result[1]["line_end"] == 4


def test_teacher_reference_generation_failure(extractor, mock_llm_client):
    """Test when LLM returns None -> return empty list."""
    mock_llm_client.return_value = None
    result = extractor._generate_teacher_references([{"rank": 1, "text": "skill1"}], "code")
    assert result == []


def test_teacher_reference_validation(extractor):
    """Test _validate_teacher_references_before_storage."""
    refs = [
        {"rank": 1, "snippet": "code", "line_start": 1, "line_end": 2},
        {"rank": 2, "snippet": "code2", "line_start": 3, "line_end": 4},
    ]
    assert extractor._validate_teacher_references_before_storage(refs, 2) is True
    # Duplicate rank
    refs[1]["rank"] = 1
    assert extractor._validate_teacher_references_before_storage(refs, 2) is False


def test_call_llm_for_list_success(extractor, mock_llm_client, mock_validators):
    """Test _call_llm_for_list returns parsed list."""
    mock_llm_client.return_value = '["item1", "item2"]'
    mock_validators.clean_json.side_effect = lambda x: x
    result = extractor._call_llm_for_list("prompt", "test")
    assert result == ["item1", "item2"]


def test_call_llm_for_list_not_list(extractor, mock_llm_client, mock_validators):
    """Test when JSON is not a list."""
    mock_llm_client.return_value = '{"key": "value"}'
    mock_validators.clean_json.side_effect = lambda x: x
    result = extractor._call_llm_for_list("prompt", "test")
    assert result is None


def test_call_llm_for_list_invalid_json(extractor, mock_llm_client, mock_validators):
    """Test when JSON parsing fails."""
    mock_llm_client.return_value = 'not json'
    mock_validators.clean_json.side_effect = lambda x: x
    result = extractor._call_llm_for_list("prompt", "test")
    assert result is None


def test_to_positive_int(extractor):
    assert extractor._to_positive_int(5) == 5
    assert extractor._to_positive_int("5") == 5
    assert extractor._to_positive_int(-1) is None
    assert extractor._to_positive_int("abc") is None


def test_get_agent1_singleton(mock_config):
    """Test that get_agent1 returns same instance."""
    a1 = get_agent1()
    a2 = get_agent1()
    assert a1 is a2
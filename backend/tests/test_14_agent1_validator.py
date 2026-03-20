"""
backend/tests/test_14_agent1_validator.py
Tests for agent1_validators package – pure Python validation, ranking, deduplication.
"""

import pytest

# Public API (import package)
from backend.agents import agent1_validators as v

# Internal helpers (import from submodules)
from backend.agents.agent1_validators.validation import (
    _normalize_whitespace,
    _normalize_skill_text,
    _extract_list_item_text,
    _is_example_line,
    _contains_output_keyword,
    _get_io_subtype,
    MAX_SKILL_WORDS,
    MIN_SKILL_WORDS,
    VALID_SKILL_VERBS,
)
from backend.agents.agent1_validators.dedup_ranking import (
    _count_non_empty_string_skills,
    _remove_action_verb,
    _shares_subject_and_operation,
    _get_concept_bucket,
    _bucket_specific_overlap,
    _meaningful_containment,
    _references_first_boundary,
    _is_shift_boundary_split,
)
from backend.config.constants import MIN_MICRO_SKILLS, MAX_MICRO_SKILLS, TOTAL_WEIGHT_TARGET


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_skills():
    """Return a list of sample skills for testing."""
    return [
        "Be able to shift elements, using arr[i-1] and handle the 1st cell",
        "Avoid losing data during shifting, using a temporary variable",
        "Use scanf to read 5 integers into an array",
        "Be able to access array elements using index (arr[i])",
    ]


# ----------------------------------------------------------------------
# Tests for helper functions
# ----------------------------------------------------------------------
def test_normalize_whitespace():
    assert _normalize_whitespace("  hello   world  ") == "hello world"
    assert _normalize_whitespace("") == ""
    assert _normalize_whitespace(None) == ""


def test_normalize_skill_text():
    assert _normalize_skill_text("  Hello, World!  ") == "hello, world!"
    assert _normalize_skill_text("Skill.") == "skill"
    assert _normalize_skill_text("") == ""


def test_remove_action_verb():
    assert _remove_action_verb("be able to shift") == "shift"
    assert _remove_action_verb("Use scanf") == "scanf"
    assert _remove_action_verb("Avoid losing data") == "losing data"
    assert _remove_action_verb("Implement function") == "function"
    assert _remove_action_verb("No verb") == "No verb"


def test_extract_list_item_text():
    assert _extract_list_item_text("- item") == "item"
    assert _extract_list_item_text("* item") == "item"
    assert _extract_list_item_text("1. item") == "item"
    assert _extract_list_item_text("2) item") == "item"
    assert _extract_list_item_text("not a list") is None
    assert _extract_list_item_text("") is None


def test_is_example_line():
    assert _is_example_line("Example: input") is True
    assert _is_example_line("input: 5") is True
    assert _is_example_line("output: 10") is True
    assert _is_example_line("Sample output") is True
    assert _is_example_line("Normal skill") is False
    assert _is_example_line("") is False


def test_count_non_empty_string_skills():
    assert _count_non_empty_string_skills(["a", "", "b", None, "c"]) == 3
    assert _count_non_empty_string_skills([]) == 0
    assert _count_non_empty_string_skills(None) == 0


# ----------------------------------------------------------------------
# Tests for skill format validation
# ----------------------------------------------------------------------
def test_is_valid_skill_format():
    # Valid – meets verb and word count (>=6)
    assert v.is_valid_skill_format("Be able to shift elements using arr[i-1]") is True
    assert v.is_valid_skill_format("Use scanf to read 5 integers into an array") is True
    assert v.is_valid_skill_format("Avoid losing data during shifting with a temp variable") is True

    # Invalid – missing verb
    assert v.is_valid_skill_format("Shift elements") is False

    # Invalid – too short
    assert v.is_valid_skill_format("Be able to") is False  # 3 words < MIN

    # Invalid – too long
    long_skill = "Be able to " + "word " * 20
    assert v.is_valid_skill_format(long_skill) is False

    # Invalid – not a string
    assert v.is_valid_skill_format(None) is False
    assert v.is_valid_skill_format(123) is False


def test_auto_fix_skill_format():
    # Already valid (>=6 words)
    valid_skill = "Be able to shift elements using arr[i-1]"
    assert v.auto_fix_skill_format(valid_skill) == valid_skill

    # Missing verb, fixable, and word count still >=6 after fix
    fixable = "shift elements using arr[i-1]"
    fixed = v.auto_fix_skill_format(fixable)
    assert fixed == "Be able to shift elements using arr[i-1]"

    # Too long (should truncate)
    long_skill = "Be able to " + "word " * 20
    fixed = v.auto_fix_skill_format(long_skill)
    assert fixed is not None
    assert len(fixed.split()) <= MAX_SKILL_WORDS

    # Unfixable (too short after fixing)
    assert v.auto_fix_skill_format("hi") is None
    assert v.auto_fix_skill_format("") is None
    assert v.auto_fix_skill_format(None) is None


def test_validate_and_fix_skills():
    skills = [
        "Be able to shift elements using arr[i-1]",    # valid
        "shift elements using arr[i-1]",                # missing verb (fixable)
        "hi",                                           # unfixable
        None,                                           # invalid type
        "",                                             # empty
    ]
    result = v.validate_and_fix_skills(skills)
    assert len(result) == 2
    assert "Be able to shift elements using arr[i-1]" in result
    assert any("Be able to shift elements using arr[i-1]" in s for s in result)  # fixed version


# ----------------------------------------------------------------------
# Tests for deduplication
# ----------------------------------------------------------------------
def test_is_same_concept_layer1_exact():
    assert v.is_same_concept(
        "Be able to shift elements",
        "Be able to shift elements"
    ) is True
    # Different verb, same core
    assert v.is_same_concept(
        "Be able to shift elements",
        "Use shift elements"
    ) is True  # after verb removal, core is "shift elements"


def test_is_same_concept_layer2_shared_subject_operation():
    # Both reference "first element" and "shift"
    skill1 = "Be able to shift elements and handle the first cell"
    skill2 = "Use a temporary variable to store the first element before shifting"
    # They share subject "first element" and operation "shift"
    assert v.is_same_concept(skill1, skill2) is True


def test_is_same_concept_layer3_bucket_transform():
    # Example from project rules: shift + boundary should be considered duplicate
    skill1 = "Be able to shift elements using arr[i-1] and handle the 1st cell"
    skill2 = "Handle the boundary when first element wraps to last position"
    # skill1 includes both shift and boundary, so _is_shift_boundary_split should detect overlap
    assert v.is_same_concept(skill1, skill2) is True


def test_is_same_concept_no_overlap():
    skill1 = "Be able to shift elements"
    skill2 = "Use scanf to read integers"
    assert v.is_same_concept(skill1, skill2) is False


def test_deduplicate_skills():
    skills = [
        "Be able to shift elements and handle the 1st cell",      # includes boundary
        "Use temporary variable to store the first element before shifting",  # overlaps with first
        "Use scanf to read integers",                             # unique
        "Be able to shift elements and handle the 1st cell",      # duplicate
    ]
    result = v.deduplicate_skills(skills)
    assert len(result) == 2
    assert "Be able to shift elements and handle the 1st cell" in result
    assert "Use scanf to read integers" in result


# ----------------------------------------------------------------------
# Tests for generic skill rejection
# ----------------------------------------------------------------------
def test_is_generic_skill():
    # Generic
    assert v.is_generic_skill("Use a loop to iterate through array elements") is True
    assert v.is_generic_skill("Declare and initialize variables") is True
    # Not generic (has specificity)
    assert v.is_generic_skill("Use scanf to read 5 integers into an array") is False
    assert v.is_generic_skill("Avoid losing data during shifting, using a temporary variable") is False


def test_reject_generic_skills():
    skills = [
        "Use a loop to iterate through array",      # generic
        "Use scanf to read 5 integers",             # specific
        "Declare variable",                          # generic
        "Be able to shift elements",                 # specific (though generic pattern not matched)
    ]
    result = v.reject_generic_skills(skills)
    assert len(result) == 2
    assert "Use scanf to read 5 integers" in result
    assert "Be able to shift elements" in result


# ----------------------------------------------------------------------
# Tests for output skill filtering
# ----------------------------------------------------------------------
def test_instructions_explicitly_allow_output_skills():
    # Instructions with explicit output skill in labeled section
    instr = """
# Micro Skills to Evaluate
- Use printf to display the result
- Use scanf to read input
"""
    assert v.instructions_explicitly_allow_output_skills(instr) is True

    # Instructions with output keyword but no explicit objective section
    instr2 = "This program should print the shifted array."
    assert v.instructions_explicitly_allow_output_skills(instr2) is False

    # Instructions with example line
    instr3 = "Example output: 2 3 4 5 1"
    assert v.instructions_explicitly_allow_output_skills(instr3) is False

    # Instructions that mention "skills" but not in an objective context
    instr4 = "No output skills are required for this assignment."
    assert v.instructions_explicitly_allow_output_skills(instr4) is False

    # Empty instructions
    assert v.instructions_explicitly_allow_output_skills("") is False


def test_is_output_display_skill():
    assert v.is_output_display_skill("Use printf to display") is True
    assert v.is_output_display_skill("Print the array with commas") is True
    assert v.is_output_display_skill("Be able to shift elements") is False


def test_filter_skills_by_assignment_context():
    instructions = "This assignment does not require any printing."
    skills = [
        "Use scanf to read integers",
        "Print the result",                     # output skill
        "Be able to shift elements",
    ]
    result = v.filter_skills_by_assignment_context(skills, instructions)
    assert len(result) == 2
    assert "Print the result" not in result

    # With output allowed
    instructions2 = "# Skills to evaluate\n- Use printf to display"
    result2 = v.filter_skills_by_assignment_context(skills, instructions2)
    assert len(result2) == 3  # all kept


# ----------------------------------------------------------------------
# Tests for ranking and weighting
# ----------------------------------------------------------------------
def test_enforce_ranking_rules():
    skills = [
        "Use scanf to read integers",             # io
        "Be able to shift elements",               # transform
        "Avoid losing data",                       # safety
        "Access array elements",                   # middle
    ]
    ranked = v.enforce_ranking_rules(skills)
    # Expected order: transform, safety, middle, io
    assert ranked[0] == "Be able to shift elements"
    assert ranked[1] == "Avoid losing data"
    assert ranked[2] == "Access array elements"
    assert ranked[3] == "Use scanf to read integers"

    # Single skill
    assert v.enforce_ranking_rules(["Only one"]) == ["Only one"]


def test_python_rank_and_weight(sample_skills):
    result = v.python_rank_and_weight(sample_skills)
    assert len(result) == 4
    assert result[0]["rank"] == 1
    assert result[0]["weight"] == 4  # from DEFAULT_WEIGHT_DISTRIBUTIONS[4] = [4,3,2,1]
    assert result[1]["weight"] == 3
    assert result[2]["weight"] == 2
    assert result[3]["weight"] == 1
    total = sum(s["weight"] for s in result)
    assert total == TOTAL_WEIGHT_TARGET

    # Too few skills
    too_few = ["Skill1", "Skill2"]
    assert v.python_rank_and_weight(too_few) == []

    # Too many – should trim to MAX
    too_many = ["Skill" + str(i) for i in range(10)]
    result = v.python_rank_and_weight(too_many)
    assert len(result) == MAX_MICRO_SKILLS


# ----------------------------------------------------------------------
# Tests for snippet helpers
# ----------------------------------------------------------------------
def test_strip_line_prefixes():
    snippet = "1: int main() {\n2:     return 0;\n}"
    cleaned = v.strip_line_prefixes(snippet)
    assert "1:" not in cleaned
    assert "int main() {" in cleaned
    assert "    return 0;" in cleaned  # leading spaces preserved

    # With range prefix
    snippet2 = "Lines 1-3: int main() {\n    return 0;\n}"
    cleaned2 = v.strip_line_prefixes(snippet2)
    assert "Lines" not in cleaned2
    assert "int main() {" in cleaned2
    assert "    return 0;" in cleaned2


def test_number_lines():
    code = "int main() {\n    return 0;\n}"
    numbered = v.number_lines(code)
    assert numbered == "1: int main() {\n2:     return 0;\n3: }"


def test_clean_json():
    content = "```json\n{\"key\": \"value\"}\n```"
    assert v.clean_json(content) == '{"key": "value"}'
    content2 = "```\n[1,2,3]\n```"
    assert v.clean_json(content2) == "[1,2,3]"
    content3 = "   just text   "
    assert v.clean_json(content3) == "just text"


# ----------------------------------------------------------------------
# Edge cases and error handling
# ----------------------------------------------------------------------
def test_deduplicate_skills_invalid_input():
    assert v.deduplicate_skills(None) == []
    assert v.deduplicate_skills("not list") == []


def test_reject_generic_skills_invalid_input():
    assert v.reject_generic_skills(None) == []
    assert v.reject_generic_skills("not list") == []


def test_filter_skills_by_assignment_context_invalid():
    assert v.filter_skills_by_assignment_context(None, "") == []
    assert v.filter_skills_by_assignment_context([], None) == []


def test_python_rank_and_weight_invalid():
    assert v.python_rank_and_weight(None) == []
    assert v.python_rank_and_weight([]) == []
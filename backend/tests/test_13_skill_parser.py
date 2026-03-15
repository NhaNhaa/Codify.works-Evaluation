"""
backend/tests/test_13_skill_parser.py
Unit tests for backend/utils/skill_parser.py

Tests all 4 tiers:
  Tier 0 — empty / too short file → None
  Tier 1 — labeled section → exact extraction
  Tier 2 — unlabeled list  → exact extraction
  Tier 3 — pure prose      → [] LLM fallback signal

Run with: pytest backend/tests/test_13_skill_parser.py -v
"""

import pytest
from backend.utils.skill_parser import (
    extract_skills,
    has_explicit_skills,
    _has_content,
    _extract_from_labeled_section,
    _extract_from_any_list,
)


# ══════════════════════════════════════════════════════════════════
# TIER 0 — Content Validation
# ══════════════════════════════════════════════════════════════════

class TestTier0Validation:

    def test_empty_string_returns_none(self):
        result = extract_skills("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        result = extract_skills("   \n\n\t  ")
        assert result is None

    def test_too_short_returns_none(self):
        result = extract_skills("Hi")
        assert result is None

    def test_valid_content_does_not_return_none(self):
        result = extract_skills("# Assignment\nWrite a program to sort an array of integers.")
        assert result is not None

    def test_has_content_empty(self):
        assert _has_content("") is False

    def test_has_content_whitespace(self):
        assert _has_content("   ") is False

    def test_has_content_valid(self):
        assert _has_content("Write a program to sort an array.") is True


# ══════════════════════════════════════════════════════════════════
# TIER 1 — Labeled Section Extraction
# ══════════════════════════════════════════════════════════════════

class TestTier1LabeledSection:

    def test_micro_skills_to_evaluate_label(self):
        instructions = """
# Array Left Shift Assignment

## MICRO SKILLS TO EVALUATE
- Be able to access array elements using index (arr[i])
- Use scanf to read 5 integers into an array
- Be able to shift elements, using arr[i-1] and handle the 1st cell
- Avoid losing data during shifting, using a temporary variable
"""
        result = extract_skills(instructions)
        assert result == [
            "Be able to access array elements using index (arr[i])",
            "Use scanf to read 5 integers into an array",
            "Be able to shift elements, using arr[i-1] and handle the 1st cell",
            "Avoid losing data during shifting, using a temporary variable",
        ]

    def test_skills_to_evaluate_label(self):
        instructions = """
# Assignment

## SKILLS TO EVALUATE
- Use a for loop to iterate over array
- Use printf to print results
"""
        result = extract_skills(instructions)
        assert result == [
            "Use a for loop to iterate over array",
            "Use printf to print results",
        ]

    def test_learning_objectives_label(self):
        instructions = """
# Sorting Assignment

## LEARNING OBJECTIVES
1. Implement bubble sort algorithm
2. Use swap with temporary variable
3. Read integers with scanf
"""
        result = extract_skills(instructions)
        assert result == [
            "Implement bubble sort algorithm",
            "Use swap with temporary variable",
            "Read integers with scanf",
        ]

    def test_exact_text_preserved_with_parentheses(self):
        """Critical: parentheses, syntax examples must be preserved exactly."""
        instructions = """
## MICRO SKILLS
- Be able to access arr[i] using index notation
- Use scanf("%d", &arr[i]) to read integers
"""
        result = extract_skills(instructions)
        assert result[0] == "Be able to access arr[i] using index notation"
        assert result[1] == 'Use scanf("%d", &arr[i]) to read integers'

    def test_stops_at_next_section_heading(self):
        instructions = """
## MICRO SKILLS TO EVALUATE
- Skill one
- Skill two

## OTHER SECTION
- Not a skill
"""
        result = extract_skills(instructions)
        assert result == ["Skill one", "Skill two"]
        assert "Not a skill" not in result

    def test_labeled_section_with_asterisk_bullets(self):
        instructions = """
## SKILLS
* First skill
* Second skill
* Third skill
"""
        result = extract_skills(instructions)
        assert result == ["First skill", "Second skill", "Third skill"]

    def test_labeled_section_empty_returns_empty_list(self):
        instructions = """
## MICRO SKILLS TO EVALUATE

## NEXT SECTION
Some content here.
"""
        result = _extract_from_labeled_section(instructions)
        assert result == []

    def test_case_insensitive_label_matching(self):
        instructions = """
## micro skills to evaluate
- Skill one
- Skill two
"""
        result = extract_skills(instructions)
        assert len(result) == 2

    def test_h1_label_also_matched(self):
        instructions = """
# MICRO SKILLS
- Skill one
- Skill two
"""
        result = extract_skills(instructions)
        assert result == ["Skill one", "Skill two"]


# ══════════════════════════════════════════════════════════════════
# TIER 2 — Unlabeled List Extraction
# ══════════════════════════════════════════════════════════════════

class TestTier2UnlabeledList:

    def test_bullet_list_no_label(self):
        instructions = """
# Assignment

Write a program to reverse an array.

- Use a temporary variable for swapping
- Access elements using arr[i] notation
- Read integers with scanf
"""
        result = extract_skills(instructions)
        assert result == [
            "Use a temporary variable for swapping",
            "Access elements using arr[i] notation",
            "Read integers with scanf",
        ]

    def test_numbered_list_no_label(self):
        instructions = """
# Assignment

Complete the following:

1. Implement the sort function
2. Use bubble sort algorithm
3. Print sorted array
"""
        result = extract_skills(instructions)
        assert result == [
            "Implement the sort function",
            "Use bubble sort algorithm",
            "Print sorted array",
        ]

    def test_largest_list_selected(self):
        """When multiple lists exist, the largest is selected."""
        instructions = """
# Assignment

Requirements:
- Item A
- Item B

Steps:
1. Step one
2. Step two
3. Step three
4. Step four
"""
        result = extract_skills(instructions)
        # Numbered list has 4 items — larger than bullet list with 2
        assert len(result) == 4

    def test_single_item_list_ignored(self):
        """Single-item lists are not considered skill lists."""
        instructions = """
# Assignment

Note:
- Important note here

Write a program that sorts an array.
"""
        result = extract_skills(instructions)
        # Single-item list ignored → falls to Tier 3
        assert result == []

    def test_exact_text_preserved_tier2(self):
        instructions = """
# Assignment

- Be able to shift elements, using arr[i-1] and handle the 1st cell
- Avoid losing data during shifting, using a temporary variable
- Use scanf to read 5 integers into an array
"""
        result = extract_skills(instructions)
        assert result[0] == "Be able to shift elements, using arr[i-1] and handle the 1st cell"
        assert result[1] == "Avoid losing data during shifting, using a temporary variable"


# ══════════════════════════════════════════════════════════════════
# TIER 3 — Pure Prose Fallback
# ══════════════════════════════════════════════════════════════════

class TestTier3ProseFallback:

    def test_pure_prose_returns_empty_list(self):
        instructions = """
# Array Reversal Assignment

Write a program that reverses an array of 5 integers.
The program should read 5 integers from the user and print
the reversed array. Make sure to preserve all original values.
"""
        result = extract_skills(instructions)
        assert result == []

    def test_empty_list_is_llm_signal(self):
        """Empty list [] signals Agent 1 to use LLM generation."""
        result = extract_skills("# Assignment\nWrite a program to sort an array of five integers.")
        assert result == []
        assert result is not None  # None = abort, [] = LLM fallback


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════

class TestHasExplicitSkills:

    def test_returns_true_when_skills_found(self):
        instructions = """
## MICRO SKILLS TO EVALUATE
- Skill one
- Skill two
"""
        assert has_explicit_skills(instructions) is True

    def test_returns_false_for_prose(self):
        instructions = """
# Assignment
Write a sorting program.
"""
        assert has_explicit_skills(instructions) is False

    def test_returns_false_for_empty(self):
        assert has_explicit_skills("") is False


# ══════════════════════════════════════════════════════════════════
# REAL WORLD — Lab 01 instructions.md
# ══════════════════════════════════════════════════════════════════

class TestRealWorldLab01:
    """
    Tests against the actual lab_01 instructions.md content.
    Skills must be extracted exactly as written — no rephrasing.
    """

    LAB_01 = """# Array Left Shift Assignment

We have an array containing 5 numbers.
Write a program to shift all elements of an array one position to the left.
The first element becomes the last.

## INPUT-OUTPUT
1, 2, 3, 4, 5  ->  2, 3, 4, 5, 1
10, 9, 8, 7, 6  ->  9, 8, 7, 6, 10

## MICRO SKILLS TO EVALUATE
- Be able to access array elements using index (arr[i])
- Use scanf to read 5 integers into an array
- Be able to shift elements, using arr[i-1] and handle the 1st cell
- Avoid losing data during shifting, using a temporary variable
"""

    def test_extracts_exactly_4_skills(self):
        result = extract_skills(self.LAB_01)
        assert len(result) == 4

    def test_skill_1_exact(self):
        result = extract_skills(self.LAB_01)
        assert result[0] == "Be able to access array elements using index (arr[i])"

    def test_skill_2_exact(self):
        result = extract_skills(self.LAB_01)
        assert result[1] == "Use scanf to read 5 integers into an array"

    def test_skill_3_exact(self):
        result = extract_skills(self.LAB_01)
        assert result[2] == "Be able to shift elements, using arr[i-1] and handle the 1st cell"

    def test_skill_4_exact(self):
        result = extract_skills(self.LAB_01)
        assert result[3] == "Avoid losing data during shifting, using a temporary variable"

    def test_input_output_section_not_included(self):
        """INPUT-OUTPUT list items must not be picked up as skills."""
        result = extract_skills(self.LAB_01)
        for skill in result:
            assert "->" not in skill
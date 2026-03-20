from backend.utils import skill_parser as skill_parser_module


def test_extract_skills_returns_none_for_empty_content():
    assert skill_parser_module.extract_skills("") is None
    assert skill_parser_module.extract_skills("   ") is None


def test_extract_skills_tier1_from_labeled_section_with_bullets():
    instructions = """
# Assignment

## Micro Skills to Evaluate
- Use a temporary variable before overwrite
- Preserve original value during swap
- Print the final result correctly

## Notes
Do not use arrays.
"""

    result = skill_parser_module.extract_skills(instructions)

    assert result == [
        "Use a temporary variable before overwrite",
        "Preserve original value during swap",
        "Print the final result correctly",
    ]


def test_extract_skills_tier1_from_labeled_section_with_numbered_list():
    instructions = """
# Assignment Brief

## Learning Objectives
1. Read two integer inputs correctly
2. Swap values without data loss
3. Print the updated values

## Extra
Practice carefully.
"""

    result = skill_parser_module.extract_skills(instructions)

    assert result == [
        "Read two integer inputs correctly",
        "Swap values without data loss",
        "Print the updated values",
    ]


def test_extract_skills_tier2_from_unlabeled_list():
    instructions = """
This assignment checks whether the student can complete the required steps correctly.

- Read two numbers from input
- Preserve the first value before overwrite
- Print the swapped values

Use clean C syntax.
"""

    result = skill_parser_module.extract_skills(instructions)

    assert result == [
        "Read two numbers from input",
        "Preserve the first value before overwrite",
        "Print the swapped values",
    ]


def test_extract_skills_tier2_returns_largest_contiguous_list():
    instructions = """
Intro text long enough to pass the parser threshold.

- Small item

More explanation here.

1. First real skill
2. Second real skill
3. Third real skill
"""

    result = skill_parser_module.extract_skills(instructions)

    assert result == [
        "First real skill",
        "Second real skill",
        "Third real skill",
    ]


def test_extract_skills_returns_empty_list_for_prose_only():
    instructions = """
This assignment asks students to write a small C program that swaps two values,
explains the logic clearly, and demonstrates correct printed output without
providing an explicit list of skills anywhere in the instructions.
"""

    result = skill_parser_module.extract_skills(instructions)

    assert result == []


def test_has_explicit_skills_returns_true_when_list_exists():
    instructions = """
# Skills
- Read input safely
- Swap values correctly
"""

    assert skill_parser_module.has_explicit_skills(instructions) is True


def test_has_explicit_skills_returns_false_for_prose_only():
    instructions = """
This assignment describes expectations in paragraph form only and does not
include any bullet list or numbered list that can be extracted directly.
"""

    assert skill_parser_module.has_explicit_skills(instructions) is False
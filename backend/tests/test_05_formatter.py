import pytest

from backend.utils import formatter as formatter_module


def test_format_skill_feedback_pass_includes_symbol_and_code_block():
    skill = {
        "skill": "Uses temporary variable correctly",
        "status": "PASS",
        "weight": 3,
        "rank": 1,
        "line_start": 4,
        "line_end": 6,
        "student_snippet": "temp = a;\na = b;\nb = temp;",
        "feedback": "Correct swap logic.",
    }

    markdown = formatter_module.format_skill_feedback(skill)

    assert "### ✅ Skill 1 — Uses temporary variable correctly" in markdown
    assert "Status: PASS" in markdown
    assert "Lines 4–6" in markdown
    assert "```c" in markdown
    assert "Correct swap logic." in markdown


def test_format_skill_feedback_high_weight_fail_includes_why_section():
    skill = {
        "skill": "Preserves original value before overwrite",
        "status": "FAIL",
        "weight": 3,
        "rank": 1,
        "student_snippet": "a = b;",
        "feedback": "You overwrite the original value too early.",
        "recommended_fix": "temp = a;\na = b;\nb = temp;",
    }

    markdown = formatter_module.format_skill_feedback(skill)

    assert "**WHAT YOU DID:**" in markdown
    assert "**WHY IT IS WRONG:**" in markdown
    assert "**HOW TO FIX IT:**" in markdown


def test_format_skill_feedback_low_weight_fail_excludes_why_section():
    skill = {
        "skill": "Prints expected output",
        "status": "FAIL",
        "weight": 1,
        "rank": 3,
        "student_snippet": "printf(\"done\");",
        "recommended_fix": "printf(\"%d\", value);",
    }

    markdown = formatter_module.format_skill_feedback(skill)

    assert "**WHAT YOU DID:**" in markdown
    assert "**HOW TO FIX IT:**" in markdown
    assert "**WHY IT IS WRONG:**" not in markdown


def test_format_report_sorts_by_weight_then_rank():
    evaluation_result = {
        "student_id": "student_01",
        "assignment_id": "swap_task",
        "skills": [
            {"skill": "Third", "status": "PASS", "weight": 1, "rank": 3},
            {"skill": "First", "status": "PASS", "weight": 3, "rank": 2},
            {"skill": "Second", "status": "PASS", "weight": 3, "rank": 1},
        ],
    }

    markdown = formatter_module.format_report(evaluation_result)

    first_index = markdown.find("Second")
    second_index = markdown.find("First")
    third_index = markdown.find("Third")

    assert first_index < second_index < third_index


def test_format_report_handles_empty_skills_list():
    evaluation_result = {
        "student_id": "student_01",
        "assignment_id": "swap_task",
        "skills": [],
    }

    markdown = formatter_module.format_report(evaluation_result)

    assert "# Evaluation Report — student_01" in markdown
    assert "No skills to display." in markdown


def test_build_output_returns_json_and_markdown():
    evaluation_result = {
        "student_id": "student_01",
        "assignment_id": "swap_task",
        "skills": [
            {"skill": "Uses temporary variable", "status": "PASS", "weight": 3, "rank": 1}
        ],
    }

    output = formatter_module.build_output(evaluation_result)

    assert "json" in output
    assert "markdown" in output
    assert output["json"] == evaluation_result
    assert "# Evaluation Report — student_01" in output["markdown"]


def test_build_output_raises_for_invalid_input():
    with pytest.raises(ValueError, match="evaluation_result must be a dictionary"):
        formatter_module.build_output(None)
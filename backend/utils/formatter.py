"""
backend/utils/formatter.py
Converts internal JSON evaluation results into human-readable Markdown reports.
JSON is the internal truth — Markdown is the display layer.
Never writes Markdown directly — always derived from JSON.
"""

from backend.config.constants import (
    HIGH_WEIGHT_THRESHOLD,
    STRUCTURE_HIGH_WEIGHT,
    STRUCTURE_LOW_WEIGHT
)
from backend.utils.logger import engine_logger

# TODO:
# Move these fixed literals into backend/config/constants.py
# to fully comply with Single Source of Truth.
PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"
DEFAULT_UNKNOWN_SKILL = "Unknown Skill"
DEFAULT_FAIL_STATUS = "FAIL"
DEFAULT_LINE_REFERENCE = "Line reference unavailable"
DEFAULT_PASS_FEEDBACK = "Well done! Your implementation is correct."
DEFAULT_FAIL_FEEDBACK = "This implementation does not satisfy the micro skill requirement."
MARKDOWN_CODE_BLOCK_LANGUAGE = "c"


def _build_line_reference(line_start, line_end) -> str:
    """
    Builds a human-readable line reference safely.
    """
    if line_start is not None and line_end is not None:
        return f"Lines {line_start}–{line_end}"
    return DEFAULT_LINE_REFERENCE


def _normalize_skill_status(status: str) -> str:
    """
    Normalizes skill status to PASS or FAIL.
    Unknown values default to FAIL for safety.
    """
    if not status:
        return DEFAULT_FAIL_STATUS

    normalized_status = str(status).strip().upper()
    if normalized_status == "PASS":
        return "PASS"
    return DEFAULT_FAIL_STATUS


def format_skill_feedback(skill: dict) -> str:
    """
    Converts a single skill verdict dict into a Markdown block.
    Applies weight-based depth:
    - PASS        → medium depth
    - FAIL high   → detailed
    - FAIL low    → brief
    """
    if not isinstance(skill, dict):
        engine_logger.error("FORMATTER: Invalid skill entry. Expected dict.")
        return "### ❌ Invalid Skill Entry\n---"

    skill_text = skill.get("skill", DEFAULT_UNKNOWN_SKILL)
    status = _normalize_skill_status(skill.get("status", DEFAULT_FAIL_STATUS))
    weight = skill.get("weight", 1)
    rank = skill.get("rank", 0)
    line_start = skill.get("line_start")
    line_end = skill.get("line_end")
    snippet = skill.get("student_snippet", "")
    fix = skill.get("recommended_fix", "")
    feedback = skill.get("feedback", "")

    symbol = PASS_SYMBOL if status == "PASS" else FAIL_SYMBOL
    line_ref = _build_line_reference(line_start, line_end)

    lines = []

    lines.append(f"### {symbol} Skill {rank} — {skill_text}")
    lines.append(f"**Weight: {weight} | Status: {status} | {line_ref}**")
    lines.append("")

    if status == "PASS":
        lines.append(feedback if feedback else DEFAULT_PASS_FEEDBACK)
        lines.append("")

        if snippet:
            lines.append("**Your correct code:**")
            lines.append(f"```{MARKDOWN_CODE_BLOCK_LANGUAGE}")
            lines.append(snippet)
            lines.append("```")
            lines.append("")
    else:
        structure = (
            STRUCTURE_HIGH_WEIGHT
            if weight >= HIGH_WEIGHT_THRESHOLD
            else STRUCTURE_LOW_WEIGHT
        )

        for section in structure:
            section_content_added = False
            lines.append(f"**{section}:**")

            if section == "WHAT YOU DID":
                if snippet:
                    lines.append(f"```{MARKDOWN_CODE_BLOCK_LANGUAGE}")
                    lines.append(snippet)
                    lines.append("```")
                    section_content_added = True
                else:
                    lines.append("Relevant student code was not detected.")
                    section_content_added = True

            elif section == "WHY IT IS WRONG":
                lines.append(feedback if feedback else DEFAULT_FAIL_FEEDBACK)
                section_content_added = True

            elif section == "HOW TO FIX IT":
                if fix:
                    lines.append(f"```{MARKDOWN_CODE_BLOCK_LANGUAGE}")
                    lines.append(fix)
                    lines.append("```")
                    section_content_added = True
                else:
                    lines.append("Recommended fix is unavailable.")
                    section_content_added = True

            if not section_content_added:
                lines.append("No details available.")

            lines.append("")

    lines.append("---")
    return "\n".join(lines)


def format_report(evaluation_result: dict) -> str:
    """
    Converts the full evaluation JSON into a complete Markdown report.
    Skills are sorted by weight descending, then by rank ascending.
    No overall summary — per skill feedback only.
    """
    if not isinstance(evaluation_result, dict):
        engine_logger.error("FORMATTER: Invalid evaluation_result. Expected dict.")
        return "# Evaluation Report — unknown\nInvalid evaluation result."

    student_id = evaluation_result.get("student_id", "unknown")
    assignment_id = evaluation_result.get("assignment_id", "unknown")
    skills = evaluation_result.get("skills", [])

    if not isinstance(skills, list) or not skills:
        engine_logger.warning(f"FORMATTER: No skills found for {student_id}")
        return f"# Evaluation Report — {student_id}\n## Assignment: {assignment_id}\nNo skills to display."

    sorted_skills = sorted(
        skills,
        key=lambda skill: (-skill.get("weight", 0), skill.get("rank", 999))
    )

    lines = []
    lines.append(f"# Evaluation Report — {student_id}")
    lines.append(f"## Assignment: {assignment_id}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for skill in sorted_skills:
        lines.append(format_skill_feedback(skill))
        lines.append("")

    engine_logger.info(
        f"FORMATTER: Markdown report generated for student: {student_id}"
    )

    return "\n".join(lines)


def build_output(evaluation_result: dict) -> dict:
    """
    Builds the final output containing both JSON and Markdown.
    JSON is the internal truth — Markdown is derived from it.
    Returns dict with both formats ready for API response and file storage.
    """
    if not isinstance(evaluation_result, dict):
        engine_logger.error("FORMATTER: Cannot build output from non-dict evaluation result.")
        raise ValueError("evaluation_result must be a dictionary.")

    markdown = format_report(evaluation_result)

    engine_logger.info(
        f"FORMATTER: Output built for student: "
        f"{evaluation_result.get('student_id', 'unknown')}"
    )

    return {
        "json": evaluation_result,
        "markdown": markdown
    }
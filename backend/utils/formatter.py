"""
backend/utils/formatter.py
Converts internal JSON evaluation results into human-readable Markdown reports.
JSON is the internal truth — Markdown is the display layer.
Never writes Markdown directly — always derived from JSON.
"""

from backend.config.constants import (
    DEFAULT_FAIL_FEEDBACK,
    DEFAULT_LINE_REFERENCE,
    DEFAULT_MISSING_RECOMMENDED_FIX,
    DEFAULT_MISSING_STUDENT_CODE,
    DEFAULT_PASS_FEEDBACK,
    DEFAULT_UNKNOWN_ASSIGNMENT_ID,
    DEFAULT_UNKNOWN_SKILL,
    DEFAULT_UNKNOWN_STUDENT_ID,
    FAIL_SYMBOL,
    HIGH_WEIGHT_THRESHOLD,
    INVALID_EVALUATION_REPORT_MARKDOWN,
    INVALID_SKILL_ENTRY_MARKDOWN,
    MARKDOWN_CODE_BLOCK_LANGUAGE,
    NO_SKILLS_TO_DISPLAY_MESSAGE,
    OUTPUT_FORMAT_JSON,
    OUTPUT_FORMAT_MARKDOWN,
    PASS_SYMBOL,
    STATUS_FAIL,
    STATUS_PASS,
    STRUCTURE_HIGH_WEIGHT,
    STRUCTURE_LOW_WEIGHT,
)
from backend.utils.logger import engine_logger


def _safe_int(value, default: int) -> int:
    """
    Safely converts a value to int.
    Returns the provided default if conversion fails.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value, default: str) -> str:
    """
    Safely normalizes text values.
    Returns the default for None or blank strings.
    """
    if value is None:
        return default

    normalized_value = str(value).strip()
    if not normalized_value:
        return default

    return normalized_value


def _build_line_reference(line_start, line_end) -> str:
    """
    Builds a human-readable line reference safely.
    """
    normalized_line_start = _safe_int(line_start, -1)
    normalized_line_end = _safe_int(line_end, -1)

    if normalized_line_start > 0 and normalized_line_end > 0:
        return f"Lines {normalized_line_start}–{normalized_line_end}"

    return DEFAULT_LINE_REFERENCE


def _normalize_skill_status(status: str) -> str:
    """
    Normalizes skill status to PASS or FAIL.
    Unknown values default to FAIL for safety.
    """
    normalized_status = _safe_text(status, STATUS_FAIL).upper()

    if normalized_status == STATUS_PASS:
        return STATUS_PASS

    return STATUS_FAIL


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
        return INVALID_SKILL_ENTRY_MARKDOWN

    skill_text = _safe_text(skill.get("skill"), DEFAULT_UNKNOWN_SKILL)
    status = _normalize_skill_status(skill.get("status", STATUS_FAIL))
    weight = _safe_int(skill.get("weight", 1), 1)
    rank = _safe_int(skill.get("rank", 0), 0)
    line_start = skill.get("line_start")
    line_end = skill.get("line_end")
    snippet = _safe_text(skill.get("student_snippet"), "")
    fix = _safe_text(skill.get("recommended_fix"), "")
    feedback = _safe_text(skill.get("feedback"), "")

    symbol = PASS_SYMBOL if status == STATUS_PASS else FAIL_SYMBOL
    line_ref = _build_line_reference(line_start, line_end)

    lines = [
        f"### {symbol} Skill {rank} — {skill_text}",
        f"**Weight: {weight} | Status: {status} | {line_ref}**",
        "",
    ]

    if status == STATUS_PASS:
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
            lines.append(f"**{section}:**")

            if section == "WHAT YOU DID":
                if snippet:
                    lines.append(f"```{MARKDOWN_CODE_BLOCK_LANGUAGE}")
                    lines.append(snippet)
                    lines.append("```")
                else:
                    lines.append(DEFAULT_MISSING_STUDENT_CODE)

            elif section == "WHY IT IS WRONG":
                lines.append(feedback if feedback else DEFAULT_FAIL_FEEDBACK)

            elif section == "HOW TO FIX IT":
                if fix:
                    lines.append(f"```{MARKDOWN_CODE_BLOCK_LANGUAGE}")
                    lines.append(fix)
                    lines.append("```")
                else:
                    lines.append(DEFAULT_MISSING_RECOMMENDED_FIX)

            else:
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
        return INVALID_EVALUATION_REPORT_MARKDOWN

    student_id = _safe_text(
        evaluation_result.get("student_id"),
        DEFAULT_UNKNOWN_STUDENT_ID,
    )
    assignment_id = _safe_text(
        evaluation_result.get("assignment_id"),
        DEFAULT_UNKNOWN_ASSIGNMENT_ID,
    )
    skills = evaluation_result.get("skills", [])

    if not isinstance(skills, list) or not skills:
        engine_logger.warning(f"FORMATTER: No skills found for {student_id}")
        return (
            f"# Evaluation Report — {student_id}\n"
            f"## Assignment: {assignment_id}\n"
            f"{NO_SKILLS_TO_DISPLAY_MESSAGE}"
        )

    sorted_skills = sorted(
        skills,
        key=lambda skill: (
            -_safe_int(skill.get("weight", 0), 0),
            _safe_int(skill.get("rank", 999), 999),
        ),
    )

    lines = [
        f"# Evaluation Report — {student_id}",
        f"## Assignment: {assignment_id}",
        "",
        "---",
        "",
    ]

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
        engine_logger.error(
            "FORMATTER: Cannot build output from non-dict evaluation result."
        )
        raise ValueError("evaluation_result must be a dictionary.")

    markdown = format_report(evaluation_result)

    engine_logger.info(
        "FORMATTER: Output built for student: "
        f"{evaluation_result.get('student_id', DEFAULT_UNKNOWN_STUDENT_ID)}"
    )

    return {
        OUTPUT_FORMAT_JSON: evaluation_result,
        OUTPUT_FORMAT_MARKDOWN: markdown,
    }
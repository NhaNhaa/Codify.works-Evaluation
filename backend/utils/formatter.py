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


# ── SKILL STATUS SYMBOLS ───────────────────────────────────────────
PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"


def format_skill_feedback(skill: dict) -> str:
    """
    Converts a single skill verdict dict into a Markdown block.
    Applies weight-based depth:
    - PASS        → medium depth (confirm + explain why correct)
    - FAIL high   → detailed (what + why + how to fix)
    - FAIL low    → brief (what + how to fix)
    """
    skill_text  = skill.get("skill", "Unknown Skill")
    status      = skill.get("status", "FAIL")
    weight      = skill.get("weight", 1)
    rank        = skill.get("rank", 0)
    line_start  = skill.get("line_start")
    line_end    = skill.get("line_end")
    snippet     = skill.get("student_snippet", "")
    fix         = skill.get("recommended_fix", "")
    feedback    = skill.get("feedback", "")

    symbol      = PASS_SYMBOL if status == "PASS" else FAIL_SYMBOL
    line_ref    = f"Lines {line_start}–{line_end}" if line_start and line_end else "Line reference unavailable"

    lines = []

    # ── SKILL HEADER ───────────────────────────────────────────────
    lines.append(f"### {symbol} Skill {rank} — {skill_text}")
    lines.append(f"**Weight: {weight} | Status: {status} | {line_ref}**")
    lines.append("")

    if status == "PASS":
        # ── PASS: Medium depth ─────────────────────────────────────
        lines.append(feedback if feedback else "Well done! Your implementation is correct.")
        lines.append("")
        if snippet:
            lines.append("**Your correct code:**")
            lines.append("```c")
            lines.append(snippet)
            lines.append("```")

    else:
        # ── FAIL: Weight-based depth ───────────────────────────────
        if weight >= HIGH_WEIGHT_THRESHOLD:
            # Detailed — WHAT + WHY + HOW TO FIX
            structure = STRUCTURE_HIGH_WEIGHT
        else:
            # Brief — WHAT + HOW TO FIX
            structure = STRUCTURE_LOW_WEIGHT

        for section in structure:
            lines.append(f"**{section}:**")
            if section == "WHAT YOU DID" and snippet:
                lines.append("```c")
                lines.append(snippet)
                lines.append("```")
            elif section == "WHY IT IS WRONG":
                lines.append(feedback if feedback else "This implementation does not satisfy the micro skill requirement.")
            elif section == "HOW TO FIX IT" and fix:
                lines.append("```c")
                lines.append(fix)
                lines.append("```")
            lines.append("")

    lines.append("---")
    return "\n".join(lines)


def format_report(evaluation_result: dict) -> str:
    """
    Converts the full evaluation JSON into a complete Markdown report.
    Skills are sorted by weight — highest first (most critical first).
    No overall summary — per skill feedback only.
    """
    student_id    = evaluation_result.get("student_id", "unknown")
    assignment_id = evaluation_result.get("assignment_id", "unknown")
    skills        = evaluation_result.get("skills", [])

    if not skills:
        engine_logger.warning(f"FORMATTER: No skills found for {student_id}")
        return f"# Evaluation Report — {student_id}\nNo skills to display."

    # Sort by weight descending — highest weight first
    sorted_skills = sorted(skills, key=lambda s: s.get("weight", 0), reverse=True)

    lines = []

    # ── REPORT HEADER ──────────────────────────────────────────────
    lines.append(f"# Evaluation Report — {student_id}")
    lines.append(f"## Assignment: {assignment_id}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── PER SKILL FEEDBACK ─────────────────────────────────────────
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
    markdown = format_report(evaluation_result)

    engine_logger.info(
        f"FORMATTER: Output built for student: "
        f"{evaluation_result.get('student_id', 'unknown')}"
    )

    return {
        "json":     evaluation_result,
        "markdown": markdown
    }
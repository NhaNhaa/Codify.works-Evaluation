"""
backend/utils/skill_parser.py
Pure Python skill extractor — no LLM, no API calls.
Reads instructions.md and extracts micro skills exactly as written.

TIER 0 — File empty or too short          → returns None (abort signal)
TIER 1 — Labeled section found            → extracts bullets exactly
TIER 2 — No label, but list found         → extracts list items exactly
TIER 3 — Pure prose, no list              → returns [] (LLM fallback signal)

Called by Agent 1 before any LLM interaction.
Guarantees exact skill text when a list exists.
"""

import re

from backend.config.constants import (
    MIN_INSTRUCTION_CONTENT_LENGTH,
    SKILL_SECTION_KEYWORDS,
)
from backend.utils.logger import engine_logger

BULLET_LIST_PATTERN = r"^[-*]\s+(.+)$"
NUMBERED_LIST_PATTERN = r"^\d+[.)]\s+(.+)$"


def extract_skills(instructions: str) -> list[str] | None:
    """
    Main entry point. Called by Agent 1 Step 2.

    Returns:
        list[str]  — skills extracted exactly (Tier 1 or 2)
        []         — no list found, LLM fallback needed (Tier 3)
        None       — file invalid, abort Phase 1 (Tier 0)

    Agent 1 interprets:
        None  → abort, return False
        []    → fall through to LLM generation
        [...] → use these exact texts, skip LLM generation
    """
    if not _has_content(instructions):
        engine_logger.error(
            "SKILL PARSER: instructions.md is empty or too short to parse."
        )
        return None

    tier1_skills = _extract_from_labeled_section(instructions)
    if tier1_skills is not None:
        if tier1_skills:
            engine_logger.info(
                f"SKILL PARSER: Tier 1 — extracted {len(tier1_skills)} skills "
                "from labeled section. ✅"
            )
            return tier1_skills

        engine_logger.warning(
            "SKILL PARSER: Tier 1 — labeled section found but empty. "
            "Falling through to Tier 2."
        )

    tier2_skills = _extract_from_any_list(instructions)
    if tier2_skills:
        engine_logger.info(
            f"SKILL PARSER: Tier 2 — extracted {len(tier2_skills)} skills "
            "from unlabeled list. ✅"
        )
        return tier2_skills

    engine_logger.info(
        "SKILL PARSER: Tier 3 — no list found. "
        "Returning [] to trigger LLM generation."
    )
    return []


def _has_content(text: str) -> bool:
    """
    Returns True if the file has enough content to parse.
    Rejects empty files, whitespace-only files, and very short files.
    """
    if not isinstance(text, str):
        return False

    stripped_text = text.strip()
    if not stripped_text:
        return False

    return len(stripped_text) >= MIN_INSTRUCTION_CONTENT_LENGTH


def _extract_list_item_text(stripped_line: str) -> str | None:
    """
    Extracts list item text from a bullet or numbered line.
    Returns None if the line is not a supported list item.
    """
    bullet_match = re.match(BULLET_LIST_PATTERN, stripped_line)
    if bullet_match:
        item = bullet_match.group(1).strip()
        return item if item else None

    numbered_match = re.match(NUMBERED_LIST_PATTERN, stripped_line)
    if numbered_match:
        item = numbered_match.group(1).strip()
        return item if item else None

    return None


def _extract_from_labeled_section(text: str) -> list[str] | None:
    """
    Looks for a known section header and extracts all bullet or numbered
    items under it.

    Returns:
        list[str]  — items found under the labeled section
        []         — labeled section found but contains no list items
        None       — no matching labeled section found
    """
    lines = text.splitlines()

    section_start_index = None
    for index, line in enumerate(lines):
        normalized_header = line.strip().lstrip("#").strip().lower()

        for keyword in SKILL_SECTION_KEYWORDS:
            if normalized_header == keyword or normalized_header.startswith(keyword):
                section_start_index = index
                engine_logger.info(
                    f"SKILL PARSER: Found labeled section at line {index + 1}: "
                    f"'{line.strip()}'"
                )
                break

        if section_start_index is not None:
            break

    if section_start_index is None:
        return None

    items: list[str] = []

    for line in lines[section_start_index + 1:]:
        stripped_line = line.strip()

        if stripped_line.startswith("#"):
            break

        item = _extract_list_item_text(stripped_line)
        if item:
            items.append(item)

    return items


def _extract_from_any_list(text: str) -> list[str]:
    """
    Scans the entire document for any bullet or numbered list.
    Returns the largest contiguous list block found.
    Skips single-item lists because they are likely not a skill list.

    Returns:
        list[str]  — items from the largest list found
        []         — no list found
    """
    lines = text.splitlines()

    list_blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in lines:
        stripped_line = line.strip()
        item = _extract_list_item_text(stripped_line)

        if item:
            current_block.append(item)
            continue

        if current_block:
            list_blocks.append(current_block)
            current_block = []

    if current_block:
        list_blocks.append(current_block)

    valid_blocks = [block for block in list_blocks if len(block) > 1]
    if not valid_blocks:
        return []

    return max(valid_blocks, key=len)


def has_explicit_skills(instructions: str) -> bool:
    """
    Quick check — returns True if instructions contain an extractable list.
    Used by Agent 1 to decide whether to log extraction source.
    """
    extracted_skills = extract_skills(instructions)
    return extracted_skills is not None and len(extracted_skills) > 0
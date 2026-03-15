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
from backend.utils.logger import engine_logger


# ── KNOWN SECTION HEADER KEYWORDS ─────────────────────────────────
# Matches any ## or # heading containing these keywords (case-insensitive)
SKILL_SECTION_KEYWORDS = [
    "micro skills",
    "micro skills to evaluate",
    "skills to evaluate",
    "learning objectives",
    "skills",
    "objectives",
    "evaluation criteria",
    "grading criteria",
]

# Minimum character count to consider a file non-empty
MIN_CONTENT_LENGTH = 30


# ── MAIN ENTRY POINT ───────────────────────────────────────────────
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
        [...]  → use these exact texts, skip LLM generation
    """
    # TIER 0 — Validate file has usable content
    if not _has_content(instructions):
        engine_logger.error(
            "SKILL PARSER: instructions.md is empty or too short to parse."
        )
        return None

    # TIER 1 — Look for labeled section
    tier1 = _extract_from_labeled_section(instructions)
    if tier1 is not None:
        if len(tier1) > 0:
            engine_logger.info(
                f"SKILL PARSER: Tier 1 — extracted {len(tier1)} skills "
                f"from labeled section. ✅"
            )
            return tier1
        else:
            engine_logger.warning(
                "SKILL PARSER: Tier 1 — labeled section found but empty. "
                "Falling through to Tier 2."
            )

    # TIER 2 — Look for any bullet or numbered list
    tier2 = _extract_from_any_list(instructions)
    if tier2:
        engine_logger.info(
            f"SKILL PARSER: Tier 2 — extracted {len(tier2)} skills "
            f"from unlabeled list. ✅"
        )
        return tier2

    # TIER 3 — Pure prose, no list found — signal LLM fallback
    engine_logger.info(
        "SKILL PARSER: Tier 3 — no list found. "
        "Returning [] to trigger LLM generation."
    )
    return []


# ── TIER 0: CONTENT VALIDATION ─────────────────────────────────────
def _has_content(text: str) -> bool:
    """
    Returns True if the file has enough content to parse.
    Rejects empty files, whitespace-only files, and very short files.
    """
    if not text or not text.strip():
        return False
    return len(text.strip()) >= MIN_CONTENT_LENGTH


# ── TIER 1: LABELED SECTION EXTRACTION ────────────────────────────
def _extract_from_labeled_section(text: str) -> list[str] | None:
    """
    Looks for a known section header (## MICRO SKILLS TO EVALUATE, etc.)
    and extracts all bullet or numbered items under it.

    Returns:
        list[str]  — items found under the labeled section
        []         — labeled section found but contains no list items
        None       — no matching labeled section found
    """
    lines = text.splitlines()

    section_start = None
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip().lower()
        for keyword in SKILL_SECTION_KEYWORDS:
            if stripped == keyword or stripped.startswith(keyword):
                section_start = i
                engine_logger.info(
                    f"SKILL PARSER: Found labeled section at line {i + 1}: "
                    f"'{line.strip()}'"
                )
                break
        if section_start is not None:
            break

    if section_start is None:
        return None

    # Extract items from this section until next heading or end of file
    items = []
    for line in lines[section_start + 1:]:
        stripped = line.strip()

        # Stop at next section heading
        if stripped.startswith("#"):
            break

        # Match bullet: "- item", "* item"
        bullet_match = re.match(r'^[-*]\s+(.+)$', stripped)
        if bullet_match:
            item = bullet_match.group(1).strip()
            if item:
                items.append(item)
            continue

        # Match numbered: "1. item", "1) item"
        numbered_match = re.match(r'^\d+[.)]\s+(.+)$', stripped)
        if numbered_match:
            item = numbered_match.group(1).strip()
            if item:
                items.append(item)

    return items


# ── TIER 2: ANY LIST EXTRACTION ────────────────────────────────────
def _extract_from_any_list(text: str) -> list[str]:
    """
    Scans the entire document for any bullet or numbered list.
    Returns the LARGEST contiguous list block found.
    Skips single-item lists (likely not a skill list).

    Returns:
        list[str]  — items from the largest list found
        []         — no list found
    """
    lines = text.splitlines()

    # Find all contiguous list blocks
    blocks     = []
    current    = []
    in_list    = False

    for line in lines:
        stripped = line.strip()

        is_bullet   = bool(re.match(r'^[-*]\s+(.+)$',    stripped))
        is_numbered = bool(re.match(r'^\d+[.)]\s+(.+)$', stripped))

        if is_bullet or is_numbered:
            in_list = True
            # Extract the item text
            match = re.match(r'^[-*\d][.)]\s+(.+)$', stripped) or \
                    re.match(r'^[-*]\s+(.+)$', stripped) or \
                    re.match(r'^\d+[.)]\s+(.+)$', stripped)
            if match:
                current.append(match.group(1).strip())
        else:
            if in_list and current:
                blocks.append(list(current))
                current = []
            in_list = False

    # Capture last block if file ends mid-list
    if current:
        blocks.append(current)

    if not blocks:
        return []

    # Return the largest block — most likely the skill list
    # Ignore single-item blocks (probably not a skill list)
    valid_blocks = [b for b in blocks if len(b) > 1]
    if not valid_blocks:
        return []

    largest = max(valid_blocks, key=len)
    return largest


# ── CONVENIENCE: CHECK IF SKILLS EXIST ────────────────────────────
def has_explicit_skills(instructions: str) -> bool:
    """
    Quick check — returns True if instructions contain an extractable list.
    Used by Agent 1 to decide whether to log extraction source.
    """
    result = extract_skills(instructions)
    return result is not None and len(result) > 0
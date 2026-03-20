"""Deduplication, ranking, weighting, snippet helpers."""

from .validation import (
    _normalize_whitespace,
    _normalize_skill_text,
    _contains_output_keyword,
    _get_io_subtype,
    VALID_SKILL_VERBS,
    SUBJECT_SYNONYM_GROUPS,
    OPERATION_CONTEXT_KEYWORDS,
    SHIFT_KEYWORDS,
    TEMP_SAFETY_KEYWORDS,
    OUTPUT_SKILL_KEYWORDS,
    LIST_ITEM_PATTERN,
    LINE_RANGE_PREFIX_PATTERN,
    SIMPLE_LINE_PREFIX_PATTERN,
    EXAMPLE_LINE_KEYWORDS,
    engine_logger,
)
from backend.config.constants import (
    DEFAULT_WEIGHT_DISTRIBUTIONS,
    MAX_MICRO_SKILLS,
    MIN_MICRO_SKILLS,
    TOTAL_WEIGHT_TARGET,
)
import re


# ── INTERNAL HELPERS ───────────────────────────────────────────────
def _count_non_empty_string_skills(skills: list[str]) -> int:
    """Counts non‑empty string skills safely."""
    if not isinstance(skills, list):
        return 0
    return sum(1 for s in skills if isinstance(s, str) and s.strip())


def _remove_action_verb(text: str) -> str:
    """Removes the leading action verb."""
    text_lower = text.lower()
    for verb in VALID_SKILL_VERBS:
        verb_lower = verb.lower()
        if text_lower.startswith(verb_lower):
            return text[len(verb):].lstrip(" ,")
    return text


def _shares_subject_and_operation(text1: str, text2: str) -> bool:
    """Checks for shared subject and operation."""
    shared_operation = any(
        kw in text1 and kw in text2 for kw in OPERATION_CONTEXT_KEYWORDS
    )
    if not shared_operation:
        return False
    for group in SUBJECT_SYNONYM_GROUPS:
        if any(term in text1 for term in group) and any(term in text2 for term in group):
            return True
    return False


def _get_concept_bucket(text: str) -> str | None:
    """Classifies a skill into a concept bucket."""
    lower = text.lower()
    if any(kw in lower for kw in TEMP_SAFETY_KEYWORDS):
        return "data_safety"
    io_kws = ["scanf", "printf", "read input", "print output", "display", "separator", "comma"]
    if any(kw in lower for kw in io_kws):
        return "io"
    transform_kws = [
        "shift", "sort", "reverse", "rotate", "swap", "merge",
        "insert", "delete", "remove", "search", "boundary",
        "wrap", "wrapping", "first element", "last position"
    ]
    if any(kw in lower for kw in transform_kws):
        return "transform"
    access_kws = ["arr[i]", "array elements", "access", "index"]
    if any(kw in lower for kw in access_kws):
        return "array_access"
    return None


def _bucket_specific_overlap(text1: str, text2: str, bucket: str) -> bool:
    """Bucket‑aware overlap rules."""
    if bucket == "data_safety":
        shared_safety = any(
            kw in text1 and kw in text2 for kw in TEMP_SAFETY_KEYWORDS
        )
        shared_op = any(
            kw in text1 and kw in text2 for kw in OPERATION_CONTEXT_KEYWORDS
        )
        return shared_safety and (
            shared_op or "temporary" in text1 or "temporary" in text2
            or "temp" in text1 or "temp" in text2
        )
    if bucket == "io":
        sub1 = _get_io_subtype(text1)
        sub2 = _get_io_subtype(text2)
        if sub1 != sub2 or sub1 is None:
            return False
        if sub1 == "input":
            return ("scanf" in text1 and "scanf" in text2) or _meaningful_containment(text1, text2)
        if sub1 == "output":
            shared_out = sum(1 for kw in OUTPUT_SKILL_KEYWORDS if kw in text1 and kw in text2)
            has_format = any(
                kw in text1 or kw in text2
                for kw in ["comma", "separator", "between elements", "trailing comma", "format"]
            )
            return (shared_out >= 1 and has_format) or _meaningful_containment(text1, text2)
        return False
    if bucket == "array_access":
        access_kws = ["arr[i]", "index", "access", "array elements"]
        shared_access = any(kw in text1 and kw in text2 for kw in access_kws)
        return shared_access and _meaningful_containment(text1, text2)
    if bucket == "transform":
        if _is_shift_boundary_split(text1, text2):
            return True
        shared_op = any(
            kw in text1 and kw in text2 for kw in OPERATION_CONTEXT_KEYWORDS
        )
        if not shared_op:
            return False
        if _shares_subject_and_operation(text1, text2):
            return True
        return _meaningful_containment(text1, text2)
    return False


def _meaningful_containment(text1: str, text2: str) -> bool:
    """Substring containment with minimum length guard."""
    shorter = min(text1, text2, key=len)
    longer = max(text1, text2, key=len)
    if len(shorter) < 12:
        return False
    return shorter in longer


def _references_first_boundary(text: str) -> bool:
    """True if text refers to the first‑element boundary case."""
    first_group = SUBJECT_SYNONYM_GROUPS[0]
    return any(term in text for term in first_group) or "last position" in text or "wrap" in text


def _is_shift_boundary_split(text1: str, text2: str) -> bool:
    """Detects split of shift + boundary into two skills."""
    t1_shift = any(kw in text1 for kw in SHIFT_KEYWORDS)
    t2_shift = any(kw in text2 for kw in SHIFT_KEYWORDS)
    t1_boundary = _references_first_boundary(text1)
    t2_boundary = _references_first_boundary(text2)
    if not (t1_boundary and t2_boundary):
        return False
    return t1_shift or t2_shift


# ── PUBLIC DEDUPLICATION ───────────────────────────────────────────
def deduplicate_skills(skills: list[str]) -> list[str]:
    """Removes overlapping concepts using is_same_concept."""
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: deduplicate_skills expected a list.")
        return []
    unique = []
    for raw_skill in skills:
        if not isinstance(raw_skill, str):
            continue
        skill = _normalize_whitespace(raw_skill)
        if not skill:
            continue
        duplicate = False
        for existing in unique:
            if is_same_concept(existing, skill):
                engine_logger.info(f"Dedup — dropped '{skill}' (overlaps with: '{existing}')")
                duplicate = True
                break
        if not duplicate:
            unique.append(skill)
    removed = _count_non_empty_string_skills(skills) - len(unique)
    if removed:
        engine_logger.info(f"Dedup — {removed} duplicate(s) removed.")
    else:
        engine_logger.info("Dedup — no duplicates found.")
    return unique


def is_same_concept(text1: str, text2: str) -> bool:
    """Three‑layer concept overlap detection."""
    norm1 = _normalize_skill_text(text1)
    norm2 = _normalize_skill_text(text2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True
    core1 = _remove_action_verb(norm1)
    core2 = _remove_action_verb(norm2)
    if core1 == core2:
        return True
    if _is_shift_boundary_split(core1, core2):
        engine_logger.info(f"Overlap detected — split shift/boundary: '{text1}' vs '{text2}'")
        return True
    if _shares_subject_and_operation(core1, core2):
        engine_logger.info(f"Overlap detected — shared subject+operation: '{text1}' vs '{text2}'")
        return True
    bucket1 = _get_concept_bucket(core1)
    bucket2 = _get_concept_bucket(core2)
    if bucket1 and bucket1 == bucket2:
        if _bucket_specific_overlap(core1, core2, bucket1):
            return True
    if _meaningful_containment(core1, core2):
        return True
    return False


# ── RANKING AND WEIGHTING ──────────────────────────────────────────
def enforce_ranking_rules(skills: list[str]) -> list[str]:
    """Pure Python ranking: transform → safety → middle → io."""
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: enforce_ranking_rules expected a list.")
        return []
    cleaned = [
        _normalize_whitespace(s)
        for s in skills if isinstance(s, str) and s.strip()
    ]
    if len(cleaned) < 2:
        return cleaned

    transform_kws = [
        "shift", "sort", "reverse", "search", "swap", "rotate",
        "merge", "insert", "delete", "remove", "boundary", "wrap",
        "wrapping", "first element", "last position"
    ]
    safety_kws = [
        "temp", "temporary", "preserve", "avoid losing",
        "avoid overwriting", "losing data", "data loss"
    ]
    io_kws = ["scanf", "printf", "read input", "print output"]

    def classify(s: str) -> str:
        low = s.lower()
        if any(kw in low for kw in safety_kws):
            return "safety"
        if any(kw in low for kw in io_kws):
            return "io"
        if any(kw in low for kw in transform_kws):
            return "transform"
        return "middle"

    buckets = {"transform": [], "safety": [], "middle": [], "io": []}
    for s in cleaned:
        buckets[classify(s)].append(s)
    result = buckets["transform"] + buckets["safety"] + buckets["middle"] + buckets["io"]
    engine_logger.info(
        f"Ranking — transform={len(buckets['transform'])}, "
        f"safety={len(buckets['safety'])}, middle={len(buckets['middle'])}, io={len(buckets['io'])}"
    )
    return result


def python_rank_and_weight(skills: list[str]) -> list[dict]:
    """100% Python ranking and weight assignment."""
    ranked = enforce_ranking_rules(skills)[:MAX_MICRO_SKILLS]
    count = len(ranked)
    if count < MIN_MICRO_SKILLS:
        engine_logger.error(f"Only {count} skills after ranking (min {MIN_MICRO_SKILLS}). Aborting.")
        return []
    if count not in DEFAULT_WEIGHT_DISTRIBUTIONS:
        engine_logger.error(f"No weight distribution for {count} skills.")
        return []
    weights = DEFAULT_WEIGHT_DISTRIBUTIONS[count]
    final = [
        {"text": s, "rank": i + 1, "weight": weights[i]}
        for i, s in enumerate(ranked)
    ]
    total = sum(s["weight"] for s in final)
    if total != TOTAL_WEIGHT_TARGET:
        engine_logger.error(f"Weight sum mismatch: {total} != {TOTAL_WEIGHT_TARGET}")
        return []
    engine_logger.info(f"{count} skills ranked and weighted. Sum = {total}.")
    return final


# ── SNIPPET HELPERS ────────────────────────────────────────────────
def strip_line_prefixes(snippet: str) -> str:
    """Removes line‑number prefixes while preserving indentation."""
    if not isinstance(snippet, str):
        return ""
    cleaned = []
    for line in snippet.splitlines():
        range_match = LINE_RANGE_PREFIX_PATTERN.match(line)
        if range_match:
            cleaned.append(range_match.group(1))
            continue
        numbered_match = SIMPLE_LINE_PREFIX_PATTERN.match(line)
        if numbered_match:
            cleaned.append(numbered_match.group(1))
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def number_lines(code: str) -> str:
    """Prepends line numbers to code."""
    if not isinstance(code, str) or not code:
        return ""
    lines = code.splitlines()
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))


def clean_json(content: str) -> str:
    """Strips markdown fences from JSON responses."""
    if not isinstance(content, str):
        return ""
    cleaned = content.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()
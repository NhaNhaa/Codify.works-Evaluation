"""
backend/agents/agent1_validators.py
Pure Python validation, ranking, deduplication, and generic rejection
for Agent 1 micro skills. Zero LLM dependency.

Handles:
- Skill formula validation and auto-fix
- Three-layer concept deduplication
- Generic language construct rejection
- Deterministic keyword-based ranking
- Weight assignment from distribution table
- Snippet and JSON cleaning helpers

All functions are standalone — imported by agent1_extractor.py.
Agent 2 and Agent 3 never import this module (agent isolation).
"""

import re
from backend.config.constants import (
    MIN_MICRO_SKILLS,
    MAX_MICRO_SKILLS,
    TOTAL_WEIGHT_TARGET,
    DEFAULT_WEIGHT_DISTRIBUTIONS,
    SKILL_QUALITY_RULES
)
from backend.utils.logger import engine_logger


# ── SKILL FORMULA CONSTANTS ────────────────────────────────────────
VALID_SKILL_VERBS = ["Be able to", "Use", "Avoid", "Handle", "Implement"]
MIN_SKILL_WORDS = 6
MAX_SKILL_WORDS = 15

# ── SUBJECT SYNONYM GROUPS (cross-bucket overlap detection) ────────
# Two skills referencing the same specific data element in the same
# operation context are likely overlapping — even if in different buckets.
SUBJECT_SYNONYM_GROUPS = [
    {"first element", "1st cell", "first cell", "1st element",
     "first position", "arr[0]", "first value"},
    {"last element", "last cell", "last position", "last value",
     "arr[n-1]"},
]

# ── OPERATION CONTEXT KEYWORDS ─────────────────────────────────────
# Used by Layer 2 of is_same_concept() to detect shared operation context.
OPERATION_CONTEXT_KEYWORDS = [
    "shift", "sort", "reverse", "rotate", "swap",
    "merge", "insert", "delete", "remove"
]

# ── GENERIC CONSTRUCT PATTERNS ─────────────────────────────────────
# Skills matching these patterns WITHOUT specificity keywords are rejected.
# Enforces SKILL_QUALITY_RULES #3 and #4 via Python — final safety net.
GENERIC_CONSTRUCT_PATTERNS = [
    "use a loop",
    "use for loop",
    "use while loop",
    "use do while",
    "iterate through",
    "iterate over",
    "loop through",
    "loop to iterate",
    "traverse array",
    "traverse the array",
    "declare variable",
    "define variable",
    "initialize variable",
    "use if statement",
    "use conditional",
    "use switch statement",
    "return value from",
    "call a function",
    "use function to",
    "print output",
    "display output",
    "display result",
    "print result",
    "output the result",
    "print the array",
    "display the array",
]

# ── SPECIFICITY KEYWORDS ───────────────────────────────────────────
# If a skill matches a generic pattern BUT also contains any of these,
# it is considered assignment-specific and NOT rejected.
SPECIFICITY_KEYWORDS = [
    # Transform operations
    "shift", "sort", "reverse", "rotate", "swap", "merge",
    "insert", "delete", "remove", "search",
    # Data safety
    "temp", "temporary", "preserve", "avoid losing",
    "losing data", "data loss", "avoid overwriting",
    # Specific C syntax
    "arr[", "arr [", "scanf", "&arr",
    # Specific details
    "boundary", "wrap", "first element", "1st cell",
    "last position", "first position",
    # Specific quantities
    "integers into", "characters into", "strings into",
]

# ── SKILL FORMULA PROMPT ──────────────────────────────────────────
# Injected into LLM prompts by agent1_extractor.py.
# Defined here as the single source of truth for skill format rules.
SKILL_FORMULA_PROMPT = """
Every skill MUST follow this exact formula:
[Action Verb] + [What Concept] + [How Specifically in C] + [Edge Case if any]

Valid action verbs: Be able to, Use, Avoid, Handle, Implement

GOOD: "Be able to shift elements, using arr[i-1] and handle the 1st cell"
GOOD: "Use scanf to read 5 integers into an array"
GOOD: "Avoid losing data during shifting, using a temporary variable"
GOOD: "Handle boundary case when first element wraps to last position"
BAD:  "Shift array elements" (too vague, missing C detail, missing verb)
BAD:  "Use loops" (too generic, not assignment-specific)
BAD:  "temp variable" (not a sentence, no verb, no context)

CRITICAL — DO NOT SPLIT CONCEPTS:
Each skill must capture ONE complete concept including its edge cases.
DO NOT create separate skills for different parts of the same operation.

BAD SPLITTING EXAMPLES (never do this):
- "Shift elements using arr[i-1]" AND "Handle boundary when first element wraps"
  → These are ONE skill: "Be able to shift elements, using arr[i-1] and handle the 1st cell"
- "Use temporary variable" AND "Avoid overwriting data during shift"
  → These are ONE skill: "Avoid losing data during shifting, using a temporary variable"
- "Use arr[i-1] for left access" AND "Use arr[i+1] for right access"
  → These describe the SAME shift operation — pick one, not both
- "Shift elements and handle the 1st cell" AND "Store first element using temp variable"
  → These OVERLAP because both reference "first element" in the shifting context.
  → Keep the transform skill as-is, and make the safety skill GENERIC:
    GOOD safety: "Avoid losing data during shifting, using a temporary variable"
    BAD safety:  "Use temporary variable to store first element before shifting"

RULE: If two skills would evaluate the same lines of student code, they must be ONE skill.

SAFETY SKILL RULE: Safety/preservation skills (temp variable, avoid data loss) must
describe the PATTERN generically — do NOT reference the same specific data element
(e.g., "first element", "1st cell") that a transform skill already handles.

GENERIC SKILL REJECTION:
Do NOT generate skills about basic language constructs that every C program uses.
These are NOT valid micro skills unless they are the specific focus of the assignment:
BAD: "Use a loop to iterate through array elements" (every C program uses loops)
BAD: "Declare and initialize variables" (every C program does this)
BAD: "Use printf to display output" (generic output, not a learning objective)
BAD: "Use a loop to traverse the array" (generic loop usage)
GOOD: "Use scanf to read 5 integers into an array" (specific to this assignment)
GOOD: "Be able to access array elements using index (arr[i])" (specific C syntax)
"""


# ═══════════════════════════════════════════════════════════════════
# SKILL FORMULA VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_and_fix_skills(skills: list[str]) -> list[str]:
    """
    Validates each LLM-generated skill against the formula.
    Auto-fixes recoverable issues. Drops unrecoverable ones.
    Only called for LLM-generated skills — never for parser-extracted.
    """
    validated = []
    for skill in skills:
        if is_valid_skill_format(skill):
            validated.append(skill)
            engine_logger.info(
                f"AGENT 1: Skill format valid: '{skill}' ✅"
            )
        else:
            fixed = auto_fix_skill_format(skill)
            if fixed:
                validated.append(fixed)
                engine_logger.info(
                    f"AGENT 1: Auto-fixed skill: '{skill}' → '{fixed}' ✅"
                )
            else:
                engine_logger.warning(
                    f"AGENT 1: Dropped invalid skill (unfixable): '{skill}'"
                )
    return validated


def is_valid_skill_format(skill: str) -> bool:
    """
    Checks if a skill matches the required formula.
    1. Starts with a valid action verb
    2. Between MIN_SKILL_WORDS and MAX_SKILL_WORDS words
    """
    has_valid_verb = any(
        skill.lower().startswith(verb.lower())
        for verb in VALID_SKILL_VERBS
    )
    if not has_valid_verb:
        return False

    word_count = len(skill.split())
    if word_count < MIN_SKILL_WORDS or word_count > MAX_SKILL_WORDS:
        return False

    return True


def auto_fix_skill_format(skill: str) -> str | None:
    """
    Attempts to fix a skill that fails formula validation.
    Returns fixed skill text or None if unfixable.
    """
    original = skill.strip()
    if not original:
        return None

    # Fix 1: Prepend "Be able to" if no valid verb
    has_valid_verb = any(
        original.lower().startswith(verb.lower())
        for verb in VALID_SKILL_VERBS
    )
    if not has_valid_verb:
        fixed = f"Be able to {original[0].lower()}{original[1:]}"
    else:
        fixed = original

    # Fix 2: Check word count after fix
    words = fixed.split()
    if len(words) < MIN_SKILL_WORDS:
        return None  # Too short to be meaningful
    if len(words) > MAX_SKILL_WORDS:
        fixed = " ".join(words[:MAX_SKILL_WORDS])

    return fixed


# ═══════════════════════════════════════════════════════════════════
# THREE-LAYER DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════

def deduplicate_skills(skills: list[str]) -> list[str]:
    """
    Removes overlapping concepts within a skill list.
    Keeps the first occurrence, drops later duplicates.
    Uses is_same_concept() for comparison.
    Called after LLM generation to catch concept splitting.
    """
    unique = []
    for skill in skills:
        is_dup = False
        for existing in unique:
            if is_same_concept(existing, skill):
                engine_logger.info(
                    f"AGENT 1: Dedup — dropped '{skill}' "
                    f"(overlaps with: '{existing}')"
                )
                is_dup = True
                break
        if not is_dup:
            unique.append(skill)

    removed_count = len(skills) - len(unique)
    if removed_count > 0:
        engine_logger.info(
            f"AGENT 1: Dedup — {len(skills)} → {len(unique)} skills "
            f"({removed_count} duplicates removed). ✅"
        )
    else:
        engine_logger.info(
            f"AGENT 1: Dedup — {len(skills)} skills, "
            f"no duplicates found. ✅"
        )

    return unique


def is_same_concept(text1: str, text2: str) -> bool:
    """
    Three-layer concept overlap detection:

    Layer 1 — Bucket classification:
      If both skills classify into the same bucket (transform,
      safety, io, array_access), they overlap.

    Layer 2 — Shared subject + shared operation (cross-bucket):
      If both skills reference the same specific data element
      (e.g., "first element" / "1st cell") AND both are in the
      context of the same operation (e.g., "shift"), they overlap
      even if they are in different buckets.
      Catches: "shift + handle 1st cell" vs "temp + store first element"

    Layer 3 — Substring containment:
      If one skill text fully contains the other, they overlap.
    """
    lower1 = text1.lower().strip().rstrip(".,;:")
    lower2 = text2.lower().strip().rstrip(".,;:")

    # ── LAYER 1: Bucket classification ─────────────────────────
    c1 = _get_concept_bucket(text1)
    c2 = _get_concept_bucket(text2)

    if c1 is not None and c2 is not None and c1 == c2:
        return True

    # ── LAYER 2: Shared subject + shared operation context ─────
    for syn_group in SUBJECT_SYNONYM_GROUPS:
        match1 = any(syn in lower1 for syn in syn_group)
        match2 = any(syn in lower2 for syn in syn_group)
        if match1 and match2:
            shared_op = any(
                op in lower1 and op in lower2
                for op in OPERATION_CONTEXT_KEYWORDS
            )
            if shared_op:
                engine_logger.info(
                    f"AGENT 1: Layer 2 overlap detected — "
                    f"shared subject + shared operation: "
                    f"'{text1}' vs '{text2}'"
                )
                return True

    # ── LAYER 3: Substring containment ─────────────────────────
    if lower1 in lower2 or lower2 in lower1:
        return True

    return False


def _get_concept_bucket(text: str) -> str | None:
    """
    Classifies a skill text into a concept bucket.
    Used by Layer 1 of is_same_concept().
    """
    lower = text.lower()
    safety_kw = [
        "temp", "temporary", "preserve", "losing data",
        "avoid losing", "data loss", "overwrite",
        "avoid overwriting"
    ]
    if any(kw in lower for kw in safety_kw):
        return "data_safety"
    io_kw = ["scanf", "printf", "read input", "print output"]
    if any(kw in lower for kw in io_kw):
        return "io"
    transform_kw = [
        "shift", "sort", "reverse", "rotate", "swap",
        "merge", "insert", "delete", "remove", "search",
        "boundary", "wrap", "first element", "last position"
    ]
    if any(kw in lower for kw in transform_kw):
        return "transform"
    access_kw = ["arr[i]", "array elements", "access", "index"]
    if any(kw in lower for kw in access_kw):
        return "array_access"
    return None


# ═══════════════════════════════════════════════════════════════════
# GENERIC SKILL REJECTION
# ═══════════════════════════════════════════════════════════════════

def reject_generic_skills(skills: list[str]) -> list[str]:
    """
    Rejects skills that describe basic language constructs
    without assignment-specific detail.
    Enforces SKILL_QUALITY_RULES #3 and #4 via Python.
    Only applied to LLM-generated skills — never parser-extracted.
    This is the final safety net — deterministic, no LLM dependency.
    """
    non_generic = []
    for skill in skills:
        if is_generic_skill(skill):
            engine_logger.warning(
                f"AGENT 1: Rejected generic skill: '{skill}'"
            )
        else:
            non_generic.append(skill)

    rejected_count = len(skills) - len(non_generic)
    if rejected_count > 0:
        engine_logger.info(
            f"AGENT 1: Generic rejection — {len(skills)} → "
            f"{len(non_generic)} skills "
            f"({rejected_count} generic skills removed). ✅"
        )
    else:
        engine_logger.info(
            f"AGENT 1: Generic rejection — {len(skills)} skills, "
            f"none were generic. ✅"
        )

    return non_generic


def is_generic_skill(skill: str) -> bool:
    """
    Returns True if a skill describes a basic language construct
    without assignment-specific detail.

    A skill is GENERIC if:
    1. It matches any GENERIC_CONSTRUCT_PATTERNS AND
    2. It does NOT contain any SPECIFICITY_KEYWORDS

    Examples:
    GENERIC:     "Use a loop to iterate through array elements"
    NOT GENERIC: "Use scanf to read 5 integers into an array"
    NOT GENERIC: "Avoid losing data during shifting, using a temporary variable"
    """
    lower = skill.lower()

    matches_generic = any(
        pattern in lower
        for pattern in GENERIC_CONSTRUCT_PATTERNS
    )

    if not matches_generic:
        return False

    has_specificity = any(
        keyword in lower
        for keyword in SPECIFICITY_KEYWORDS
    )

    return not has_specificity


# ═══════════════════════════════════════════════════════════════════
# DETERMINISTIC RANKING + WEIGHT ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════

def enforce_ranking_rules(skills: list[str]) -> list[str]:
    """
    Pure Python ranking. No LLM.
    RULE 1: transform → first
    RULE 2: safety → second
    RULE 3: io → last
    RULE 4: everything else → middle
    Includes boundary/wrap keywords under transform for consistency
    with is_same_concept classification.
    """
    if len(skills) < 2:
        return list(skills)

    transform_kw = [
        "shift", "sort", "reverse", "search", "swap",
        "rotate", "merge", "insert", "delete", "remove",
        "boundary", "wrap", "first element", "last position"
    ]
    safety_kw = [
        "temp", "temporary", "preserve", "avoid losing",
        "avoid overwriting", "losing data", "data loss"
    ]
    io_kw = ["scanf", "printf", "read input", "print output"]

    def classify(skill_text: str) -> str:
        lower = skill_text.lower()
        if any(k in lower for k in safety_kw):
            return "safety"
        if any(k in lower for k in io_kw):
            return "io"
        if any(k in lower for k in transform_kw):
            return "transform"
        return "middle"

    buckets = {"transform": [], "safety": [], "middle": [], "io": []}
    for skill in skills:
        buckets[classify(skill)].append(skill)

    result = (
        buckets["transform"]
        + buckets["safety"]
        + buckets["middle"]
        + buckets["io"]
    )

    engine_logger.info(
        f"AGENT 1: Ranking — "
        f"transform={len(buckets['transform'])}, "
        f"safety={len(buckets['safety'])}, "
        f"middle={len(buckets['middle'])}, "
        f"io={len(buckets['io'])}"
    )
    return result


def python_rank_and_weight(skills: list[str]) -> list[dict]:
    """
    100% Python — no LLM call.
    1. Classify each skill into bucket (transform/safety/middle/io)
    2. Order: transform → safety → middle → io
    3. Trim to MAX_MICRO_SKILLS
    4. Validate minimum count
    5. Assign weights from distribution table
    6. Validate sum = 10
    """
    ranked = enforce_ranking_rules(skills)

    # Trim to MAX
    ranked = ranked[:MAX_MICRO_SKILLS]
    count = len(ranked)

    # Validate minimum
    if count < MIN_MICRO_SKILLS:
        engine_logger.error(
            f"AGENT 1: Only {count} skills available after ranking. "
            f"Minimum required is {MIN_MICRO_SKILLS}. Aborting."
        )
        return []

    # Get weight distribution
    if count not in DEFAULT_WEIGHT_DISTRIBUTIONS:
        engine_logger.error(
            f"AGENT 1: No weight distribution defined for {count} skills."
        )
        return []

    weights = DEFAULT_WEIGHT_DISTRIBUTIONS[count]

    final_skills = [
        {
            "text":   skill_text,
            "rank":   i + 1,
            "weight": weights[i]
        }
        for i, skill_text in enumerate(ranked)
    ]

    total = sum(s["weight"] for s in final_skills)
    if total != TOTAL_WEIGHT_TARGET:
        engine_logger.error(
            f"AGENT 1: Weight sum mismatch — expected "
            f"{TOTAL_WEIGHT_TARGET}, got {total}."
        )
        return []

    engine_logger.info(
        f"AGENT 1: {count} skills ranked and weighted (Python). "
        f"Weight sum = {total}. ✅"
    )
    return final_skills


# ═══════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════

def strip_line_prefixes(snippet: str) -> str:
    """Removes accidental line-number prefixes from LLM-generated snippets."""
    cleaned_lines = []
    for line in snippet.splitlines():
        stripped = line.strip()
        match = re.match(
            r'^Lines?\s*\d+[\s\u2013\-–]*\d*\s*:\s*(.+)$',
            stripped,
            re.IGNORECASE
        )
        if match:
            stripped = match.group(1)
        elif stripped and stripped[0].isdigit():
            colon_pos = stripped.find(": ")
            if colon_pos != -1 and stripped[:colon_pos].strip().isdigit():
                stripped = stripped[colon_pos + 2:]
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def number_lines(code: str) -> str:
    """Prepends line numbers to code for LLM reference."""
    lines = code.splitlines()
    return "\n".join(
        f"{i + 1}: {line}"
        for i, line in enumerate(lines)
    )


def clean_json(content: str) -> str:
    """Strips markdown fences and whitespace from LLM JSON responses."""
    return content.replace("```json", "").replace("```", "").strip()
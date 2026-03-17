"""
backend/agents/agent1_validators.py
Pure Python validation, ranking, deduplication, and assignment-context filtering
for Agent 1 micro skills. Zero LLM dependency.

Handles:
- Skill formula validation and auto-fix
- Three-layer concept deduplication
- Generic language construct rejection
- Assignment-context filtering for output/display skills
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
)
from backend.utils.logger import engine_logger


# TODO:
# Move these fixed literals into constants.py later
# to fully comply with Single Source of Truth.
VALID_SKILL_VERBS = ["Be able to", "Use", "Avoid", "Handle", "Implement"]
MIN_SKILL_WORDS = 6
MAX_SKILL_WORDS = 15

SUBJECT_SYNONYM_GROUPS = [
    {
        "first element", "1st cell", "first cell", "1st element",
        "first position", "arr[0]", "first value"
    },
    {
        "last element", "last cell", "last position", "last value",
        "arr[n-1]", "arr[4]"
    },
]

OPERATION_CONTEXT_KEYWORDS = [
    "shift", "left shift", "rotate", "wrap", "wrapping",
    "sort", "reverse", "swap", "merge", "insert",
    "delete", "remove", "search"
]

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

SPECIFICITY_KEYWORDS = [
    "shift", "sort", "reverse", "rotate", "swap", "merge",
    "insert", "delete", "remove", "search",
    "temp", "temporary", "preserve", "avoid losing",
    "losing data", "data loss", "avoid overwriting",
    "arr[", "arr [", "scanf", "&arr",
    "boundary", "wrap", "wrapping", "first element", "1st cell",
    "last position", "first position",
    "integers into", "characters into", "strings into",
]

OUTPUT_SKILL_KEYWORDS = [
    "printf",
    "print",
    "display",
    "output",
    "separator",
    "comma",
    "format",
    "trailing comma",
    "between elements",
]

OUTPUT_OBJECTIVE_HEADER_KEYWORDS = [
    "micro skills",
    "micro skills to evaluate",
    "skills to evaluate",
    "learning objectives",
    "objectives",
    "evaluation criteria",
    "grading criteria",
]

EXPLICIT_OBJECTIVE_SIGNAL_KEYWORDS = [
    "learning objective",
    "learning objectives",
    "skill",
    "skills",
    "evaluate",
    "evaluation",
    "objective",
    "criteria",
]

SHIFT_KEYWORDS = [
    "shift",
    "left shift",
    "arr[i-1]",
    "arr[i - 1]",
    "arr[i+1]",
    "arr[i + 1]",
]

BOUNDARY_WRAP_KEYWORDS = [
    "boundary",
    "wrap",
    "wrapping",
    "first element",
    "1st cell",
    "first position",
    "last position",
    "arr[0]",
    "arr[4]",
]

TEMP_SAFETY_KEYWORDS = [
    "temp",
    "temporary",
    "preserve",
    "avoid losing",
    "losing data",
    "data loss",
    "overwrite",
    "overwriting",
]

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


def validate_and_fix_skills(skills: list[str]) -> list[str]:
    """
    Validates each LLM-generated skill against the formula.
    Auto-fixes recoverable issues. Drops unrecoverable ones.
    Only called for LLM-generated skills — never for parser-extracted.
    """
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: validate_and_fix_skills expected a list.")
        return []

    validated = []

    for raw_skill in skills:
        if not isinstance(raw_skill, str):
            engine_logger.warning(
                f"AGENT 1: Dropped invalid skill (not a string): {raw_skill}"
            )
            continue

        skill = _normalize_whitespace(raw_skill)

        if not skill:
            engine_logger.warning("AGENT 1: Dropped invalid skill (empty string).")
            continue

        if is_valid_skill_format(skill):
            validated.append(skill)
            engine_logger.info(f"AGENT 1: Skill format valid: '{skill}'")
            continue

        fixed = auto_fix_skill_format(skill)
        if fixed:
            validated.append(fixed)
            engine_logger.info(
                f"AGENT 1: Auto-fixed skill: '{skill}' -> '{fixed}'"
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
    if not isinstance(skill, str):
        return False

    normalized = _normalize_whitespace(skill)
    if not normalized:
        return False

    has_valid_verb = any(
        normalized.lower().startswith(verb.lower())
        for verb in VALID_SKILL_VERBS
    )
    if not has_valid_verb:
        return False

    word_count = len(normalized.split())
    if word_count < MIN_SKILL_WORDS or word_count > MAX_SKILL_WORDS:
        return False

    return True


def auto_fix_skill_format(skill: str) -> str | None:
    """
    Attempts to fix a skill that fails formula validation.
    Returns fixed skill text or None if unfixable.
    """
    if not isinstance(skill, str):
        return None

    original = _normalize_whitespace(skill)
    if not original:
        return None

    fixed = original

    has_valid_verb = any(
        original.lower().startswith(verb.lower())
        for verb in VALID_SKILL_VERBS
    )
    if not has_valid_verb:
        first_char = original[0].lower()
        fixed = f"Be able to {first_char}{original[1:]}"

    words = fixed.split()

    if len(words) < MIN_SKILL_WORDS:
        return None

    if len(words) > MAX_SKILL_WORDS:
        fixed = " ".join(words[:MAX_SKILL_WORDS])

    fixed = _normalize_whitespace(fixed)
    return fixed if is_valid_skill_format(fixed) else None


def deduplicate_skills(skills: list[str]) -> list[str]:
    """
    Removes overlapping concepts within a skill list.
    Keeps the first occurrence, drops later duplicates.
    Uses is_same_concept() for comparison.
    """
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

        is_duplicate = False
        for existing in unique:
            if is_same_concept(existing, skill):
                engine_logger.info(
                    f"AGENT 1: Dedup — dropped '{skill}' "
                    f"(overlaps with: '{existing}')"
                )
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(skill)

    removed_count = len([s for s in skills if isinstance(s, str) and s.strip()]) - len(unique)

    if removed_count > 0:
        engine_logger.info(
            f"AGENT 1: Dedup — {removed_count} duplicate skill(s) removed."
        )
    else:
        engine_logger.info("AGENT 1: Dedup — no duplicates found.")

    return unique


def is_same_concept(text1: str, text2: str) -> bool:
    """
    Three-layer concept overlap detection.

    Layer 1 — Exact or same core sentence after removing action verb
    Layer 2 — Shared subject + shared operation context
    Layer 3 — Bucket-aware overlap rules
    """
    normalized_1 = _normalize_skill_text(text1)
    normalized_2 = _normalize_skill_text(text2)

    if not normalized_1 or not normalized_2:
        return False

    if normalized_1 == normalized_2:
        return True

    core_1 = _remove_action_verb(normalized_1)
    core_2 = _remove_action_verb(normalized_2)

    if core_1 == core_2:
        return True

    if _is_shift_boundary_split(core_1, core_2):
        engine_logger.info(
            f"AGENT 1: Overlap detected — split shift/boundary concept: "
            f"'{text1}' vs '{text2}'"
        )
        return True

    if _shares_subject_and_operation(core_1, core_2):
        engine_logger.info(
            f"AGENT 1: Overlap detected — shared subject + shared operation: "
            f"'{text1}' vs '{text2}'"
        )
        return True

    bucket_1 = _get_concept_bucket(core_1)
    bucket_2 = _get_concept_bucket(core_2)

    if bucket_1 and bucket_1 == bucket_2:
        if _bucket_specific_overlap(core_1, core_2, bucket_1):
            return True

    if _meaningful_containment(core_1, core_2):
        return True

    return False


def reject_generic_skills(skills: list[str]) -> list[str]:
    """
    Rejects skills that describe basic language constructs
    without assignment-specific detail.
    """
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: reject_generic_skills expected a list.")
        return []

    non_generic = []

    for raw_skill in skills:
        if not isinstance(raw_skill, str):
            continue

        skill = _normalize_whitespace(raw_skill)
        if not skill:
            continue

        if is_generic_skill(skill):
            engine_logger.warning(f"AGENT 1: Rejected generic skill: '{skill}'")
        else:
            non_generic.append(skill)

    rejected_count = len([s for s in skills if isinstance(s, str) and s.strip()]) - len(non_generic)

    if rejected_count > 0:
        engine_logger.info(
            f"AGENT 1: Generic rejection — {rejected_count} generic skill(s) removed."
        )
    else:
        engine_logger.info("AGENT 1: Generic rejection — none were generic.")

    return non_generic


def is_generic_skill(skill: str) -> bool:
    """
    Returns True if a skill describes a basic language construct
    without assignment-specific detail.
    """
    if not isinstance(skill, str):
        return True

    lower = _normalize_skill_text(skill)

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


def filter_skills_by_assignment_context(
    skills: list[str],
    instructions: str
) -> list[str]:
    """
    Removes LLM-generated skills that are invalid for this assignment context.

    Current enforced rule:
    - Output/display skills are rejected unless instructions.md explicitly
      lists output/printing as a learning objective.

    Applied ONLY to LLM-generated skills — never to parser-extracted teacher skills.
    """
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: filter_skills_by_assignment_context expected a list.")
        return []

    output_allowed = instructions_explicitly_allow_output_skills(instructions)
    filtered_skills = []

    for raw_skill in skills:
        if not isinstance(raw_skill, str):
            continue

        skill = _normalize_whitespace(raw_skill)
        if not skill:
            continue

        if is_output_display_skill(skill) and not output_allowed:
            engine_logger.warning(
                f"AGENT 1: Rejected output/display skill because instructions.md "
                f"does not explicitly list output as a learning objective: '{skill}'"
            )
            continue

        filtered_skills.append(skill)

    return filtered_skills


def instructions_explicitly_allow_output_skills(instructions: str) -> bool:
    """
    Returns True only when instructions.md explicitly frames output/printing
    as a learning objective or micro skill.

    Conservative by design:
    - 'INPUT-OUTPUT' example lines do NOT count.
    - Generic assignment wording like 'print the result' does NOT count.
    - A labeled objective/skill section containing output keywords DOES count.
    """
    if not isinstance(instructions, str) or not instructions.strip():
        return False

    lines = instructions.splitlines()

    section_start_index = None
    for index, line in enumerate(lines):
        normalized_heading = line.strip().lstrip("#").strip().lower()

        if any(
            normalized_heading == keyword or normalized_heading.startswith(keyword)
            for keyword in OUTPUT_OBJECTIVE_HEADER_KEYWORDS
        ):
            section_start_index = index
            break

    if section_start_index is not None:
        for line in lines[section_start_index + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                break

            lowered = stripped.lower()
            if _contains_output_keyword(lowered):
                return True

    for line in lines:
        lowered = line.strip().lower()
        if not lowered:
            continue

        has_objective_signal = any(
            signal in lowered for signal in EXPLICIT_OBJECTIVE_SIGNAL_KEYWORDS
        )
        if has_objective_signal and _contains_output_keyword(lowered):
            return True

    return False


def is_output_display_skill(skill: str) -> bool:
    """
    Returns True if the skill is primarily about output/printing/formatting.
    """
    if not isinstance(skill, str):
        return False

    lowered = _normalize_skill_text(skill)
    return _get_io_subtype(lowered) == "output"


def enforce_ranking_rules(skills: list[str]) -> list[str]:
    """
    Pure Python ranking. No LLM.
    RULE 1: transform -> first
    RULE 2: safety -> second
    RULE 3: io -> last
    RULE 4: everything else -> middle
    """
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: enforce_ranking_rules expected a list.")
        return []

    cleaned_skills = [
        _normalize_whitespace(skill)
        for skill in skills
        if isinstance(skill, str) and skill.strip()
    ]

    if len(cleaned_skills) < 2:
        return cleaned_skills

    transform_keywords = [
        "shift", "sort", "reverse", "search", "swap",
        "rotate", "merge", "insert", "delete", "remove",
        "boundary", "wrap", "wrapping", "first element", "last position"
    ]
    safety_keywords = [
        "temp", "temporary", "preserve", "avoid losing",
        "avoid overwriting", "losing data", "data loss"
    ]
    io_keywords = ["scanf", "printf", "read input", "print output"]

    def classify(skill_text: str) -> str:
        lower = skill_text.lower()

        if any(keyword in lower for keyword in safety_keywords):
            return "safety"
        if any(keyword in lower for keyword in io_keywords):
            return "io"
        if any(keyword in lower for keyword in transform_keywords):
            return "transform"
        return "middle"

    buckets = {"transform": [], "safety": [], "middle": [], "io": []}

    for skill in cleaned_skills:
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
    1. Order: transform -> safety -> middle -> io
    2. Trim to MAX_MICRO_SKILLS
    3. Validate minimum count
    4. Assign weights from distribution table
    5. Validate sum = TOTAL_WEIGHT_TARGET
    """
    ranked = enforce_ranking_rules(skills)
    ranked = ranked[:MAX_MICRO_SKILLS]

    count = len(ranked)

    if count < MIN_MICRO_SKILLS:
        engine_logger.error(
            f"AGENT 1: Only {count} skills available after ranking. "
            f"Minimum required is {MIN_MICRO_SKILLS}. Aborting."
        )
        return []

    if count not in DEFAULT_WEIGHT_DISTRIBUTIONS:
        engine_logger.error(
            f"AGENT 1: No weight distribution defined for {count} skills."
        )
        return []

    weights = DEFAULT_WEIGHT_DISTRIBUTIONS[count]

    final_skills = [
        {
            "text": skill_text,
            "rank": index + 1,
            "weight": weights[index]
        }
        for index, skill_text in enumerate(ranked)
    ]

    total = sum(skill["weight"] for skill in final_skills)
    if total != TOTAL_WEIGHT_TARGET:
        engine_logger.error(
            f"AGENT 1: Weight sum mismatch — expected "
            f"{TOTAL_WEIGHT_TARGET}, got {total}."
        )
        return []

    engine_logger.info(
        f"AGENT 1: {count} skills ranked and weighted (Python). "
        f"Weight sum = {total}."
    )
    return final_skills


def strip_line_prefixes(snippet: str) -> str:
    """
    Removes accidental line-number prefixes from LLM-generated snippets.
    """
    if not isinstance(snippet, str):
        return ""

    cleaned_lines = []

    for line in snippet.splitlines():
        stripped = line.rstrip()

        match = re.match(
            r"^Lines?\s*\d+[\s\u2013\-–]*\d*\s*:\s*(.+)$",
            stripped.strip(),
            re.IGNORECASE
        )
        if match:
            stripped = match.group(1)
        else:
            numbered_match = re.match(r"^\s*\d+\s*:\s*(.+)$", stripped)
            if numbered_match:
                stripped = numbered_match.group(1)

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def number_lines(code: str) -> str:
    """
    Prepends line numbers to code for LLM reference.
    """
    if not isinstance(code, str) or not code:
        return ""

    lines = code.splitlines()
    return "\n".join(
        f"{index + 1}: {line}"
        for index, line in enumerate(lines)
    )


def clean_json(content: str) -> str:
    """
    Strips markdown fences and surrounding whitespace from LLM JSON responses.
    """
    if not isinstance(content, str):
        return ""

    cleaned = content.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _normalize_whitespace(text: str) -> str:
    """
    Collapses repeated whitespace into single spaces.
    """
    return " ".join(text.strip().split())


def _normalize_skill_text(text: str) -> str:
    """
    Lowercases and trims punctuation for concept comparison.
    """
    normalized = _normalize_whitespace(text).lower()
    return normalized.rstrip(".,;:")


def _remove_action_verb(text: str) -> str:
    """
    Removes the leading action verb for better concept comparison.
    """
    for verb in VALID_SKILL_VERBS:
        verb_lower = verb.lower()
        if text.startswith(verb_lower):
            remainder = text[len(verb_lower):].strip(" ,")
            return remainder
    return text


def _shares_subject_and_operation(text1: str, text2: str) -> bool:
    """
    Detects overlap when two skills reference the same subject
    and the same operation context.
    """
    shared_operation = any(
        keyword in text1 and keyword in text2
        for keyword in OPERATION_CONTEXT_KEYWORDS
    )

    if not shared_operation:
        return False

    for synonym_group in SUBJECT_SYNONYM_GROUPS:
        has_subject_1 = any(term in text1 for term in synonym_group)
        has_subject_2 = any(term in text2 for term in synonym_group)

        if has_subject_1 and has_subject_2:
            return True

    return False


def _get_concept_bucket(text: str) -> str | None:
    """
    Classifies a skill text into a concept bucket.
    """
    lower = text.lower()

    if any(keyword in lower for keyword in TEMP_SAFETY_KEYWORDS):
        return "data_safety"

    io_keywords = ["scanf", "printf", "read input", "print output", "display", "separator", "comma"]
    if any(keyword in lower for keyword in io_keywords):
        return "io"

    transform_keywords = [
        "shift", "sort", "reverse", "rotate", "swap",
        "merge", "insert", "delete", "remove", "search",
        "boundary", "wrap", "wrapping", "first element", "last position"
    ]
    if any(keyword in lower for keyword in transform_keywords):
        return "transform"

    access_keywords = ["arr[i]", "array elements", "access", "index"]
    if any(keyword in lower for keyword in access_keywords):
        return "array_access"

    return None


def _bucket_specific_overlap(text1: str, text2: str, bucket: str) -> bool:
    """
    Safer bucket-aware overlap logic.
    Same bucket does NOT automatically mean duplicate.
    """
    if bucket == "data_safety":
        shared_safety = any(keyword in text1 and keyword in text2 for keyword in TEMP_SAFETY_KEYWORDS)
        shared_operation = any(
            keyword in text1 and keyword in text2
            for keyword in OPERATION_CONTEXT_KEYWORDS
        )
        return shared_safety and (
            shared_operation
            or "temporary" in text1 or "temporary" in text2
            or "temp" in text1 or "temp" in text2
        )

    if bucket == "io":
        subtype_1 = _get_io_subtype(text1)
        subtype_2 = _get_io_subtype(text2)

        if subtype_1 != subtype_2 or subtype_1 is None:
            return False

        if subtype_1 == "input":
            return (
                ("scanf" in text1 and "scanf" in text2)
                or _meaningful_containment(text1, text2)
            )

        if subtype_1 == "output":
            shared_output_keywords = sum(
                1 for keyword in OUTPUT_SKILL_KEYWORDS
                if keyword in text1 and keyword in text2
            )

            has_formatting_focus = any(
                keyword in text1 or keyword in text2
                for keyword in ["comma", "separator", "between elements", "trailing comma", "format"]
            )

            return (
                shared_output_keywords >= 1 and has_formatting_focus
            ) or _meaningful_containment(text1, text2)

        return False

    if bucket == "array_access":
        access_keywords = ["arr[i]", "index", "access", "array elements"]
        shared_access = any(keyword in text1 and keyword in text2 for keyword in access_keywords)
        return shared_access and _meaningful_containment(text1, text2)

    if bucket == "transform":
        if _is_shift_boundary_split(text1, text2):
            return True

        shared_operation = any(
            keyword in text1 and keyword in text2
            for keyword in OPERATION_CONTEXT_KEYWORDS
        )
        if not shared_operation:
            return False

        if _shares_subject_and_operation(text1, text2):
            return True

        return _meaningful_containment(text1, text2)

    return False


def _meaningful_containment(text1: str, text2: str) -> bool:
    """
    Substring containment with minimum length guard.
    """
    shorter = min(text1, text2, key=len)
    longer = max(text1, text2, key=len)

    if len(shorter) < 12:
        return False

    return shorter in longer


def _get_io_subtype(text: str) -> str | None:
    """
    Splits io bucket into input vs output for safer dedup.
    """
    lowered = text.lower()

    if "scanf" in lowered or "read input" in lowered or "read 5 integers" in lowered:
        return "input"

    if _contains_output_keyword(lowered):
        return "output"

    return None


def _contains_output_keyword(text: str) -> bool:
    """
    Returns True if text contains any output/display keyword.
    """
    return any(keyword in text for keyword in OUTPUT_SKILL_KEYWORDS)


def _references_first_boundary(text: str) -> bool:
    """
    Returns True if text refers to the first-element boundary case.
    """
    first_group = SUBJECT_SYNONYM_GROUPS[0]
    return any(term in text for term in first_group) or "last position" in text or "wrap" in text or "wrapping" in text


def _is_shift_boundary_split(text1: str, text2: str) -> bool:
    """
    Detects when one skill describes the main shift operation and another
    separately describes the first-element boundary/wrap case.

    This exact split is forbidden by the project rules:
    - shift logic + first-cell boundary should be ONE skill, not two.
    """
    text1_has_shift = any(keyword in text1 for keyword in SHIFT_KEYWORDS)
    text2_has_shift = any(keyword in text2 for keyword in SHIFT_KEYWORDS)

    text1_has_boundary = _references_first_boundary(text1)
    text2_has_boundary = _references_first_boundary(text2)

    if not (text1_has_boundary and text2_has_boundary):
        return False

    if text1_has_shift or text2_has_shift:
        return True

    return False
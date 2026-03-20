"""Skill validation, generic rejection, assignment-context filtering."""

import re

from backend.config.constants import (
    DEFAULT_WEIGHT_DISTRIBUTIONS,
    MAX_MICRO_SKILLS,
    MIN_MICRO_SKILLS,
    TOTAL_WEIGHT_TARGET,
)
from backend.utils.logger import engine_logger

# ── CONSTANTS ───────────────────────────────────────────────────────
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
    "use a loop", "use for loop", "use while loop", "use do while",
    "iterate through", "iterate over", "loop through", "loop to iterate",
    "traverse array", "traverse the array", "declare variable",
    "define variable", "initialize variable", "use if statement",
    "use conditional", "use switch statement", "return value from",
    "call a function", "use function to", "print output", "display output",
    "display result", "print result", "output the result",
    "print the array", "display the array",
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
    "printf", "print", "display", "output",
    "separator", "comma", "format", "trailing comma", "between elements",
]

OUTPUT_OBJECTIVE_HEADER_KEYWORDS = [
    "micro skills", "micro skills to evaluate", "skills to evaluate",
    "learning objectives", "objectives", "evaluation criteria", "grading criteria",
]

EXPLICIT_OBJECTIVE_SIGNAL_KEYWORDS = [
    "learning objective", "learning objectives", "skill", "skills",
    "evaluate", "evaluation", "objective", "criteria",
]

SHIFT_KEYWORDS = [
    "shift", "left shift", "arr[i-1]", "arr[i - 1]", "arr[i+1]", "arr[i + 1]",
]

TEMP_SAFETY_KEYWORDS = [
    "temp", "temporary", "preserve", "avoid losing",
    "losing data", "data loss", "overwrite", "overwriting",
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

LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$")
LINE_RANGE_PREFIX_PATTERN = re.compile(
    r"^Lines?\s*\d+[\s\u2013\-–]*\d*\s*:(.+)$",
    re.IGNORECASE,
)
SIMPLE_LINE_PREFIX_PATTERN = re.compile(r"^\s*\d+\s*:(.+)$")
EXAMPLE_LINE_KEYWORDS = [
    "example", "sample", "expected output", "output example",
    "input/output", "input-output",
]

# ── HELPER FUNCTIONS (used by both modules) ─────────────────────────
def _normalize_whitespace(text: str) -> str:
    """Collapses repeated whitespace into single spaces."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def _normalize_skill_text(text: str) -> str:
    """Lowercases and trims punctuation for concept comparison."""
    normalized = _normalize_whitespace(text).lower()
    return normalized.rstrip(".,;:")


def _extract_list_item_text(text: str) -> str | None:
    """Extracts the payload of a bullet or numbered list item."""
    if not isinstance(text, str):
        return None
    match = LIST_ITEM_PATTERN.match(text.strip())
    if not match:
        return None
    payload = _normalize_whitespace(match.group(1))
    return payload or None


def _is_example_line(text: str) -> bool:
    """Returns True for example/sample lines."""
    if not isinstance(text, str):
        return False
    lowered = text.strip().lower()
    if lowered.startswith("input:") or lowered.startswith("output:"):
        return True
    return any(keyword in lowered for keyword in EXAMPLE_LINE_KEYWORDS)


def _contains_output_keyword(text: str) -> bool:
    """Returns True if text contains any output/display keyword."""
    return any(keyword in text for keyword in OUTPUT_SKILL_KEYWORDS)


def _get_io_subtype(text: str) -> str | None:
    """Splits io bucket into input vs output."""
    lowered = text.lower()
    if "scanf" in lowered or "read input" in lowered or "read 5 integers" in lowered:
        return "input"
    if _contains_output_keyword(lowered):
        return "output"
    return None


# ── PUBLIC VALIDATION FUNCTIONS ────────────────────────────────────
def validate_and_fix_skills(skills: list[str]) -> list[str]:
    """Validates each LLM‑generated skill against the formula."""
    if not isinstance(skills, list):
        engine_logger.error("AGENT 1: validate_and_fix_skills expected a list.")
        return []

    validated = []
    for raw_skill in skills:
        if not isinstance(raw_skill, str):
            engine_logger.warning(f"Dropped invalid skill (not a string): {raw_skill}")
            continue
        skill = _normalize_whitespace(raw_skill)
        if not skill:
            engine_logger.warning("Dropped invalid skill (empty string).")
            continue
        if is_valid_skill_format(skill):
            validated.append(skill)
            engine_logger.info(f"Skill format valid: '{skill}'")
            continue
        fixed = auto_fix_skill_format(skill)
        if fixed:
            validated.append(fixed)
            engine_logger.info(f"Auto-fixed skill: '{skill}' -> '{fixed}'")
        else:
            engine_logger.warning(f"Dropped invalid skill (unfixable): '{skill}'")
    return validated


def is_valid_skill_format(skill: str) -> bool:
    """Checks if a skill matches the required formula."""
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
    return MIN_SKILL_WORDS <= word_count <= MAX_SKILL_WORDS


def auto_fix_skill_format(skill: str) -> str | None:
    """Attempts to fix a skill that fails formula validation."""
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
        fixed = f"Be able to {original[0].lower()}{original[1:]}"
    words = fixed.split()
    if len(words) < MIN_SKILL_WORDS:
        return None
    if len(words) > MAX_SKILL_WORDS:
        fixed = " ".join(words[:MAX_SKILL_WORDS])
    fixed = _normalize_whitespace(fixed)
    return fixed if is_valid_skill_format(fixed) else None


def is_generic_skill(skill: str) -> bool:
    """Returns True if a skill describes a basic language construct."""
    if not isinstance(skill, str):
        return True
    lower = _normalize_skill_text(skill)
    matches_generic = any(
        pattern in lower for pattern in GENERIC_CONSTRUCT_PATTERNS
    )
    if not matches_generic:
        return False
    has_specificity = any(
        keyword in lower for keyword in SPECIFICITY_KEYWORDS
    )
    return not has_specificity


def reject_generic_skills(skills: list[str]) -> list[str]:
    """Rejects skills that describe basic language constructs."""
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
            engine_logger.warning(f"Rejected generic skill: '{skill}'")
        else:
            non_generic.append(skill)
    return non_generic


def is_output_display_skill(skill: str) -> bool:
    """Returns True if the skill is primarily about output/printing."""
    if not isinstance(skill, str):
        return False
    lowered = _normalize_skill_text(skill)
    return _get_io_subtype(lowered) == "output"


def instructions_explicitly_allow_output_skills(instructions: str) -> bool:
    """True only if a labeled section contains an output skill bullet."""
    if not isinstance(instructions, str) or not instructions.strip():
        return False
    lines = instructions.splitlines()
    section_start_index = None
    for idx, line in enumerate(lines):
        heading = line.strip().lstrip("#").strip().lower()
        if any(
            heading == kw or heading.startswith(kw)
            for kw in OUTPUT_OBJECTIVE_HEADER_KEYWORDS
        ):
            section_start_index = idx
            break
    if section_start_index is not None:
        for line in lines[section_start_index + 1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            list_item = _extract_list_item_text(stripped)
            if not list_item:
                continue
            lowered_item = list_item.lower()
            if _is_example_line(lowered_item):
                continue
            if _contains_output_keyword(lowered_item):
                return True
    return False


def filter_skills_by_assignment_context(
    skills: list[str], instructions: str
) -> list[str]:
    """Removes output skills unless explicitly allowed by instructions."""
    if not isinstance(skills, list):
        engine_logger.error("filter_skills_by_assignment_context expected a list.")
        return []
    output_allowed = instructions_explicitly_allow_output_skills(instructions)
    filtered = []
    for raw_skill in skills:
        if not isinstance(raw_skill, str):
            continue
        skill = _normalize_whitespace(raw_skill)
        if not skill:
            continue
        if is_output_display_skill(skill) and not output_allowed:
            engine_logger.warning(f"Rejected output skill: '{skill}'")
            continue
        filtered.append(skill)
    return filtered
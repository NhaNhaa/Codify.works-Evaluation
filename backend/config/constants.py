"""
backend/config/constants.py

Central repository for all fixed literals and configuration boundaries.
Follows the 'Single Source of Truth' standard.
"""

# ── MICRO SKILL BOUNDARIES ─────────────────────────────────────────
MIN_MICRO_SKILLS = 3
MAX_MICRO_SKILLS = 6
TOTAL_WEIGHT_TARGET = 10

# ── WEIGHT DISTRIBUTIONS ───────────────────────────────────────────
# Format: {skill_count: [rank_1_weight, rank_2_weight, ...]}
# Rank 1 = most important = highest weight
DEFAULT_WEIGHT_DISTRIBUTIONS = {
    3: [5, 3, 2],
    4: [4, 3, 2, 1],
    5: [3, 3, 2, 1, 1],
    6: [3, 2, 2, 1, 1, 1],
}

# ── RANKING CRITERIA ───────────────────────────────────────────────
# Used by Agent 1 to rank micro skills by importance.
# Higher priority criteria appear first.
RANKING_CRITERIA = [
    "1. Core algorithm — the central logic of the assignment (e.g. the shift operation itself). Always rank highest.",
    "2. Data safety — prevents data loss or corruption (e.g. using a temporary variable). Always rank second if present.",
    "3. Program-breaking if missing — the program produces wrong output without this skill.",
    "4. Addressed specifically and explicitly in teacher correction code.",
    "5. Common student mistake pattern — frequently missed by beginners.",
    "6. Input/output skills (scanf, printf) rank LOWEST — they are supporting skills, not the core algorithm.",
]

# ── SKILL QUALITY RULES ────────────────────────────────────────────
# Used by Agent 1 prompts to enforce clean, non-redundant skill lists.
SKILL_QUALITY_RULES = [
    "Each skill must test ONE distinct concept — never split one concept into two skills.",
    "No two skills may evaluate the same lines of student code.",
    "Generic programming skills (loops, variables, functions) are NOT valid unless they are the specific focus of this assignment.",
    "Output/display skills (printf) are NOT valid unless explicitly listed as a learning objective in instructions.md.",
    "Skill text must be one clean sentence — maximum 15 words. No reasoning, no comparisons, no explanations inside the skill text.",
]

# ── AGENT RE-EVALUATION & SELF-CHECK ───────────────────────────────
MAX_REVERIFICATION_ATTEMPTS = 3
MAX_SELF_CHECK_ATTEMPTS = 2
MAX_SNIPPET_LINES = 6

# ── FEEDBACK CONFIGURATION ─────────────────────────────────────────
HIGH_WEIGHT_THRESHOLD = 3

STRUCTURE_HIGH_WEIGHT = ["WHAT YOU DID", "WHY IT IS WRONG", "HOW TO FIX IT"]
STRUCTURE_LOW_WEIGHT = ["WHAT YOU DID", "HOW TO FIX IT"]

FEEDBACK_TONE = "encouraging mentor — warm and supportive"
FEEDBACK_LANGUAGE = "English"
FEEDBACK_PASS_DEPTH = "medium"
FEEDBACK_SUMMARY = False

# ── FEEDBACK LENGTH LIMITS ─────────────────────────────────────────
# Maximum sentences per feedback type.
# Enforced in Agent 3 prompts.
FEEDBACK_PASS_MAX_SENTENCES = 3
FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION = 2

# ── SYSTEM MODES ───────────────────────────────────────────────────
EVALUATION_MODE_SEQUENTIAL = "sequential"
EVALUATION_MODE_BATCH = "batch"

# ── OUTPUT FORMATS ─────────────────────────────────────────────────
# JSON is the source of truth. Markdown is derived from JSON.
OUTPUT_FORMAT_JSON = "json"
OUTPUT_FORMAT_MARKDOWN = "markdown"

# ── LINE DETECTION ─────────────────────────────────────────────────
LINE_FORMAT_RANGE = "range"

LINE_DETECTION_PASS_LINE_KEY = "pass_line"
LINE_DETECTION_FAIL_LINE_KEY = "fail_line"
LINE_DETECTION_MISSING_SKILL_KEY = "missing_skill"

LINE_DETECTION = {
    LINE_DETECTION_PASS_LINE_KEY: LINE_FORMAT_RANGE,
    LINE_DETECTION_FAIL_LINE_KEY: LINE_FORMAT_RANGE,
    LINE_DETECTION_MISSING_SKILL_KEY: None,
}

# ── SECURITY.py ──────────────────────────────────────────────────
ALLOWED_INPUT_FILE_EXTENSIONS = frozenset({".c", ".md"})
PROHIBITED_CONTENT_KEYWORDS = (
    "system(",
    "popen(",
    "exec(",
    "fork(",
    "execve(",
)
DEFAULT_FILE_ENCODING = "utf-8"

# ── FORMATTER.py ───────────────────────────────────────────────
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"

DEFAULT_UNKNOWN_STUDENT_ID = "unknown"
DEFAULT_UNKNOWN_ASSIGNMENT_ID = "unknown"
DEFAULT_UNKNOWN_SKILL = "Unknown Skill"

DEFAULT_LINE_REFERENCE = "Line reference unavailable"
DEFAULT_PASS_FEEDBACK = "Well done! Your implementation is correct."
DEFAULT_FAIL_FEEDBACK = "This implementation does not satisfy the micro skill requirement."
DEFAULT_MISSING_STUDENT_CODE = "Relevant student code was not detected."
DEFAULT_MISSING_RECOMMENDED_FIX = "Recommended fix is unavailable."

INVALID_SKILL_ENTRY_MARKDOWN = "### ❌ Invalid Skill Entry\n---"
INVALID_EVALUATION_REPORT_MARKDOWN = "# Evaluation Report — unknown\nInvalid evaluation result."
NO_SKILLS_TO_DISPLAY_MESSAGE = "No skills to display."

MARKDOWN_CODE_BLOCK_LANGUAGE = "c"

# ── SKILL_PARSER.py ───────────────────────────────────────────────────
SKILL_SECTION_KEYWORDS = (
    "micro skills",
    "micro skills to evaluate",
    "skills to evaluate",
    "learning objectives",
    "skills",
    "objectives",
    "evaluation criteria",
    "grading criteria",
)

MIN_INSTRUCTION_CONTENT_LENGTH = 30

# ── LLM_CLIENT.py ────────────────────────────────────────────────────
LLM_MAX_ERROR_LOG_LENGTH = 200
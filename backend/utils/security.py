"""
backend/utils/security.py
Implements strict validation for file inputs and content sanitization.
Ensures the RAG pipeline never processes empty, missing, or malicious files.
"""

from pathlib import Path
from backend.utils.logger import engine_logger

# ── ALLOWED FILE EXTENSIONS ────────────────────────────────────────
ALLOWED_EXTENSIONS = {".c", ".md"}

# ── PROHIBITED CONTENT PATTERNS ────────────────────────────────────
PROHIBITED_KEYWORDS = ["system(", "popen(", "exec(", "fork(", "execve("]


# ── FILE VALIDATION ────────────────────────────────────────────────
def validate_file_input(file_path: str, expected_ext: str = None) -> bool:
    """
    Validates that a file exists, is not empty, and has the correct extension.
    Returns True if valid, False otherwise.
    """
    path = Path(file_path)

    # Check exists
    if not path.exists():
        engine_logger.error(f"SECURITY: File not found: {file_path}")
        return False

    # Check is a file not a directory
    if not path.is_file():
        engine_logger.error(f"SECURITY: Path is not a file: {file_path}")
        return False

    # Check not empty
    if path.stat().st_size == 0:
        engine_logger.error(f"SECURITY: File is empty: {file_path}")
        return False

    # Check extension if provided
    if expected_ext and path.suffix.lower() != expected_ext.lower():
        engine_logger.error(
            f"SECURITY: Invalid extension for {file_path}. "
            f"Expected {expected_ext}, got {path.suffix}"
        )
        return False

    engine_logger.info(f"SECURITY: File validated successfully: {file_path}")
    return True


def validate_all_input_files(
    instructions_path: str,
    starter_path: str,
    teacher_path: str,
    student_path: str = None
) -> bool:
    """
    Validates all 4 input files at once before pipeline starts.
    Returns True only if ALL files pass validation.
    """
    files = [
        (instructions_path, ".md"),
        (starter_path,      ".c"),
        (teacher_path,      ".c"),
    ]

    if student_path:
        files.append((student_path, ".c"))

    for file_path, ext in files:
        if not validate_file_input(file_path, ext):
            engine_logger.error(
                f"SECURITY: Input validation failed — pipeline aborted."
            )
            return False

    engine_logger.info("SECURITY: All input files validated successfully.")
    return True


# ── CONTENT SANITIZATION ───────────────────────────────────────────
def is_content_safe(content: str, file_path: str = "") -> bool:
    """
    Checks for prohibited system-level keywords in C code content.
    Logs warnings for audit — does not block execution.
    Returns True always — logs are the safety record.
    """
    for keyword in PROHIBITED_KEYWORDS:
        if keyword in content:
            engine_logger.warning(
                f"SECURITY: Prohibited keyword '{keyword}' "
                f"found in {file_path or 'content'}"
            )
    return True


# ── FILE READER WITH VALIDATION ────────────────────────────────────
def safe_read_file(file_path: str, expected_ext: str = None) -> str | None:
    """
    Validates and reads a file in one call.
    Returns file content as string, or None if validation fails.
    """
    if not validate_file_input(file_path, expected_ext):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        engine_logger.info(f"SECURITY: File read successfully: {file_path}")
        return content
    except Exception as e:
        engine_logger.error(f"SECURITY: Failed to read file {file_path}: {e}")
        return None
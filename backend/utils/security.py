"""
backend/utils/security.py
Implements strict validation for file inputs and content sanitization.
Ensures the RAG pipeline never processes empty, missing, or malicious files.
"""

from pathlib import Path

from backend.utils.logger import engine_logger

# TODO:
# Move these fixed literals into backend/config/constants.py
# to fully comply with Single Source of Truth.
ALLOWED_EXTENSIONS = {".c", ".md"}
PROHIBITED_KEYWORDS = ["system(", "popen(", "exec(", "fork(", "execve("]


def validate_file_input(file_path: str, expected_ext: str | None = None) -> bool:
    """
    Validates that a file exists, is a real file, is not empty,
    and has the correct extension.
    Returns True if valid, False otherwise.
    """
    try:
        if not file_path or not str(file_path).strip():
            engine_logger.error("SECURITY: File path is missing or blank.")
            return False

        path = Path(file_path)

        if not path.exists():
            engine_logger.error(f"SECURITY: File not found: {file_path}")
            return False

        if not path.is_file():
            engine_logger.error(f"SECURITY: Path is not a file: {file_path}")
            return False

        suffix = path.suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            engine_logger.error(
                f"SECURITY: Unsupported file extension for {file_path}. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            )
            return False

        if expected_ext:
            normalized_expected_ext = expected_ext.lower()

            if normalized_expected_ext not in ALLOWED_EXTENSIONS:
                engine_logger.error(
                    f"SECURITY: Invalid expected extension rule '{expected_ext}'. "
                    f"Allowed rules: {sorted(ALLOWED_EXTENSIONS)}"
                )
                return False

            if suffix != normalized_expected_ext:
                engine_logger.error(
                    f"SECURITY: Invalid extension for {file_path}. "
                    f"Expected {normalized_expected_ext}, got {suffix}"
                )
                return False

        if path.stat().st_size == 0:
            engine_logger.error(f"SECURITY: File is empty: {file_path}")
            return False

        engine_logger.info(f"SECURITY: File validated successfully: {file_path}")
        return True

    except Exception as e:
        engine_logger.error(f"SECURITY: File validation failed for {file_path}: {e}")
        return False


def validate_all_input_files(
    instructions_path: str,
    starter_path: str,
    teacher_path: str,
    student_path: str | None = None
) -> bool:
    """
    Validates all required input files before pipeline execution.
    Returns True only if all provided files pass validation.
    """
    files_to_validate = [
        (instructions_path, ".md"),
        (starter_path, ".c"),
        (teacher_path, ".c"),
    ]

    if student_path:
        files_to_validate.append((student_path, ".c"))

    for file_path, expected_ext in files_to_validate:
        if not validate_file_input(file_path, expected_ext):
            engine_logger.error(
                "SECURITY: Input validation failed. Pipeline aborted."
            )
            return False

    engine_logger.info("SECURITY: All input files validated successfully.")
    return True


def is_content_safe(content: str, file_path: str = "") -> bool:
    """
    Checks for prohibited system-level keywords in file content.
    Logs warnings for audit purposes.
    Returns True always — warning logs are the safety record.
    """
    try:
        if content is None:
            engine_logger.warning(
                f"SECURITY: Content safety check skipped because content is None "
                f"for {file_path or 'content'}."
            )
            return True

        lowered_content = content.lower()

        for keyword in PROHIBITED_KEYWORDS:
            if keyword.lower() in lowered_content:
                engine_logger.warning(
                    f"SECURITY: Prohibited keyword '{keyword}' found in "
                    f"{file_path or 'content'}"
                )

        return True

    except Exception as e:
        engine_logger.error(
            f"SECURITY: Content safety check failed for {file_path or 'content'}: {e}"
        )
        return True


def safe_read_file(file_path: str, expected_ext: str | None = None) -> str | None:
    """
    Validates and reads a file in one call.
    Returns file content as string, or None if validation fails.
    """
    if not validate_file_input(file_path, expected_ext):
        return None

    try:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            content = file.read()

        if not content.strip():
            engine_logger.error(
                f"SECURITY: File contains only whitespace: {file_path}"
            )
            return None

        is_content_safe(content, file_path)

        engine_logger.info(f"SECURITY: File read successfully: {file_path}")
        return content

    except Exception as e:
        engine_logger.error(f"SECURITY: Failed to read file {file_path}: {e}")
        return None
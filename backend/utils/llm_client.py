"""
backend/utils/llm_client.py
Shared LLM call wrapper with automatic rate limit retry.
All 3 agents use this instead of calling OpenAI client directly.

When provider returns rate limit or transient errors, this wrapper:
  1. Catches the error
  2. Waits automatically
  3. Retries the same call
  4. Returns clean content or None on final failure

Strips <think>...</think> tags from models that output chain-of-thought.
All agents receive clean output automatically.
"""

import re
import time

from openai import OpenAI

from backend.config.constants import LLM_MAX_RETRIES, LLM_RETRY_DELAY_SECONDS
from backend.utils.logger import engine_logger

# TODO:
# Move these to constants.py later for full Single Source of Truth compliance.
DEFAULT_LLM_TEMPERATURE = 0.1
MAX_ERROR_LOG_LENGTH = 200

THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(content: str | None) -> str:
    """
    Removes <think>...</think> blocks from LLM responses.
    Returns cleaned text. If content is None, returns empty string.
    """
    if content is None:
        return ""

    cleaned = THINK_TAG_PATTERN.sub("", content).strip()
    return cleaned


def _is_rate_limit_error(error_text: str) -> bool:
    """
    Detects rate limit style failures from provider error text.
    """
    lowered = error_text.lower()
    return (
        "429" in lowered
        or "rate limit" in lowered
        or "too many requests" in lowered
        or "rate_limit_exceeded" in lowered
        or "tokens per minute" in lowered
        or "requests per minute" in lowered
    )


def _is_transient_error(error_text: str) -> bool:
    """
    Detects transient failures that are worth retrying.
    """
    lowered = error_text.lower()
    return (
        "timeout" in lowered
        or "connection" in lowered
        or "503" in lowered
        or "502" in lowered
        or "server error" in lowered
        or "temporarily unavailable" in lowered
    )


def _normalize_temperature(temperature: float | None, agent_label: str) -> float:
    """
    Enforces low-temperature usage for consistency.
    Caps temperature at 0.1 if a higher value is passed.
    """
    if temperature is None:
        return DEFAULT_LLM_TEMPERATURE

    if temperature > DEFAULT_LLM_TEMPERATURE:
        engine_logger.warning(
            f"{agent_label}: Temperature {temperature} requested. "
            f"Capped to {DEFAULT_LLM_TEMPERATURE} for consistency."
        )
        return DEFAULT_LLM_TEMPERATURE

    if temperature < 0:
        engine_logger.warning(
            f"{agent_label}: Negative temperature {temperature} requested. "
            f"Reset to {DEFAULT_LLM_TEMPERATURE}."
        )
        return DEFAULT_LLM_TEMPERATURE

    return temperature


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    agent_label: str = "LLM"
) -> str | None:
    """
    Calls LLM with automatic retry on rate limit and transient errors.

    Returns:
        str  — clean response content on success
        None — on failure after all retries are exhausted

    Does NOT retry on:
        - Invalid API key
        - Model not found
        - Malformed request
    """
    if client is None:
        engine_logger.error(f"{agent_label}: LLM client is None.")
        return None

    if not model or not str(model).strip():
        engine_logger.error(f"{agent_label}: Model name is missing.")
        return None

    if not isinstance(messages, list) or not messages:
        engine_logger.error(f"{agent_label}: Messages must be a non-empty list.")
        return None

    normalized_temperature = _normalize_temperature(temperature, agent_label)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=normalized_temperature
            )

            if not response or not getattr(response, "choices", None):
                engine_logger.error(
                    f"{agent_label}: LLM response missing choices."
                )
                return None

            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)

            if message is None:
                engine_logger.error(
                    f"{agent_label}: LLM response missing message object."
                )
                return None

            content = getattr(message, "content", None)
            cleaned_content = strip_think_tags(content)

            if content and "<think>" in content:
                engine_logger.info(
                    f"{agent_label}: Stripped <think> tags from response "
                    f"({len(content)} -> {len(cleaned_content)} chars)."
                )

            if not cleaned_content:
                engine_logger.error(
                    f"{agent_label}: LLM returned empty content after cleaning."
                )
                return None

            return cleaned_content

        except Exception as e:
            error_text = str(e)
            is_rate_limit = _is_rate_limit_error(error_text)
            is_transient = _is_transient_error(error_text)

            if (is_rate_limit or is_transient) and attempt < LLM_MAX_RETRIES:
                wait_time = LLM_RETRY_DELAY_SECONDS * attempt
                engine_logger.warning(
                    f"{agent_label}: "
                    f"{'Rate limited' if is_rate_limit else 'Transient error'}. "
                    f"Waiting {wait_time}s before retry "
                    f"({attempt}/{LLM_MAX_RETRIES}). "
                    f"Error: {error_text[:MAX_ERROR_LOG_LENGTH]}"
                )
                time.sleep(wait_time)
                continue

            engine_logger.error(
                f"{agent_label}: LLM call failed after "
                f"{attempt} attempt(s): {error_text}"
            )
            return None

    return None
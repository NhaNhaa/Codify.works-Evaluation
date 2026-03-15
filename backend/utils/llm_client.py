"""
backend/utils/llm_client.py
Shared LLM call wrapper with automatic rate limit retry.
All 3 agents use this instead of calling OpenAI client directly.

When Groq/Fireworks returns 429 (rate limit), this wrapper:
  1. Catches the error
  2. Waits automatically (20s × attempt number)
  3. Retries the same call
  4. Pipeline continues — nothing crashes, nothing lost

Strips <think>...</think> tags from models that output chain-of-thought
(e.g. qwen3-32b). All agents receive clean output automatically.

Zero quality compromise — same calls, same accuracy, just doesn't crash.
"""

import re
import time
from openai import OpenAI
from backend.config.constants import LLM_MAX_RETRIES, LLM_RETRY_DELAY_SECONDS
from backend.utils.logger import engine_logger

# ── THINK TAG PATTERN ──────────────────────────────────────────────
# Matches <think>...</think> blocks including newlines.
# Some models (e.g. qwen3-32b) output chain-of-thought in these tags.
# Must be stripped before JSON parsing or feedback processing.
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(content: str) -> str:
    """
    Removes <think>...</think> blocks from LLM responses.
    Returns the remaining content stripped of leading/trailing whitespace.
    If response is ONLY think tags with nothing after, returns empty string.
    """
    cleaned = THINK_TAG_PATTERN.sub("", content).strip()
    return cleaned


def call_llm_with_retry(
    client:      OpenAI,
    model:       str,
    messages:    list[dict],
    temperature: float = 0.1,
    agent_label: str = "LLM"
) -> str | None:
    """
    Calls LLM with automatic retry on rate limit (429) errors.

    Returns:
        str  — response content on success (think tags stripped)
        None — on failure after all retries exhausted

    Handles:
        - 429 Too Many Requests (rate limit)
        - Connection timeouts
        - Transient API errors
        - <think>...</think> tag stripping

    Does NOT retry on:
        - Invalid API key (401)
        - Model not found (404)
        - Malformed request (400)
    """
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content

            # Strip <think> tags if present
            if content and "<think>" in content:
                original_len = len(content)
                content = strip_think_tags(content)
                engine_logger.info(
                    f"{agent_label}: Stripped <think> tags from response "
                    f"({original_len} → {len(content)} chars)."
                )

            return content

        except Exception as e:
            error_str = str(e).lower()

            # Detect rate limit errors
            is_rate_limit = (
                "429" in error_str
                or "rate limit" in error_str
                or "too many requests" in error_str
                or "rate_limit_exceeded" in error_str
                or "tokens per minute" in error_str
                or "requests per minute" in error_str
            )

            # Detect transient errors worth retrying
            is_transient = (
                "timeout" in error_str
                or "connection" in error_str
                or "503" in error_str
                or "502" in error_str
                or "server error" in error_str
            )

            if (is_rate_limit or is_transient) and attempt < LLM_MAX_RETRIES:
                wait_time = LLM_RETRY_DELAY_SECONDS * attempt
                engine_logger.warning(
                    f"{agent_label}: {'Rate limited' if is_rate_limit else 'Transient error'}. "
                    f"Waiting {wait_time}s before retry "
                    f"({attempt}/{LLM_MAX_RETRIES})... "
                    f"Error: {str(e)[:200]}"
                )
                time.sleep(wait_time)
            else:
                engine_logger.error(
                    f"{agent_label}: LLM call failed after "
                    f"{attempt} attempt(s): {e}"
                )
                return None

    return None
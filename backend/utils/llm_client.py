"""
backend/utils/llm_client.py
Shared LLM call wrapper with automatic rate limit retry.
All 3 agents use this instead of calling OpenAI client directly.

When provider returns rate limit or transient errors, this wrapper:
  1. Catches the error
  2. Waits automatically
  3. Retries the same call
  4. Returns clean content or None on final failure

No thinking‑tag stripping – all selected providers (Groq kimi‑k2, Cerebras,
Mistral, Zhipu, Cloudflare) produce clean JSON without reasoning blocks.
"""

import re
import time

from openai import OpenAI

from backend.config import config
from backend.utils.logger import engine_logger


def _is_rate_limit_error(error_text: str) -> bool:
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
    lowered = error_text.lower()
    return (
        "timeout" in lowered
        or "connection" in lowered
        or "503" in lowered
        or "502" in lowered
        or "server error" in lowered
        or "temporarily unavailable" in lowered
    )


def _is_unsupported_parameter_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "unexpected keyword argument" in lowered
        or "unsupported parameter" in lowered
        or "unknown parameter" in lowered
        or "unrecognized request argument" in lowered
        or "extra inputs are not permitted" in lowered
        or "extra_forbidden" in lowered
        or "reasoning_format" in lowered
    )


def _normalize_temperature(temperature: float | None, agent_label: str) -> float:
    """
    Returns a safe temperature value.
    - If None, uses the global default from config.
    - Ensures value is between 0.0 and 2.0 (clamped).
    - Logs a warning if out of range.
    """
    default_temp = config.LLM_TEMPERATURE

    if temperature is None:
        return default_temp

    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        engine_logger.warning(
            f"{agent_label}: Invalid temperature {temperature}. Using default {default_temp}."
        )
        return default_temp

    if temp < 0.0:
        engine_logger.warning(
            f"{agent_label}: Temperature {temp} < 0. Clamping to 0.0."
        )
        return 0.0
    if temp > 2.0:
        engine_logger.warning(
            f"{agent_label}: Temperature {temp} > 2.0. Clamping to 2.0."
        )
        return 2.0

    return temp


def _validate_messages(messages: list[dict], agent_label: str) -> bool:
    if not isinstance(messages, list) or not messages:
        engine_logger.error(f"{agent_label}: Messages must be a non-empty list.")
        return False

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            engine_logger.error(f"{agent_label}: Message at index {index} is not a dict.")
            return False

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not role.strip():
            engine_logger.error(f"{agent_label}: Message role is missing or blank at index {index}.")
            return False

        if content is None:
            engine_logger.error(f"{agent_label}: Message content is missing at index {index}.")
            return False

        if isinstance(content, str) and not content.strip():
            engine_logger.error(f"{agent_label}: Message content is blank at index {index}.")
            return False

    return True


def _messages_expect_json(messages: list[dict]) -> bool:
    combined_text_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            combined_text_parts.append(content)
        else:
            combined_text_parts.append(str(content))
    combined_text = " ".join(combined_text_parts).lower()

    json_signals = [
        "valid json",
        "json only",
        "return only json",
        "return only a valid json",
        "no markdown fences",
        "json object",
        "json list",
        "return only with a valid json object",
        "answer only with a valid json object",
        "json",                # added to catch simple "Return JSON" prompts
    ]
    return any(signal in combined_text for signal in json_signals)


def _normalize_response_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                    continue
                if isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
                    continue
                text_parts.append(str(item))
                continue
            item_text = getattr(item, "text", None)
            if isinstance(item_text, str):
                text_parts.append(item_text)
            else:
                text_parts.append(str(item))
        return "".join(text_parts)
    return str(content)


def _extract_json_candidate_from_text(text: str) -> str:
    """
    Extracts the first valid-looking top-level JSON object or list from noisy text.
    Handles markdown code fences and surrounding text.
    Returns the original text if no JSON candidate is found.
    """
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped:
        return ""

    # If the whole text is already a JSON object/array, return it
    if (stripped.startswith("{") and stripped.endswith("}")) or \
       (stripped.startswith("[") and stripped.endswith("]")):
        return stripped

    # Try to extract JSON from markdown code blocks first
    # Match ```json ... ``` or ``` ... ``` with potential whitespace
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(code_block_pattern, stripped, re.DOTALL | re.IGNORECASE)
    for match in matches:
        candidate = match.strip()
        if (candidate.startswith("{") and candidate.endswith("}")) or \
           (candidate.startswith("[") and candidate.endswith("]")):
            return candidate

    # If no code block, try to find the first balanced JSON object/array
    start_positions = [i for i, ch in enumerate(stripped) if ch in "{["]
    for start in start_positions:
        opening = stripped[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:i+1].strip()
                    if candidate:
                        return candidate
    return stripped


def _build_request_variants(
    model: str,
    messages: list[dict],
    temperature: float,
    response_format: dict | None = None,
) -> list[dict]:
    """
    Builds request variants from most capable to most compatible.
    First attempt includes the provided response_format (if any).
    Falls back to a plain request without it.
    """
    base_request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    variant1 = dict(base_request)
    if response_format:
        variant1["response_format"] = response_format

    variant2 = dict(base_request)

    return [variant1, variant2]


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    response_format: dict | None = None,
    agent_label: str = "LLM",
) -> str | None:
    """
    Calls LLM with automatic retry on rate limit and transient errors.

    Args:
        client: OpenAI client instance.
        model: Model name string.
        messages: Chat messages list.
        temperature: Override global temperature (optional).
        response_format: Optional dict like {"type": "json_object"}.
        agent_label: For logging.

    Returns:
        str — clean response content on success
        None — on failure after all retries are exhausted
    """
    if client is None:
        engine_logger.error(f"{agent_label}: LLM client is None.")
        return None

    if not model or not str(model).strip():
        engine_logger.error(f"{agent_label}: Model name is missing.")
        return None

    if not _validate_messages(messages, agent_label):
        return None

    normalized_temperature = _normalize_temperature(temperature, agent_label)
    expects_json = _messages_expect_json(messages)

    # Use provided response_format if any, otherwise decide based on config and prompt
    effective_response_format = response_format
    if effective_response_format is None and expects_json:
        # If the prompt asks for JSON and the environment enables json_object, set it
        if config.LLM_RESPONSE_FORMAT == "json_object":
            effective_response_format = {"type": "json_object"}

    retries = config.LLM_MAX_RETRIES
    delay = config.LLM_RETRY_DELAY_SECONDS

    for attempt in range(1, retries + 1):
        try:
            response = None
            last_variant_error = None

            for request_kwargs in _build_request_variants(
                model=model,
                messages=messages,
                temperature=normalized_temperature,
                response_format=effective_response_format,
            ):
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except Exception as variant_exc:
                    error_text = str(variant_exc)
                    if _is_unsupported_parameter_error(error_text):
                        last_variant_error = variant_exc
                        engine_logger.info(
                            f"{agent_label}: Falling back to a simpler LLM request "
                            f"because the current model/provider rejected an optional parameter. "
                            f"Error: {error_text[:200]}"
                        )
                        continue
                    raise

            if response is None:
                if last_variant_error is not None:
                    raise last_variant_error
                engine_logger.error(f"{agent_label}: LLM response is missing.")
                return None

            if not getattr(response, "choices", None):
                engine_logger.error(f"{agent_label}: LLM response missing choices.")
                return None

            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)
            if message is None:
                engine_logger.error(f"{agent_label}: LLM response missing message object.")
                return None

            raw_content = _normalize_response_content(getattr(message, "content", None))
            final_content = raw_content  # No think tags to strip

            # If we expected JSON, attempt extraction
            if expects_json:
                extracted = _extract_json_candidate_from_text(raw_content)
                if extracted != raw_content:
                    engine_logger.info(
                        f"{agent_label}: Extracted JSON candidate from noisy response "
                        f"({len(raw_content)} -> {len(extracted)} chars)."
                    )
                    final_content = extracted

            # After all processing, check for empty/whitespace content
            if not final_content or not final_content.strip():
                engine_logger.error(f"{agent_label}: LLM returned empty content after cleaning.")
                return None

            return final_content

        except Exception as exc:
            error_text = str(exc)
            is_rate_limit = _is_rate_limit_error(error_text)
            is_transient = _is_transient_error(error_text)

            if (is_rate_limit or is_transient) and attempt < retries:
                wait_time = delay * attempt
                engine_logger.warning(
                    f"{agent_label}: "
                    f"{'Rate limited' if is_rate_limit else 'Transient error'}. "
                    f"Waiting {wait_time}s before retry ({attempt}/{retries}). "
                    f"Error: {error_text[:200]}"
                )
                time.sleep(wait_time)
                continue

            engine_logger.error(
                f"{agent_label}: LLM call failed after {attempt} attempt(s): {error_text[:200]}"
            )
            return None

    return None
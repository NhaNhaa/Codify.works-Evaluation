"""
backend/tests/test_llm_quota.py
Tests all 11 API slots from the Free API Rotation Strategy.
Run: python -m backend.tests.test_llm_quota

Tests each slot independently:
  - Validates API key exists
  - Sends one minimal LLM call
  - Reports success/failure per slot
  - Shows which slots are ready for rotation

Does NOT run as pytest — this is a manual diagnostic tool.
"""

import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── ALL 11 SLOTS ───────────────────────────────────────────────────
# Each slot: (slot_number, provider_name, base_url, api_key_env, model, rpm_info)
SLOTS = [
    (1,  "Groq A",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "moonshotai/kimi-k2-instruct",                "60 RPM"),
    (2,  "Groq B",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "qwen/qwen3-32b",                             "60 RPM"),
    (3,  "Groq C",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "llama-3.3-70b-versatile",                    "30 RPM"),
    (4,  "Groq D",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "openai/gpt-oss-120b",                        "30 RPM"),
    (5,  "Groq E",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "openai/gpt-oss-20b",                         "30 RPM"),
    (6,  "Groq F",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "meta-llama/llama-4-scout-17b-16e-instruct",  "30 RPM"),
    (7,  "Groq G",     "https://api.groq.com/openai/v1",      "GROQ_API_KEY",       "llama-3.1-8b-instant",                       "30 RPM / 14.4K RPD"),
    (8,  "Cerebras",   "https://api.cerebras.ai/v1",           "CEREBRAS_API_KEY",   "qwen-3-235b-a22b-instruct-2507",            "~1M tokens/day"),
    (9,  "SambaNova",  "https://api.sambanova.ai/v1",          "SAMBANOVA_API_KEY",  "Meta-Llama-3.3-70B-Instruct",                "20 RPM"),
    (10, "StepFun",    "https://api.stepfun.ai/v1",            "STEPFUN_API_KEY",    "step-3.5-flash",                             "Own free pool"),
    (11, "OpenRouter", "https://openrouter.ai/api/v1",         "OPENROUTER_API_KEY", "arcee-ai/trinity-large-preview:free",         "50 RPD shared"),
]

# ── MINIMAL TEST PROMPT ────────────────────────────────────────────
TEST_MESSAGES = [
    {"role": "user", "content": "Reply with exactly: SLOT_OK"}
]

# ── DISPLAY CONSTANTS ──────────────────────────────────────────────
PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"
SKIP_SYMBOL = "⏭️"
SEPARATOR   = "─" * 70


def test_single_slot(slot_num, provider, base_url, api_key_env, model, rpm_info):
    """
    Tests one API slot. Returns (slot_num, status, detail).
    status: "PASS", "FAIL", "SKIP"
    """
    api_key = os.getenv(api_key_env)

    # Check API key exists
    if not api_key or api_key.startswith("your_"):
        return (slot_num, "SKIP", f"No API key set for {api_key_env}")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=TEST_MESSAGES,
            temperature=0.1,
            max_tokens=10
        )

        elapsed = time.time() - start_time
        content = response.choices[0].message.content.strip()

        return (
            slot_num,
            "PASS",
            f"Response: '{content[:50]}' | {elapsed:.1f}s"
        )

    except Exception as e:
        error_str = str(e)[:150]

        # Detect rate limit specifically
        if "429" in error_str or "rate limit" in error_str.lower():
            return (slot_num, "FAIL", f"Rate limited (429) — window cooling: {error_str}")

        # Detect auth errors
        if "401" in error_str or "403" in error_str or "invalid" in error_str.lower():
            return (slot_num, "FAIL", f"Auth error — check {api_key_env}: {error_str}")

        # Detect model not found
        if "404" in error_str or "not found" in error_str.lower():
            return (slot_num, "FAIL", f"Model not found — may be deprecated: {error_str}")

        return (slot_num, "FAIL", f"Error: {error_str}")


def run_all_slots():
    """Tests all 11 slots and prints a summary report."""
    print()
    print(SEPARATOR)
    print("  AutoEval-C — Free API Rotation: 11-Slot Diagnostic")
    print(SEPARATOR)
    print()

    results = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for slot_num, provider, base_url, api_key_env, model, rpm_info in SLOTS:
        # Print header
        print(f"  Slot {slot_num:>2} | {provider:<12} | {model}")
        print(f"         | {rpm_info}")

        # Run test
        _, status, detail = test_single_slot(
            slot_num, provider, base_url, api_key_env, model, rpm_info
        )

        # Print result
        if status == "PASS":
            symbol = PASS_SYMBOL
            pass_count += 1
        elif status == "FAIL":
            symbol = FAIL_SYMBOL
            fail_count += 1
        else:
            symbol = SKIP_SYMBOL
            skip_count += 1

        print(f"         | {symbol} {status}: {detail}")
        print()

        results.append((slot_num, provider, model, status, detail))

        # Small delay between slots to avoid triggering rate limits
        if status == "PASS" and slot_num < len(SLOTS):
            time.sleep(2)

    # ── SUMMARY ────────────────────────────────────────────────────
    print(SEPARATOR)
    print("  SUMMARY")
    print(SEPARATOR)
    print()

    # Ready slots
    ready = [r for r in results if r[3] == "PASS"]
    failed = [r for r in results if r[3] == "FAIL"]
    skipped = [r for r in results if r[3] == "SKIP"]

    print(f"  {PASS_SYMBOL} Ready:   {pass_count} slots")
    print(f"  {FAIL_SYMBOL} Failed:  {fail_count} slots")
    print(f"  {SKIP_SYMBOL} Skipped: {skip_count} slots (no API key)")
    print()

    if ready:
        print("  Ready for rotation:")
        for slot_num, provider, model, _, _ in ready:
            print(f"    Slot {slot_num:>2} — {provider:<12} ({model})")
        print()

    if failed:
        print("  Failed (check keys/models):")
        for slot_num, provider, model, _, detail in failed:
            print(f"    Slot {slot_num:>2} — {provider:<12}: {detail[:80]}")
        print()

    if skipped:
        print("  Skipped (add API key to .env):")
        for slot_num, provider, model, _, _ in skipped:
            print(f"    Slot {slot_num:>2} — {provider:<12}")
        print()

    # ── ROTATION READINESS ─────────────────────────────────────────
    groq_ready = [r for r in ready if r[1].startswith("Groq")]
    non_groq_ready = [r for r in ready if not r[1].startswith("Groq")]

    print(SEPARATOR)
    print("  ROTATION READINESS")
    print(SEPARATOR)
    print()

    if len(groq_ready) >= 3:
        print(f"  {PASS_SYMBOL} Groq rotation: {len(groq_ready)}/7 slots ready — sufficient for debugging")
    elif len(groq_ready) >= 1:
        print(f"  ⚠️  Groq rotation: {len(groq_ready)}/7 slots ready — limited, add more models")
    else:
        print(f"  {FAIL_SYMBOL} Groq rotation: 0/7 slots ready — set GROQ_API_KEY in .env")

    if non_groq_ready:
        print(f"  {PASS_SYMBOL} Backup providers: {len(non_groq_ready)} ready")
    else:
        print(f"  ⏭️  Backup providers: none configured (optional)")

    print()
    print(SEPARATOR)
    print()


# ── ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_slots()
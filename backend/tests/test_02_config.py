import os
from backend.config.config import (
    LLM_PROVIDER, 
    PROVIDERS, 
    get_provider_config, 
    get_model,
    LOG_FILE_PATH
)

def run_test():
    print("--- 🔍 STEP 2: Testing config.py ---")
    errors = 0

    # 1. Provider Logic Check
    try:
        current_config = get_provider_config()
        if current_config['api_key'] is not None:
             print(f"✅ PASS: Active Provider ({LLM_PROVIDER}) has an API Key loaded.")
        else:
             print(f"⚠️ WARNING: Provider ({LLM_PROVIDER}) found, but API Key is None. Check your .env!")
    except KeyError:
        print(f"❌ FAIL: LLM_PROVIDER '{LLM_PROVIDER}' not found in PROVIDERS dict.")
        errors += 1

    # 2. Agent Model Slot Check
    test_agent = "agent1_skill_extractor"
    model_name = get_model(test_agent)
    if isinstance(model_name, str) and len(model_name) > 0:
        print(f"✅ PASS: Model slot for '{test_agent}' returned: {model_name}")
    else:
        print(f"❌ FAIL: Model retrieval for '{test_agent}' failed.")
        errors += 1

    # 3. Path Normalization Check
    if "logs" in LOG_FILE_PATH and "engine.log" in LOG_FILE_PATH:
        print(f"✅ PASS: Log path is correctly structured: {LOG_FILE_PATH}")
    else:
        print(f"❌ FAIL: Log path seems incorrect.")
        errors += 1

    if errors == 0:
        print("\n🏆 RESULT: Step 2 configuration loading is 100% operational.")
    else:
        print(f"\n⚠️ RESULT: Step 2 failed with {errors} errors.")

if __name__ == "__main__":
    run_test()
import os
from backend.utils.logger import engine_logger
from backend.config.config import LOG_FILE_PATH

def run_test():
    print("--- 🔍 STEP 3: Testing logger.py ---")
    test_message = "LOGGER TEST: Verification of UTF-8 and Persistence 🚀"
    
    # 1. Trigger a log entry
    engine_logger.info(test_message)
    
    # 2. Check if file exists
    if os.path.exists(LOG_FILE_PATH):
        print(f"✅ PASS: Log file created at {LOG_FILE_PATH}")
    else:
        print(f"❌ FAIL: Log file was not created.")
        return

    # 3. Check for content persistence and encoding
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        if test_message in content:
            print("✅ PASS: Test message successfully written to file.")
        else:
            print("❌ FAIL: Test message not found in log file.")

    # 4. Check for log level formatting
    if "| INFO     |" in content:
        print("✅ PASS: Log level formatting is correct (8-character padding).")
    else:
        print("❌ FAIL: Formatting mismatch in log file.")

if __name__ == "__main__":
    run_test()
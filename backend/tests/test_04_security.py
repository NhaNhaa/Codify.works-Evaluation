import os
from backend.utils.security import validate_file_input, safe_read_file

def run_test():
    print("--- 🔍 STEP 4: Testing security.py ---")
    
    # Setup test files
    test_dir = "data/tests"
    os.makedirs(test_dir, exist_ok=True)
    
    empty_file = os.path.join(test_dir, "empty.c")
    valid_file = os.path.join(test_dir, "valid.c")
    wrong_ext = os.path.join(test_dir, "wrong.txt")
    
    with open(empty_file, "w") as f: pass
    with open(valid_file, "w") as f: f.write("int main() { return 0; }")
    with open(wrong_ext, "w") as f: f.write("some text")

    # 1. Test: Missing File
    res1 = validate_file_input("non_existent.c")
    print(f"{'✅' if res1 == False else '❌'} PASS: Corrected blocked missing file.")

    # 2. Test: Empty File
    res2 = validate_file_input(empty_file)
    print(f"{'✅' if res2 == False else '❌'} PASS: Corrected blocked empty file.")

    # 3. Test: Wrong Extension
    res3 = validate_file_input(wrong_ext, expected_ext=".c")
    print(f"{'✅' if res3 == False else '❌'} PASS: Corrected blocked wrong extension.")

    # 4. Test: Valid Read
    content = safe_read_file(valid_file, expected_ext=".c")
    if content == "int main() { return 0; }":
        print("✅ PASS: Valid file read successfully.")
    else:
        print("❌ FAIL: Valid file read failed.")

if __name__ == "__main__":
    run_test()
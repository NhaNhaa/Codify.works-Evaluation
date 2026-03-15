from backend.config.constants import (
    TOTAL_WEIGHT_TARGET, 
    DEFAULT_WEIGHT_DISTRIBUTIONS,
    MIN_MICRO_SKILLS, 
    MAX_MICRO_SKILLS,
    RANKING_CRITERIA
)

def run_test():
    print("--- 🔍 STEP 1: Testing constants.py ---")
    errors = 0

    # 1. Math Check: Do all weight distributions sum to 10?
    for count, weights in DEFAULT_WEIGHT_DISTRIBUTIONS.items():
        current_sum = sum(weights)
        if current_sum == TOTAL_WEIGHT_TARGET:
            print(f"✅ PASS: Count {count} sums to {TOTAL_WEIGHT_TARGET}")
        else:
            print(f"❌ FAIL: Count {count} sums to {current_sum} (Expected {TOTAL_WEIGHT_TARGET})")
            errors += 1

    # 2. Structure Check: Does each list match its key? (e.g., 4 skills should have 4 weights)
    for count, weights in DEFAULT_WEIGHT_DISTRIBUTIONS.items():
        if len(weights) == count:
            print(f"✅ PASS: Count {count} has exactly {count} weight entries.")
        else:
            print(f"❌ FAIL: Count {count} has {len(weights)} entries instead of {count}.")
            errors += 1

    # 3. Boundary Check
    if MIN_MICRO_SKILLS < MAX_MICRO_SKILLS:
        print(f"✅ PASS: Boundaries are logical ({MIN_MICRO_SKILLS} to {MAX_MICRO_SKILLS})")
    else:
        print(f"❌ FAIL: MIN_MICRO_SKILLS must be less than MAX_MICRO_SKILLS")
        errors += 1

    # 4. Criteria Check
    if isinstance(RANKING_CRITERIA, list) and len(RANKING_CRITERIA) >= 3:
        print(f"✅ PASS: Ranking criteria is a robust list.")
    else:
        print("❌ FAIL: Ranking criteria is missing or too short.")
        errors += 1

    if errors == 0:
        print("\n🏆 RESULT: Step 1 constants are valid and logically sound.")
    else:
        print(f"\n⚠️ RESULT: Step 1 failed with {errors} logic errors.")

if __name__ == "__main__":
    run_test()
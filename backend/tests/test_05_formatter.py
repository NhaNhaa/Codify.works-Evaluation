from backend.utils.formatter import build_output

def run_test():
    print("--- 🔍 STEP 5: Testing formatter.py ---")
    
    # Mock JSON Data: One High Weight FAIL, One Low Weight PASS
    mock_data = {
        "student_id": "STUDENT_001",
        "assignment_id": "LAB_01",
        "skills": [
            {
                "skill": "Array Boundary Check",
                "rank": 1,
                "weight": 5, # HIGH WEIGHT
                "status": "FAIL",
                "line_start": 10,
                "line_end": 12,
                "student_snippet": "for(int i=0; i<=5; i++)",
                "feedback": "Index out of bounds at i=5",
                "recommended_fix": "for(int i=0; i<5; i++)"
            },
            {
                "skill": "Comments",
                "rank": 2,
                "weight": 1, # LOW WEIGHT
                "status": "PASS",
                "line_start": 1,
                "line_end": 1,
                "student_snippet": "// Main function",
                "feedback": "Good documentation."
            }
        ]
    }

    output = build_output(mock_data)
    markdown = output["markdown"]

    # 1. Check for High Weight Structure (WHY IT IS WRONG should be present)
    if "**WHY IT IS WRONG:**" in markdown:
        print("✅ PASS: High weight FAIL includes 'WHY' section.")
    else:
        print("❌ FAIL: High weight FAIL missing 'WHY' section.")

    # 2. Check for Sorting (Rank 1 should appear before Rank 2 because weight 5 > 1)
    if markdown.find("Array Boundary Check") < markdown.find("Comments"):
        print("✅ PASS: Skills sorted by weight (descending).")
    else:
        print("❌ FAIL: Skills not sorted correctly.")

    # 3. Check for Code Blocks
    if "```c" in markdown:
        print("✅ PASS: C-code blocks formatted correctly.")
    else:
        print("❌ FAIL: Markdown missing code blocks.")

if __name__ == "__main__":
    run_test()
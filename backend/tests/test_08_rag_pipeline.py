from backend.rag.rag_pipeline import rag_pipeline

def run_test():
    print("--- 🔍 STEP 8: Testing rag_pipeline.py ---")
    
    test_id = "INTEGRATION_LAB_1"
    mock_skills = [
        {"text": "Pointer dereferencing", "rank": 1, "weight": 6},
        {"text": "Null check", "rank": 2, "weight": 4}
    ]
    
    # 1. Clear any existing test data
    rag_pipeline.clear_assignment(test_id)

    # 2. Test Phase 1: Store Micro Skills (Triggers Embedder + Chroma)
    success = rag_pipeline.store_micro_skills(mock_skills, test_id)
    if success:
        print("✅ PASS: Pipeline successfully coordinated embedding and storage.")
    else:
        print("❌ FAIL: Pipeline storage coordination failed.")

    # 3. Test Phase 2: Retrieval for Agent 2
    retrieved = rag_pipeline.retrieve_micro_skills(test_id)
    if len(retrieved) == 2 and "Pointer" in retrieved[0]["text"]:
        print(f"✅ PASS: Pipeline retrieved sorted skills correctly.")
    else:
        print("❌ FAIL: Pipeline retrieval failed or data corrupted.")

    # 4. Test Exist Check
    if rag_pipeline.assignment_exists(test_id):
        print("✅ PASS: assignment_exists() correctly identified the record.")
    else:
        print("❌ FAIL: assignment_exists() failed.")

if __name__ == "__main__":
    run_test()
from backend.rag.chroma_client import chroma_client

def run_test():
    print("--- 🔍 STEP 7: Testing chroma_client.py ---")
    
    test_id = "TEST_LAB_99"
    mock_skills = [
        {"text": "Skill A", "rank": 1, "weight": 5},
        {"text": "Skill B", "rank": 2, "weight": 5}
    ]
    mock_embeddings = [[0.1] * 384, [0.2] * 384] # Mock vectors

    # 1. Test: Weight Sum Guard (Should FAIL if sum != 10)
    bad_skills = [{"text": "Skill A", "weight": 1}]
    res_bad = chroma_client.store_micro_skills(bad_skills, test_id, [[0.1]*384])
    print(f"{'✅' if res_bad == False else '❌'} PASS: Correctly blocked invalid weight sum.")

    # 2. Test: Initial Store (Should PASS)
    # Clear first to ensure fresh state
    chroma_client.clear_assignment(test_id)
    res_store = chroma_client.store_micro_skills(mock_skills, test_id, mock_embeddings)
    print(f"{'✅' if res_store == True else '❌'} PASS: Initial skill storage successful.")

    # 3. Test: Overwrite Guard (Should FAIL without force flag)
    res_overwrite = chroma_client.store_micro_skills(mock_skills, test_id, mock_embeddings)
    print(f"{'✅' if res_overwrite == False else '❌'} PASS: Overwrite guard blocked silent update.")

    # 4. Test: Retrieval and Rank Integrity
    retrieved = chroma_client.retrieve_micro_skills(test_id)
    if len(retrieved) == 2 and retrieved[0]['rank'] == 1:
        print("✅ PASS: Retrieval successful and sorted by rank.")
    else:
        print(f"❌ FAIL: Retrieval issues. Count: {len(retrieved)}")

if __name__ == "__main__":
    run_test()
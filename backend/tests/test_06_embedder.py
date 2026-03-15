import numpy as np
from backend.rag.embedder import embedder

def run_test():
    print("--- 🔍 STEP 6: Testing embedder.py ---")
    
    # 1. Test Single Embedding
    text = "Implement a for-loop to iterate through an array."
    vector = embedder.embed_single(text)
    
    if isinstance(vector, list) and len(vector) > 0:
        print(f"✅ PASS: Single embedding generated. Dimensions: {len(vector)}")
    else:
        print("❌ FAIL: Single embedding failed or returned empty.")
        return

    # 2. Test Batch Embedding
    texts = ["Variable declaration", "Pointer arithmetic", "Memory allocation"]
    vectors = embedder.embed_texts(texts)
    
    if len(vectors) == 3:
        print(f"✅ PASS: Batch embedding generated {len(vectors)} vectors.")
    else:
        print(f"❌ FAIL: Batch embedding count mismatch (Got {len(vectors)}).")

    # 3. Semantic Similarity Check (The "Skeptical Engineer" Check)
    # Vectors for similar phrases should be closer than unrelated ones
    v1 = np.array(embedder.embed_single("How to use a loop"))
    v2 = np.array(embedder.embed_single("Looping structures in C"))
    v3 = np.array(embedder.embed_single("Setting up a database")) # Unrelated

    dist_similar = np.linalg.norm(v1 - v2)
    dist_different = np.linalg.norm(v1 - v3)

    if dist_similar < dist_different:
        print(f"✅ PASS: Semantic similarity confirmed. (Dist: {dist_similar:.4f} < {dist_different:.4f})")
    else:
        print("❌ FAIL: Embedder lacks semantic nuance.")

if __name__ == "__main__":
    run_test()
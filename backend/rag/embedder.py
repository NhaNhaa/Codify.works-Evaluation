"""
backend/rag/embedder.py
Handles all text-to-vector embedding operations.
Uses local sentence-transformers model — zero API cost.
"""

from sentence_transformers import SentenceTransformer
from backend.config.config import EMBEDDING_MODEL_NAME
from backend.utils.logger import engine_logger


class Embedder:
    """
    Converts text into vector embeddings using a local model.
    Loaded once and reused — no repeated initialization cost.
    """

    def __init__(self):
        engine_logger.info(
            f"EMBEDDER: Loading embedding model: {EMBEDDING_MODEL_NAME}"
        )
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            engine_logger.info("EMBEDDER: Model loaded successfully.")
        except Exception as e:
            engine_logger.error(f"EMBEDDER: Failed to load model: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of texts into a list of embedding vectors.
        Used for storing multiple micro skills at once.
        """
        if not texts:
            engine_logger.warning("EMBEDDER: Empty text list received.")
            return []

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            engine_logger.info(
                f"EMBEDDER: Embedded {len(texts)} text(s) successfully."
            )
            return embeddings.tolist()
        except Exception as e:
            engine_logger.error(f"EMBEDDER: Failed to embed texts: {e}")
            raise

    def embed_single(self, text: str) -> list[float]:
        """
        Converts a single text into an embedding vector.
        Used for query-time retrieval.
        """
        if not text or not text.strip():
            engine_logger.warning("EMBEDDER: Empty text received for single embed.")
            return []

        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding[0].tolist()
        except Exception as e:
            engine_logger.error(f"EMBEDDER: Failed to embed single text: {e}")
            raise


# ── GLOBAL INSTANCE ────────────────────────────────────────────────
# Loaded once at startup — shared across all agents via rag_pipeline
embedder = Embedder()
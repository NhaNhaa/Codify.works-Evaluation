"""
backend/rag/embedder.py
Handles all text-to-vector embedding operations.
Uses local sentence-transformers model — zero API cost.
Lazy initialization avoids import-time crashes.
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
        self.model = None

    def _ensure_model_loaded(self) -> None:
        """
        Lazily loads the embedding model only when first needed.
        """
        if self.model is not None:
            return

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
        if not isinstance(texts, list):
            engine_logger.error("EMBEDDER: texts must be a list.")
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not cleaned_texts:
            engine_logger.warning("EMBEDDER: Empty or invalid text list received.")
            return []

        try:
            self._ensure_model_loaded()

            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            engine_logger.info(
                f"EMBEDDER: Embedded {len(cleaned_texts)} text(s) successfully."
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
        if not isinstance(text, str) or not text.strip():
            engine_logger.warning("EMBEDDER: Empty or invalid text received for single embed.")
            return []

        try:
            self._ensure_model_loaded()

            embedding = self.model.encode(
                [text.strip()],
                convert_to_numpy=True,
                show_progress_bar=False
            )

            engine_logger.info("EMBEDDER: Embedded 1 text successfully.")
            return embedding[0].tolist()

        except Exception as e:
            engine_logger.error(f"EMBEDDER: Failed to embed single text: {e}")
            raise


# ── GLOBAL SINGLETON ACCESSOR ──────────────────────────────────────
_embedder_instance = None


def get_embedder() -> Embedder:
    """
    Returns the shared Embedder singleton with lazy initialization.
    """
    global _embedder_instance

    if _embedder_instance is None:
        _embedder_instance = Embedder()

    return _embedder_instance
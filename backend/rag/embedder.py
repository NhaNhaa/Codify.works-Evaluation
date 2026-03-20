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
        except Exception as exc:
            engine_logger.error(f"EMBEDDER: Failed to load model: {exc}")
            raise

    def _normalize_text(self, text: str) -> str | None:
        """
        Normalizes one text input.
        Returns cleaned text or None if invalid.
        """
        if not isinstance(text, str):
            return None

        cleaned_text = text.strip()
        if not cleaned_text:
            return None

        return cleaned_text

    def _validate_text_list(self, texts: list[str]) -> list[str]:
        """
        Validates a full text list without changing its intended alignment.
        Returns cleaned texts only if every entry is valid.
        """
        if not isinstance(texts, list):
            engine_logger.error("EMBEDDER: texts must be a list.")
            return []

        cleaned_texts = []

        for index, text in enumerate(texts):
            cleaned_text = self._normalize_text(text)
            if cleaned_text is None:
                engine_logger.error(
                    f"EMBEDDER: Invalid text at index {index}. "
                    "All items must be non-empty strings."
                )
                return []

            cleaned_texts.append(cleaned_text)

        if not cleaned_texts:
            engine_logger.warning("EMBEDDER: Empty text list received.")
            return []

        return cleaned_texts

    def _convert_embeddings_to_list(self, embeddings) -> list[list[float]]:
        """
        Converts model output to a Python list safely.
        Supports numpy-like objects and already-materialized lists.
        """
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        if not isinstance(embeddings, list):
            raise ValueError("Embedding output must be convertible to a list.")

        converted_embeddings = []

        for index, vector in enumerate(embeddings):
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            if not isinstance(vector, list) or not vector:
                raise ValueError(
                    f"Embedding vector at index {index} is invalid or empty."
                )

            normalized_vector = []
            for value in vector:
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Embedding vector at index {index} contains non-numeric values."
                    )
                normalized_vector.append(float(value))

            converted_embeddings.append(normalized_vector)

        return converted_embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of texts into a list of embedding vectors.
        Used for storing multiple micro skills at once.
        """
        cleaned_texts = self._validate_text_list(texts)
        if not cleaned_texts:
            return []

        try:
            self._ensure_model_loaded()

            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            converted_embeddings = self._convert_embeddings_to_list(embeddings)

            if len(converted_embeddings) != len(cleaned_texts):
                raise ValueError(
                    "Embedding count mismatch between input texts and output vectors."
                )

            engine_logger.info(
                f"EMBEDDER: Embedded {len(cleaned_texts)} text(s) successfully."
            )
            return converted_embeddings

        except Exception as exc:
            engine_logger.error(f"EMBEDDER: Failed to embed texts: {exc}")
            raise

    def embed_single(self, text: str) -> list[float]:
        """
        Converts a single text into an embedding vector.
        Used for query-time retrieval.
        """
        cleaned_text = self._normalize_text(text)
        if cleaned_text is None:
            engine_logger.warning(
                "EMBEDDER: Empty or invalid text received for single embed."
            )
            return []

        try:
            embeddings = self.embed_texts([cleaned_text])
            if not embeddings:
                return []

            engine_logger.info("EMBEDDER: Embedded 1 text successfully.")
            return embeddings[0]

        except Exception as exc:
            engine_logger.error(f"EMBEDDER: Failed to embed single text: {exc}")
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
"""
backend/rag/chroma_client.py
Handles all ChromaDB operations — store, retrieve, delete, validate.
Two collections: micro_skills and teacher_references.
Single Source of Truth for all RAG data operations.
"""

import chromadb
from chromadb.config import Settings

from backend.config.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_SKILLS_COLLECTION,
    CHROMA_REFERENCES_COLLECTION
)
from backend.config.constants import TOTAL_WEIGHT_TARGET
from backend.utils.logger import engine_logger


class ChromaClient:
    """
    Manages both ChromaDB collections:
    - micro_skills:       stores ranked + weighted micro skills per assignment
    - teacher_references: stores correct implementation references per skill
    """

    def __init__(self):
        self.client = None
        self.skills_collection = None
        self.references_collection = None

    def _ensure_initialized(self) -> None:
        """
        Lazily initializes ChromaDB client and both collections.
        Prevents import-time crashes.
        """
        if (
            self.client is not None
            and self.skills_collection is not None
            and self.references_collection is not None
        ):
            return

        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )

            self.skills_collection = self.client.get_or_create_collection(
                name=CHROMA_SKILLS_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )

            self.references_collection = self.client.get_or_create_collection(
                name=CHROMA_REFERENCES_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )

            engine_logger.info("CHROMA: ChromaDB initialized successfully.")

        except Exception as e:
            engine_logger.error(f"CHROMA: Initialization failed: {e}")
            raise

    def _is_valid_assignment_id(self, assignment_id: str) -> bool:
        """
        Validates assignment_id before any Chroma operation.
        """
        if not assignment_id or not str(assignment_id).strip():
            engine_logger.error("CHROMA: assignment_id is missing or blank.")
            return False
        return True

    def _validate_skill_payload(
        self,
        skills: list[dict],
        embeddings: list[list[float]],
        assignment_id: str
    ) -> bool:
        """
        Validates micro skill payload before storing.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not isinstance(skills, list) or not skills:
            engine_logger.error("CHROMA: skills must be a non-empty list.")
            return False

        if not isinstance(embeddings, list) or not embeddings:
            engine_logger.error("CHROMA: embeddings must be a non-empty list.")
            return False

        if len(skills) != len(embeddings):
            engine_logger.error(
                f"CHROMA: Skills/embeddings length mismatch for '{assignment_id}'. "
                f"Skills={len(skills)}, embeddings={len(embeddings)}"
            )
            return False

        for index, skill in enumerate(skills):
            if not isinstance(skill, dict):
                engine_logger.error(
                    f"CHROMA: Skill at index {index} is not a dict."
                )
                return False

            skill_text = skill.get("text", "")
            if not isinstance(skill_text, str) or not skill_text.strip():
                engine_logger.error(
                    f"CHROMA: Skill text missing or blank at index {index}."
                )
                return False

        total_weight = sum(skill.get("weight", 0) for skill in skills)
        if total_weight != TOTAL_WEIGHT_TARGET:
            engine_logger.error(
                f"CHROMA: Weight sum mismatch for '{assignment_id}'. "
                f"Expected {TOTAL_WEIGHT_TARGET}, got {total_weight}. "
                f"Aborting store."
            )
            return False

        return True

    def _validate_reference_payload(
        self,
        references: list[dict],
        embeddings: list[list[float]],
        assignment_id: str
    ) -> bool:
        """
        Validates teacher reference payload before storing.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not isinstance(references, list) or not references:
            engine_logger.error("CHROMA: references must be a non-empty list.")
            return False

        if not isinstance(embeddings, list) or not embeddings:
            engine_logger.error("CHROMA: embeddings must be a non-empty list.")
            return False

        if len(references) != len(embeddings):
            engine_logger.error(
                f"CHROMA: References/embeddings length mismatch for '{assignment_id}'. "
                f"References={len(references)}, embeddings={len(embeddings)}"
            )
            return False

        for index, reference in enumerate(references):
            if not isinstance(reference, dict):
                engine_logger.error(
                    f"CHROMA: Reference at index {index} is not a dict."
                )
                return False

            snippet = reference.get("snippet", "")
            if not isinstance(snippet, str) or not snippet.strip():
                engine_logger.error(
                    f"CHROMA: Reference snippet missing or blank at index {index}."
                )
                return False

        return True

    def assignment_exists(self, assignment_id: str) -> bool:
        """
        Checks if micro skills already exist for this assignment.
        Always call before running Phase 1.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        try:
            self._ensure_initialized()

            results = self.skills_collection.get(
                where={"assignment_id": assignment_id}
            )

            exists = len(results.get("documents", [])) > 0

            engine_logger.info(
                f"CHROMA: Assignment '{assignment_id}' exists: {exists}"
            )
            return exists

        except Exception as e:
            engine_logger.error(f"CHROMA: assignment_exists check failed: {e}")
            return False

    def store_micro_skills(
        self,
        skills: list[dict],
        assignment_id: str,
        embeddings: list[list[float]],
        force_regenerate: bool = False
    ) -> bool:
        """
        Stores ranked and weighted micro skills into ChromaDB.
        Validates weight sum = 10 before storing.
        Never overwrites without explicit force_regenerate flag.
        """
        if not self._validate_skill_payload(skills, embeddings, assignment_id):
            return False

        try:
            self._ensure_initialized()

            exists = self.assignment_exists(assignment_id)

            if exists and not force_regenerate:
                engine_logger.warning(
                    f"CHROMA: Skills already exist for '{assignment_id}'. "
                    f"Use force_regenerate=True to overwrite."
                )
                return False

            if exists and force_regenerate:
                if not self.clear_assignment(assignment_id):
                    engine_logger.error(
                        f"CHROMA: Failed to clear existing data for '{assignment_id}' "
                        f"before regenerating skills."
                    )
                    return False

            ids = [f"{assignment_id}_skill_{index}" for index in range(len(skills))]
            documents = [skill["text"].strip() for skill in skills]
            metadatas = [
                {
                    "assignment_id": assignment_id,
                    "skill_index": index,
                    "rank": skill.get("rank", index + 1),
                    "weight": skill.get("weight", 1),
                }
                for index, skill in enumerate(skills)
            ]

            self.skills_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            engine_logger.info(
                f"CHROMA: Stored {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )
            return True

        except Exception as e:
            engine_logger.error(f"CHROMA: store_micro_skills failed: {e}")
            return False

    def retrieve_micro_skills(self, assignment_id: str) -> list[dict]:
        """
        Retrieves all micro skills for an assignment ordered by rank.
        Returns list of skill dicts with text, rank, weight.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return []

        try:
            self._ensure_initialized()

            results = self.skills_collection.get(
                where={"assignment_id": assignment_id}
            )

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not documents:
                engine_logger.warning(
                    f"CHROMA: No micro skills found for '{assignment_id}'."
                )
                return []

            paired = sorted(
                zip(metadatas, documents),
                key=lambda item: item[0].get("rank", 0)
            )

            skills = [
                {
                    "text": document,
                    "rank": metadata.get("rank", 0),
                    "weight": metadata.get("weight", 1)
                }
                for metadata, document in paired
            ]

            engine_logger.info(
                f"CHROMA: Retrieved {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )
            return skills

        except Exception as e:
            engine_logger.error(f"CHROMA: retrieve_micro_skills failed: {e}")
            return []

    def store_teacher_references(
        self,
        references: list[dict],
        assignment_id: str,
        embeddings: list[list[float]],
        force_regenerate: bool = False
    ) -> bool:
        """
        Stores teacher technical feedback references per micro skill.
        Generated once in Phase 1 — reused for every student in Phase 2.
        Never overwrites without explicit force_regenerate flag.
        """
        if not self._validate_reference_payload(references, embeddings, assignment_id):
            return False

        try:
            self._ensure_initialized()

            existing = self.references_collection.get(
                where={"assignment_id": assignment_id}
            )
            existing_documents = existing.get("documents", [])

            if existing_documents and not force_regenerate:
                engine_logger.warning(
                    f"CHROMA: Teacher references already exist for '{assignment_id}'. "
                    f"Use force_regenerate=True to overwrite."
                )
                return False

            if existing_documents and force_regenerate:
                if not self.clear_assignment(assignment_id):
                    engine_logger.error(
                        f"CHROMA: Failed to clear existing data for '{assignment_id}' "
                        f"before regenerating teacher references."
                    )
                    return False

            ids = [f"{assignment_id}_ref_{index}" for index in range(len(references))]
            documents = [reference["snippet"].strip() for reference in references]
            metadatas = [
                {
                    "assignment_id": assignment_id,
                    "skill_index": index,
                    "rank": reference.get("rank", index + 1),
                    "status": "PASS",
                    "line_start": reference.get("line_start", 0),
                    "line_end": reference.get("line_end", 0),
                }
                for index, reference in enumerate(references)
            ]

            self.references_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            engine_logger.info(
                f"CHROMA: Stored {len(references)} teacher references "
                f"for assignment '{assignment_id}'."
            )
            return True

        except Exception as e:
            engine_logger.error(f"CHROMA: store_teacher_references failed: {e}")
            return False

    def retrieve_teacher_reference(
        self,
        assignment_id: str,
        skill_rank: int
    ) -> dict | None:
        """
        Retrieves teacher reference for one specific skill by rank.
        Called by Agent 2 during verification loop per micro skill.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return None

        try:
            self._ensure_initialized()

            results = self.references_collection.get(
                where={
                    "$and": [
                        {"assignment_id": assignment_id},
                        {"rank": skill_rank}
                    ]
                }
            )

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not documents:
                engine_logger.warning(
                    f"CHROMA: No teacher reference found for "
                    f"assignment '{assignment_id}' rank {skill_rank}."
                )
                return None

            metadata = metadatas[0]

            return {
                "snippet": documents[0],
                "rank": metadata.get("rank", skill_rank),
                "line_start": metadata.get("line_start", 0),
                "line_end": metadata.get("line_end", 0),
                "status": metadata.get("status", "PASS")
            }

        except Exception as e:
            engine_logger.error(f"CHROMA: retrieve_teacher_reference failed: {e}")
            return None

    def clear_assignment(self, assignment_id: str) -> bool:
        """
        Clears all micro skills and teacher references for an assignment.
        Required before force regeneration.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        try:
            self._ensure_initialized()

            skills = self.skills_collection.get(
                where={"assignment_id": assignment_id}
            )
            skill_ids = skills.get("ids", [])
            if skill_ids:
                self.skills_collection.delete(ids=skill_ids)

            references = self.references_collection.get(
                where={"assignment_id": assignment_id}
            )
            reference_ids = references.get("ids", [])
            if reference_ids:
                self.references_collection.delete(ids=reference_ids)

            engine_logger.info(
                f"CHROMA: Cleared all data for assignment '{assignment_id}'."
            )
            return True

        except Exception as e:
            engine_logger.error(f"CHROMA: clear_assignment failed: {e}")
            return False


_chroma_client_instance = None


def get_chroma_client() -> ChromaClient:
    """
    Returns the shared ChromaClient singleton with lazy initialization.
    """
    global _chroma_client_instance

    if _chroma_client_instance is None:
        _chroma_client_instance = ChromaClient()

    return _chroma_client_instance
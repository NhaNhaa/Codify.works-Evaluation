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
    CHROMA_REFERENCES_COLLECTION,
    CHROMA_SKILLS_COLLECTION,
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
                settings=Settings(anonymized_telemetry=False),
            )

            self.skills_collection = self.client.get_or_create_collection(
                name=CHROMA_SKILLS_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

            self.references_collection = self.client.get_or_create_collection(
                name=CHROMA_REFERENCES_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

            engine_logger.info("CHROMA: ChromaDB initialized successfully.")

        except Exception as exc:
            engine_logger.error(f"CHROMA: Initialization failed: {exc}")
            raise

    def _is_valid_assignment_id(self, assignment_id: str) -> bool:
        """
        Validates assignment_id before any Chroma operation.
        """
        if not assignment_id or not str(assignment_id).strip():
            engine_logger.error("CHROMA: assignment_id is missing or blank.")
            return False
        return True

    def _is_valid_positive_int(self, value) -> bool:
        """
        Returns True only for positive integers.
        """
        return isinstance(value, int) and value > 0

    def _is_valid_embedding_vector(self, embedding) -> bool:
        """
        Validates one embedding vector.
        """
        if not isinstance(embedding, list) or not embedding:
            return False

        for value in embedding:
            if not isinstance(value, (int, float)):
                return False

        return True

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
        assignment_id: str,
        payload_label: str,
    ) -> bool:
        """
        Validates embeddings before storing.
        """
        if not isinstance(embeddings, list) or not embeddings:
            engine_logger.error(f"CHROMA: {payload_label} embeddings must be a non-empty list.")
            return False

        for index, embedding in enumerate(embeddings):
            if not self._is_valid_embedding_vector(embedding):
                engine_logger.error(
                    f"CHROMA: Invalid embedding vector at index {index} "
                    f"for '{assignment_id}'."
                )
                return False

        return True

    def _validate_skill_payload(
        self,
        skills: list[dict],
        embeddings: list[list[float]],
        assignment_id: str,
    ) -> bool:
        """
        Validates micro skill payload before storing.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not isinstance(skills, list) or not skills:
            engine_logger.error("CHROMA: skills must be a non-empty list.")
            return False

        if not self._validate_embeddings(embeddings, assignment_id, "skill"):
            return False

        if len(skills) != len(embeddings):
            engine_logger.error(
                f"CHROMA: Skills/embeddings length mismatch for '{assignment_id}'. "
                f"Skills={len(skills)}, embeddings={len(embeddings)}"
            )
            return False

        seen_ranks = set()
        total_weight = 0

        for index, skill in enumerate(skills):
            if not isinstance(skill, dict):
                engine_logger.error(f"CHROMA: Skill at index {index} is not a dict.")
                return False

            skill_text = skill.get("text", "")
            rank = skill.get("rank")
            weight = skill.get("weight")

            if not isinstance(skill_text, str) or not skill_text.strip():
                engine_logger.error(
                    f"CHROMA: Skill text missing or blank at index {index}."
                )
                return False

            if not self._is_valid_positive_int(rank):
                engine_logger.error(
                    f"CHROMA: Skill rank invalid at index {index}: {rank}"
                )
                return False

            if rank in seen_ranks:
                engine_logger.error(
                    f"CHROMA: Duplicate skill rank detected for '{assignment_id}': {rank}"
                )
                return False

            if not self._is_valid_positive_int(weight):
                engine_logger.error(
                    f"CHROMA: Skill weight invalid at index {index}: {weight}"
                )
                return False

            seen_ranks.add(rank)
            total_weight += weight

        if total_weight != TOTAL_WEIGHT_TARGET:
            engine_logger.error(
                f"CHROMA: Weight sum mismatch for '{assignment_id}'. "
                f"Expected {TOTAL_WEIGHT_TARGET}, got {total_weight}. "
                "Aborting store."
            )
            return False

        return True

    def _validate_reference_payload(
        self,
        references: list[dict],
        embeddings: list[list[float]],
        assignment_id: str,
    ) -> bool:
        """
        Validates teacher reference payload before storing.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not isinstance(references, list) or not references:
            engine_logger.error("CHROMA: references must be a non-empty list.")
            return False

        if not self._validate_embeddings(embeddings, assignment_id, "reference"):
            return False

        if len(references) != len(embeddings):
            engine_logger.error(
                f"CHROMA: References/embeddings length mismatch for '{assignment_id}'. "
                f"References={len(references)}, embeddings={len(embeddings)}"
            )
            return False

        seen_ranks = set()

        for index, reference in enumerate(references):
            if not isinstance(reference, dict):
                engine_logger.error(f"CHROMA: Reference at index {index} is not a dict.")
                return False

            snippet = reference.get("snippet", "")
            rank = reference.get("rank")
            line_start = reference.get("line_start")
            line_end = reference.get("line_end")

            if not isinstance(snippet, str) or not snippet.strip():
                engine_logger.error(
                    f"CHROMA: Reference snippet missing or blank at index {index}."
                )
                return False

            if not self._is_valid_positive_int(rank):
                engine_logger.error(
                    f"CHROMA: Reference rank invalid at index {index}: {rank}"
                )
                return False

            if rank in seen_ranks:
                engine_logger.error(
                    f"CHROMA: Duplicate reference rank detected for '{assignment_id}': {rank}"
                )
                return False

            if not self._is_valid_positive_int(line_start) or not self._is_valid_positive_int(line_end):
                engine_logger.error(
                    f"CHROMA: Invalid reference line range at index {index}: "
                    f"{line_start}-{line_end}"
                )
                return False

            if line_start > line_end:
                engine_logger.error(
                    f"CHROMA: Reversed reference line range at index {index}: "
                    f"{line_start}-{line_end}"
                )
                return False

            seen_ranks.add(rank)

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

            results = self.skills_collection.get(where={"assignment_id": assignment_id})
            exists = len(results.get("documents", [])) > 0

            engine_logger.info(
                f"CHROMA: Assignment '{assignment_id}' exists: {exists}"
            )
            return exists

        except Exception as exc:
            engine_logger.error(f"CHROMA: assignment_exists check failed: {exc}")
            return False

    def store_micro_skills(
        self,
        skills: list[dict],
        assignment_id: str,
        embeddings: list[list[float]],
        force_regenerate: bool = False,
    ) -> bool:
        """
        Stores ranked and weighted micro skills into ChromaDB.
        Validates weight sum = 10 before storing.
        Never overwrites without explicit force_regenerate flag.
        Regenerating micro skills clears both skills and teacher references
        because references depend on the final skill set.
        """
        if not self._validate_skill_payload(skills, embeddings, assignment_id):
            return False

        try:
            self._ensure_initialized()

            exists = self.assignment_exists(assignment_id)

            if exists and not force_regenerate:
                engine_logger.warning(
                    f"CHROMA: Skills already exist for '{assignment_id}'. "
                    "Use force_regenerate=True to overwrite."
                )
                return False

            if exists and force_regenerate:
                if not self.clear_assignment(assignment_id):
                    engine_logger.error(
                        f"CHROMA: Failed to clear existing data for '{assignment_id}' "
                        "before regenerating skills."
                    )
                    return False

            ids = [f"{assignment_id}_skill_{index}" for index in range(len(skills))]
            documents = [skill["text"].strip() for skill in skills]
            metadatas = [
                {
                    "assignment_id": assignment_id,
                    "skill_index": index,
                    "rank": skill["rank"],
                    "weight": skill["weight"],
                }
                for index, skill in enumerate(skills)
            ]

            self.skills_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            engine_logger.info(
                f"CHROMA: Stored {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )
            return True

        except Exception as exc:
            engine_logger.error(f"CHROMA: store_micro_skills failed: {exc}")
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

            results = self.skills_collection.get(where={"assignment_id": assignment_id})

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not documents:
                engine_logger.warning(
                    f"CHROMA: No micro skills found for '{assignment_id}'."
                )
                return []

            paired_records = []
            for metadata, document in zip(metadatas, documents):
                if not isinstance(metadata, dict):
                    continue
                if not isinstance(document, str) or not document.strip():
                    continue

                paired_records.append((metadata, document.strip()))

            paired_records.sort(key=lambda item: item[0].get("rank", 0))

            skills = [
                {
                    "text": document,
                    "rank": metadata.get("rank", 0),
                    "weight": metadata.get("weight", 1),
                }
                for metadata, document in paired_records
            ]

            engine_logger.info(
                f"CHROMA: Retrieved {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )
            return skills

        except Exception as exc:
            engine_logger.error(f"CHROMA: retrieve_micro_skills failed: {exc}")
            return []

    def store_teacher_references(
        self,
        references: list[dict],
        assignment_id: str,
        embeddings: list[list[float]],
        force_regenerate: bool = False,
    ) -> bool:
        """
        Stores teacher technical feedback references per micro skill.
        Generated once in Phase 1 — reused for every student in Phase 2.
        Never overwrites without explicit force_regenerate flag.
        Regenerating teacher references clears only teacher references.
        """
        if not self._validate_reference_payload(references, embeddings, assignment_id):
            return False

        try:
            self._ensure_initialized()

            existing = self.references_collection.get(where={"assignment_id": assignment_id})
            existing_documents = existing.get("documents", [])

            if existing_documents and not force_regenerate:
                engine_logger.warning(
                    f"CHROMA: Teacher references already exist for '{assignment_id}'. "
                    "Use force_regenerate=True to overwrite."
                )
                return False

            if existing_documents and force_regenerate:
                if not self.clear_teacher_references(assignment_id):
                    engine_logger.error(
                        f"CHROMA: Failed to clear existing teacher references for "
                        f"'{assignment_id}' before regenerating."
                    )
                    return False

            ids = [f"{assignment_id}_ref_{index}" for index in range(len(references))]
            documents = [reference["snippet"].strip() for reference in references]
            metadatas = [
                {
                    "assignment_id": assignment_id,
                    "skill_index": index,
                    "rank": reference["rank"],
                    "status": "PASS",
                    "line_start": reference["line_start"],
                    "line_end": reference["line_end"],
                }
                for index, reference in enumerate(references)
            ]

            self.references_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            engine_logger.info(
                f"CHROMA: Stored {len(references)} teacher references "
                f"for assignment '{assignment_id}'."
            )
            return True

        except Exception as exc:
            engine_logger.error(f"CHROMA: store_teacher_references failed: {exc}")
            return False

    def retrieve_teacher_reference(
        self,
        assignment_id: str,
        skill_rank: int,
    ) -> dict | None:
        """
        Retrieves teacher reference for one specific skill by rank.
        Called by Agent 2 during verification loop per micro skill.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return None

        if not self._is_valid_positive_int(skill_rank):
            engine_logger.error("CHROMA: skill_rank must be a positive integer.")
            return None

        try:
            self._ensure_initialized()

            results = self.references_collection.get(
                where={
                    "$and": [
                        {"assignment_id": assignment_id},
                        {"rank": skill_rank},
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
                "status": metadata.get("status", "PASS"),
            }

        except Exception as exc:
            engine_logger.error(f"CHROMA: retrieve_teacher_reference failed: {exc}")
            return None

    def clear_micro_skills(self, assignment_id: str) -> bool:
        """
        Clears only micro skills for an assignment.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        try:
            self._ensure_initialized()

            skills = self.skills_collection.get(where={"assignment_id": assignment_id})
            skill_ids = skills.get("ids", [])
            if skill_ids:
                self.skills_collection.delete(ids=skill_ids)

            engine_logger.info(
                f"CHROMA: Cleared micro skills for assignment '{assignment_id}'."
            )
            return True

        except Exception as exc:
            engine_logger.error(f"CHROMA: clear_micro_skills failed: {exc}")
            return False

    def clear_teacher_references(self, assignment_id: str) -> bool:
        """
        Clears only teacher references for an assignment.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        try:
            self._ensure_initialized()

            references = self.references_collection.get(where={"assignment_id": assignment_id})
            reference_ids = references.get("ids", [])
            if reference_ids:
                self.references_collection.delete(ids=reference_ids)

            engine_logger.info(
                f"CHROMA: Cleared teacher references for assignment '{assignment_id}'."
            )
            return True

        except Exception as exc:
            engine_logger.error(f"CHROMA: clear_teacher_references failed: {exc}")
            return False

    def clear_assignment(self, assignment_id: str) -> bool:
        """
        Clears all micro skills and teacher references for an assignment.
        Required before full force regeneration.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        skills_cleared = self.clear_micro_skills(assignment_id)
        references_cleared = self.clear_teacher_references(assignment_id)

        if not skills_cleared or not references_cleared:
            engine_logger.error(
                f"CHROMA: Failed to fully clear assignment '{assignment_id}'."
            )
            return False

        engine_logger.info(
            f"CHROMA: Cleared all data for assignment '{assignment_id}'."
        )
        return True


_chroma_client_instance = None


def get_chroma_client() -> ChromaClient:
    """
    Returns the shared ChromaClient singleton with lazy initialization.
    """
    global _chroma_client_instance

    if _chroma_client_instance is None:
        _chroma_client_instance = ChromaClient()

    return _chroma_client_instance
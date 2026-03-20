"""
backend/rag/rag_pipeline.py
Public interface for all RAG operations.
Agents never touch chroma_client or embedder directly — only via this pipeline.
Orchestrates: embedder -> chroma_client -> store/retrieve.
"""

from backend.config.constants import TOTAL_WEIGHT_TARGET
from backend.rag.chroma_client import get_chroma_client
from backend.rag.embedder import get_embedder
from backend.utils.logger import engine_logger


class RAGPipeline:
    """
    Single access point for all RAG operations.
    Agent 1 uses store methods.
    Agent 2 uses retrieve methods.
    Agent 3 never touches RAG directly.
    """

    def _is_valid_assignment_id(self, assignment_id: str) -> bool:
        """
        Validates assignment_id before any RAG operation.
        """
        if not assignment_id or not str(assignment_id).strip():
            engine_logger.error("RAG PIPELINE: assignment_id is missing or blank.")
            return False
        return True

    def _is_valid_positive_int(self, value) -> bool:
        """
        Returns True only for positive integers.
        """
        return isinstance(value, int) and value > 0

    def _validate_skills_for_storage(self, skills: list[dict]) -> bool:
        """
        Validates ranked skill payload before embedding or storage.
        Ensures weight sum = TOTAL_WEIGHT_TARGET before any write path continues.
        """
        if not isinstance(skills, list) or not skills:
            engine_logger.error("RAG PIPELINE: Empty or invalid skills list — aborting store.")
            return False

        seen_ranks = set()
        total_weight = 0

        for index, skill in enumerate(skills):
            if not isinstance(skill, dict):
                engine_logger.error(
                    f"RAG PIPELINE: Skill entry at index {index} is not a dict."
                )
                return False

            skill_text = skill.get("text", "")
            rank = skill.get("rank")
            weight = skill.get("weight")

            if not isinstance(skill_text, str) or not skill_text.strip():
                engine_logger.error(
                    f"RAG PIPELINE: Skill text is missing or blank at index {index}."
                )
                return False

            if not self._is_valid_positive_int(rank):
                engine_logger.error(
                    f"RAG PIPELINE: Skill rank is invalid at index {index}: {rank}"
                )
                return False

            if rank in seen_ranks:
                engine_logger.error(
                    f"RAG PIPELINE: Duplicate skill rank detected: {rank}"
                )
                return False

            if not self._is_valid_positive_int(weight):
                engine_logger.error(
                    f"RAG PIPELINE: Skill weight is invalid at index {index}: {weight}"
                )
                return False

            seen_ranks.add(rank)
            total_weight += weight

        if total_weight != TOTAL_WEIGHT_TARGET:
            engine_logger.error(
                f"RAG PIPELINE: Weight sum mismatch — expected "
                f"{TOTAL_WEIGHT_TARGET}, got {total_weight}. Aborting store."
            )
            return False

        return True

    def _validate_references_for_storage(self, references: list[dict]) -> bool:
        """
        Validates teacher reference payload before embedding or storage.
        """
        if not isinstance(references, list) or not references:
            engine_logger.error(
                "RAG PIPELINE: Empty or invalid references list — aborting store."
            )
            return False

        seen_ranks = set()

        for index, reference in enumerate(references):
            if not isinstance(reference, dict):
                engine_logger.error(
                    f"RAG PIPELINE: Reference entry at index {index} is not a dict."
                )
                return False

            snippet = reference.get("snippet", "")
            rank = reference.get("rank")
            line_start = reference.get("line_start")
            line_end = reference.get("line_end")

            if not isinstance(snippet, str) or not snippet.strip():
                engine_logger.error(
                    f"RAG PIPELINE: Reference snippet is missing or blank at index {index}."
                )
                return False

            if not self._is_valid_positive_int(rank):
                engine_logger.error(
                    f"RAG PIPELINE: Reference rank is invalid at index {index}: {rank}"
                )
                return False

            if rank in seen_ranks:
                engine_logger.error(
                    f"RAG PIPELINE: Duplicate reference rank detected: {rank}"
                )
                return False

            if not self._is_valid_positive_int(line_start) or not self._is_valid_positive_int(line_end):
                engine_logger.error(
                    f"RAG PIPELINE: Reference line range is invalid at index {index}: "
                    f"{line_start}-{line_end}"
                )
                return False

            if line_start > line_end:
                engine_logger.error(
                    f"RAG PIPELINE: Reference line range is reversed at index {index}: "
                    f"{line_start}-{line_end}"
                )
                return False

            seen_ranks.add(rank)

        return True

    def assignment_exists(self, assignment_id: str) -> bool:
        """
        Checks if micro skills already exist for this assignment.
        Always called first before running Phase 1.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        return get_chroma_client().assignment_exists(assignment_id)

    def store_micro_skills(
        self,
        skills: list[dict],
        assignment_id: str,
        force_regenerate: bool = False,
    ) -> bool:
        """
        Embeds and stores ranked + weighted micro skills into ChromaDB.
        Validates weight sum = 10 before storing.
        Never overwrites without explicit force_regenerate flag.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not self._validate_skills_for_storage(skills):
            return False

        try:
            texts = [skill["text"].strip() for skill in skills]

            engine_logger.info(
                f"RAG PIPELINE: Storing {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )

            embeddings = get_embedder().embed_texts(texts)
            if not embeddings:
                engine_logger.error("RAG PIPELINE: Skill embeddings are empty — aborting store.")
                return False

            if len(embeddings) != len(texts):
                engine_logger.error(
                    f"RAG PIPELINE: Skill embedding count mismatch for '{assignment_id}'. "
                    f"Texts={len(texts)}, embeddings={len(embeddings)}"
                )
                return False

            return get_chroma_client().store_micro_skills(
                skills=skills,
                assignment_id=assignment_id,
                embeddings=embeddings,
                force_regenerate=force_regenerate,
            )

        except Exception as exc:
            engine_logger.error(f"RAG PIPELINE: store_micro_skills failed: {exc}")
            return False

    def retrieve_micro_skills(self, assignment_id: str) -> list[dict]:
        """
        Retrieves all micro skills for an assignment ordered by rank.
        Called by Agent 2 at start of evaluation loop.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return []

        engine_logger.info(
            f"RAG PIPELINE: Retrieving micro skills "
            f"for assignment '{assignment_id}'."
        )
        return get_chroma_client().retrieve_micro_skills(assignment_id)

    def store_teacher_references(
        self,
        references: list[dict],
        assignment_id: str,
        force_regenerate: bool = False,
    ) -> bool:
        """
        Embeds and stores teacher technical feedback references.
        Generated once in Phase 1 — reused for every student in Phase 2.
        Never overwrites without explicit force_regenerate flag.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not self._validate_references_for_storage(references):
            return False

        try:
            texts = [reference["snippet"].strip() for reference in references]

            engine_logger.info(
                f"RAG PIPELINE: Storing {len(references)} teacher references "
                f"for assignment '{assignment_id}'."
            )

            embeddings = get_embedder().embed_texts(texts)
            if not embeddings:
                engine_logger.error("RAG PIPELINE: Reference embeddings are empty — aborting store.")
                return False

            if len(embeddings) != len(texts):
                engine_logger.error(
                    f"RAG PIPELINE: Reference embedding count mismatch for '{assignment_id}'. "
                    f"Texts={len(texts)}, embeddings={len(embeddings)}"
                )
                return False

            return get_chroma_client().store_teacher_references(
                references=references,
                assignment_id=assignment_id,
                embeddings=embeddings,
                force_regenerate=force_regenerate,
            )

        except Exception as exc:
            engine_logger.error(
                f"RAG PIPELINE: store_teacher_references failed: {exc}"
            )
            return False

    def retrieve_teacher_reference(
        self,
        assignment_id: str,
        skill_rank: int,
    ) -> dict | None:
        """
        Retrieves teacher reference for one specific skill by rank.
        Called by Agent 2 per micro skill during verification loop.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return None

        if not self._is_valid_positive_int(skill_rank):
            engine_logger.error("RAG PIPELINE: skill_rank must be a positive integer.")
            return None

        engine_logger.info(
            f"RAG PIPELINE: Retrieving teacher reference for "
            f"assignment '{assignment_id}' rank {skill_rank}."
        )
        return get_chroma_client().retrieve_teacher_reference(
            assignment_id=assignment_id,
            skill_rank=skill_rank,
        )

    def clear_assignment(self, assignment_id: str) -> bool:
        """
        Clears all micro skills and teacher references for an assignment.
        Called by DELETE /skills/{assignment_id} endpoint.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        engine_logger.info(
            f"RAG PIPELINE: Clearing all data for assignment '{assignment_id}'."
        )
        return get_chroma_client().clear_assignment(assignment_id)


# ── GLOBAL INSTANCE ────────────────────────────────────────────────
# Single shared instance — all agents import and use this
rag_pipeline = RAGPipeline()
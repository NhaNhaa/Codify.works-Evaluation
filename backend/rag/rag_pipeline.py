"""
backend/rag/rag_pipeline.py
Public interface for all RAG operations.
Agents never touch chroma_client or embedder directly — only via this pipeline.
Orchestrates: embedder -> chroma_client -> store/retrieve.
"""

from backend.rag.embedder import get_embedder
from backend.rag.chroma_client import get_chroma_client
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
        force_regenerate: bool = False
    ) -> bool:
        """
        Embeds and stores ranked + weighted micro skills into ChromaDB.
        Validates weight sum = 10 before storing.
        Never overwrites without explicit force_regenerate flag.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not isinstance(skills, list) or not skills:
            engine_logger.error("RAG PIPELINE: Empty or invalid skills list — aborting store.")
            return False

        try:
            texts = []
            for skill in skills:
                if not isinstance(skill, dict):
                    engine_logger.error("RAG PIPELINE: Skill entry is not a dict.")
                    return False

                skill_text = skill.get("text", "")
                if not isinstance(skill_text, str) or not skill_text.strip():
                    engine_logger.error("RAG PIPELINE: Skill text is missing or blank.")
                    return False

                texts.append(skill_text.strip())

            engine_logger.info(
                f"RAG PIPELINE: Storing {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )

            embeddings = get_embedder().embed_texts(texts)
            if not embeddings:
                engine_logger.error("RAG PIPELINE: Skill embeddings are empty — aborting store.")
                return False

            return get_chroma_client().store_micro_skills(
                skills=skills,
                assignment_id=assignment_id,
                embeddings=embeddings,
                force_regenerate=force_regenerate
            )

        except Exception as e:
            engine_logger.error(f"RAG PIPELINE: store_micro_skills failed: {e}")
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
        force_regenerate: bool = False
    ) -> bool:
        """
        Embeds and stores teacher technical feedback references.
        Generated once in Phase 1 — reused for every student in Phase 2.
        Never overwrites without explicit force_regenerate flag.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        if not isinstance(references, list) or not references:
            engine_logger.error("RAG PIPELINE: Empty or invalid references list — aborting store.")
            return False

        try:
            texts = []
            for reference in references:
                if not isinstance(reference, dict):
                    engine_logger.error("RAG PIPELINE: Reference entry is not a dict.")
                    return False

                snippet = reference.get("snippet", "")
                if not isinstance(snippet, str) or not snippet.strip():
                    engine_logger.error("RAG PIPELINE: Reference snippet is missing or blank.")
                    return False

                texts.append(snippet.strip())

            engine_logger.info(
                f"RAG PIPELINE: Storing {len(references)} teacher references "
                f"for assignment '{assignment_id}'."
            )

            embeddings = get_embedder().embed_texts(texts)
            if not embeddings:
                engine_logger.error("RAG PIPELINE: Reference embeddings are empty — aborting store.")
                return False

            return get_chroma_client().store_teacher_references(
                references=references,
                assignment_id=assignment_id,
                embeddings=embeddings,
                force_regenerate=force_regenerate
            )

        except Exception as e:
            engine_logger.error(
                f"RAG PIPELINE: store_teacher_references failed: {e}"
            )
            return False

    def retrieve_teacher_reference(
        self,
        assignment_id: str,
        skill_rank: int
    ) -> dict | None:
        """
        Retrieves teacher reference for one specific skill by rank.
        Called by Agent 2 per micro skill during verification loop.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return None

        engine_logger.info(
            f"RAG PIPELINE: Retrieving teacher reference for "
            f"assignment '{assignment_id}' rank {skill_rank}."
        )
        return get_chroma_client().retrieve_teacher_reference(
            assignment_id=assignment_id,
            skill_rank=skill_rank
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
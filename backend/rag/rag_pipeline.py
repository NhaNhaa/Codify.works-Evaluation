"""
backend/rag/rag_pipeline.py
Public interface for all RAG operations.
Agents never touch chroma_client or embedder directly — only via this pipeline.
Orchestrates: embedder → chroma_client → store/retrieve.
"""

from backend.rag.embedder import embedder
from backend.rag.chroma_client import chroma_client
from backend.utils.logger import engine_logger


class RAGPipeline:
    """
    Single access point for all RAG operations.
    Agent 1 uses store methods.
    Agent 2 uses retrieve methods.
    Agent 3 never touches RAG directly.
    """

    # ── ASSIGNMENT EXISTS CHECK ────────────────────────────────────
    def assignment_exists(self, assignment_id: str) -> bool:
        """
        Checks if micro skills already exist for this assignment.
        Always called first before running Phase 1.
        """
        return chroma_client.assignment_exists(assignment_id)


    # ── STORE MICRO SKILLS ─────────────────────────────────────────
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
        engine_logger.info(
            f"RAG PIPELINE: Storing {len(skills)} micro skills "
            f"for assignment '{assignment_id}'."
        )

        if not skills:
            engine_logger.error("RAG PIPELINE: Empty skills list — aborting store.")
            return False

        try:
            texts      = [s["text"] for s in skills]
            embeddings = embedder.embed_texts(texts)

            return chroma_client.store_micro_skills(
                skills=skills,
                assignment_id=assignment_id,
                embeddings=embeddings,
                force_regenerate=force_regenerate
            )
        except Exception as e:
            engine_logger.error(f"RAG PIPELINE: store_micro_skills failed: {e}")
            return False


    # ── RETRIEVE MICRO SKILLS ──────────────────────────────────────
    def retrieve_micro_skills(self, assignment_id: str) -> list[dict]:
        """
        Retrieves all micro skills for an assignment ordered by rank.
        Called by Agent 2 at start of evaluation loop.
        """
        engine_logger.info(
            f"RAG PIPELINE: Retrieving micro skills "
            f"for assignment '{assignment_id}'."
        )
        return chroma_client.retrieve_micro_skills(assignment_id)


    # ── STORE TEACHER REFERENCES ───────────────────────────────────
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
        engine_logger.info(
            f"RAG PIPELINE: Storing {len(references)} teacher references "
            f"for assignment '{assignment_id}'."
        )

        if not references:
            engine_logger.error(
                "RAG PIPELINE: Empty references list — aborting store."
            )
            return False

        try:
            texts      = [r.get("snippet", "") for r in references]
            embeddings = embedder.embed_texts(texts)

            return chroma_client.store_teacher_references(
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


    # ── RETRIEVE TEACHER REFERENCE FOR ONE SKILL ──────────────────
    def retrieve_teacher_reference(
        self,
        assignment_id: str,
        skill_rank: int
    ) -> dict | None:
        """
        Retrieves teacher reference for one specific skill by rank.
        Called by Agent 2 per micro skill during verification loop.
        """
        engine_logger.info(
            f"RAG PIPELINE: Retrieving teacher reference for "
            f"assignment '{assignment_id}' rank {skill_rank}."
        )
        return chroma_client.retrieve_teacher_reference(
            assignment_id=assignment_id,
            skill_rank=skill_rank
        )


    # ── CLEAR ASSIGNMENT ───────────────────────────────────────────
    def clear_assignment(self, assignment_id: str) -> bool:
        """
        Clears all micro skills and teacher references for an assignment.
        Called by DELETE /skills/{assignment_id} endpoint.
        """
        engine_logger.info(
            f"RAG PIPELINE: Clearing all data for assignment '{assignment_id}'."
        )
        return chroma_client.clear_assignment(assignment_id)


# ── GLOBAL INSTANCE ────────────────────────────────────────────────
# Single shared instance — all agents import and use this
rag_pipeline = RAGPipeline()
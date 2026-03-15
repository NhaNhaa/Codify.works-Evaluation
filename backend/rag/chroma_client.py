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


    # ── ASSIGNMENT EXISTS CHECK ────────────────────────────────────
    def assignment_exists(self, assignment_id: str) -> bool:
        """
        Checks if micro skills already exist for this assignment.
        Always call before running Phase 1.
        """
        try:
            results = self.skills_collection.get(
                where={"assignment_id": assignment_id}
            )
            exists = len(results["documents"]) > 0
            engine_logger.info(
                f"CHROMA: Assignment '{assignment_id}' exists: {exists}"
            )
            return exists
        except Exception as e:
            engine_logger.error(f"CHROMA: assignment_exists check failed: {e}")
            return False


    # ── STORE MICRO SKILLS ─────────────────────────────────────────
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
        # Guard — never overwrite without force flag
        if not force_regenerate and self.assignment_exists(assignment_id):
            engine_logger.warning(
                f"CHROMA: Skills already exist for '{assignment_id}'. "
                f"Use force_regenerate=True to overwrite."
            )
            return False

        # Validate weight sum before any write
        total_weight = sum(s.get("weight", 0) for s in skills)
        if total_weight != TOTAL_WEIGHT_TARGET:
            engine_logger.error(
                f"CHROMA: Weight sum mismatch for '{assignment_id}'. "
                f"Expected {TOTAL_WEIGHT_TARGET}, got {total_weight}. "
                f"Aborting store."
            )
            return False

        try:
            ids       = [f"{assignment_id}_skill_{i}" for i in range(len(skills))]
            documents = [s["text"] for s in skills]
            metadatas = [
                {
                    "assignment_id": assignment_id,
                    "skill_index":   i,
                    "rank":          s.get("rank", i + 1),
                    "weight":        s.get("weight", 1),
                }
                for i, s in enumerate(skills)
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


    # ── RETRIEVE MICRO SKILLS ──────────────────────────────────────
    def retrieve_micro_skills(self, assignment_id: str) -> list[dict]:
        """
        Retrieves all micro skills for an assignment ordered by rank.
        Returns list of skill dicts with text, rank, weight.
        """
        try:
            results = self.skills_collection.get(
                where={"assignment_id": assignment_id}
            )
            if not results["documents"]:
                engine_logger.warning(
                    f"CHROMA: No micro skills found for '{assignment_id}'."
                )
                return []

            # Pair documents with metadata and sort by rank
            paired = sorted(
                zip(results["metadatas"], results["documents"]),
                key=lambda x: x[0].get("rank", 0)
            )

            skills = [
                {
                    "text":   doc,
                    "rank":   meta.get("rank", 0),
                    "weight": meta.get("weight", 1)
                }
                for meta, doc in paired
            ]

            engine_logger.info(
                f"CHROMA: Retrieved {len(skills)} micro skills "
                f"for assignment '{assignment_id}'."
            )
            return skills
        except Exception as e:
            engine_logger.error(f"CHROMA: retrieve_micro_skills failed: {e}")
            return []


    # ── STORE TEACHER REFERENCES ───────────────────────────────────
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
        # Guard — never overwrite without force flag
        existing = self.references_collection.get(
            where={"assignment_id": assignment_id}
        )
        if not force_regenerate and len(existing["documents"]) > 0:
            engine_logger.warning(
                f"CHROMA: Teacher references already exist for '{assignment_id}'. "
                f"Use force_regenerate=True to overwrite."
            )
            return False

        try:
            ids       = [f"{assignment_id}_ref_{i}" for i in range(len(references))]
            documents = [r.get("snippet", "") for r in references]
            metadatas = [
                {
                    "assignment_id": assignment_id,
                    "skill_index":   i,
                    "rank":          r.get("rank", i + 1),
                    "status":        "PASS",
                    "line_start":    r.get("line_start", 0),
                    "line_end":      r.get("line_end", 0),
                }
                for i, r in enumerate(references)
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


    # ── RETRIEVE TEACHER REFERENCE FOR ONE SKILL ──────────────────
    def retrieve_teacher_reference(
        self,
        assignment_id: str,
        skill_rank: int
    ) -> dict | None:
        """
        Retrieves teacher reference for one specific skill by rank.
        Called by Agent 2 during verification loop per micro skill.
        """
        try:
            results = self.references_collection.get(
                where={
                    "$and": [
                        {"assignment_id": assignment_id},
                        {"rank": skill_rank}
                    ]
                }
            )
            if not results["documents"]:
                engine_logger.warning(
                    f"CHROMA: No teacher reference found for "
                    f"assignment '{assignment_id}' rank {skill_rank}."
                )
                return None

            meta = results["metadatas"][0]
            return {
                "snippet":    results["documents"][0],
                "rank":       meta.get("rank", skill_rank),
                "line_start": meta.get("line_start", 0),
                "line_end":   meta.get("line_end", 0),
                "status":     meta.get("status", "PASS")
            }
        except Exception as e:
            engine_logger.error(
                f"CHROMA: retrieve_teacher_reference failed: {e}"
            )
            return None


    # ── CLEAR ASSIGNMENT ───────────────────────────────────────────
    def clear_assignment(self, assignment_id: str) -> bool:
        """
        Clears all micro skills and teacher references for an assignment.
        Required before force regeneration via DELETE /skills endpoint.
        """
        try:
            # Clear micro skills
            skills = self.skills_collection.get(
                where={"assignment_id": assignment_id}
            )
            if skills["ids"]:
                self.skills_collection.delete(ids=skills["ids"])

            # Clear teacher references
            refs = self.references_collection.get(
                where={"assignment_id": assignment_id}
            )
            if refs["ids"]:
                self.references_collection.delete(ids=refs["ids"])

            engine_logger.info(
                f"CHROMA: Cleared all data for assignment '{assignment_id}'."
            )
            return True
        except Exception as e:
            engine_logger.error(f"CHROMA: clear_assignment failed: {e}")
            return False


# ── GLOBAL INSTANCE ────────────────────────────────────────────────
# Initialized once — shared across all agents via rag_pipeline
chroma_client = ChromaClient()
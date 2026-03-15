"""
backend/agents/agent2_evaluator.py
Agent 2: Evaluator — Phase 2.
Runs per student.
Evaluates student_code.c against each micro skill from ChromaDB.
Two retrievals per skill: micro_skill first, teacher_reference after.
Verification loop — max 3 attempts per skill.
If verifier is inconclusive, trusts initial unbiased evaluation.
Enforces snippet size limits via Python post-processing.
Forces recommended_fix from teacher reference — never invents fixes.
All LLM calls go through llm_client.call_llm_with_retry — auto rate limit retry.
"""

import json
import re
from openai import OpenAI
from backend.config.config import get_provider_config, get_model
from backend.config.constants import (
    MAX_REVERIFICATION_ATTEMPTS,
    MAX_SNIPPET_LINES
)
from backend.rag.rag_pipeline import rag_pipeline
from backend.utils.logger import engine_logger
from backend.utils.security import safe_read_file
from backend.utils.llm_client import call_llm_with_retry


class Evaluator:
    """
    Agent 2 — Evaluator.
    Loops through each micro skill from ChromaDB.
    Produces student technical feedback per skill.
    Verifies each verdict against teacher reference.
    Self-corrects silently — no teacher escalation ever.
    Forces recommended_fix from teacher_references ChromaDB — zero hallucination.
    """

    def __init__(self):
        provider = get_provider_config()
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"]
        )
        self.model = get_model("agent2_evaluator")
        engine_logger.info("AGENT 2: Evaluator initialized.")


    # ── ENTRY POINT ────────────────────────────────────────────────
    def run(
        self,
        assignment_id: str,
        student_id:    str,
        student_path:  str
    ) -> dict | None:
        """
        Full Phase 2 pipeline.
        Returns evaluation result dict ready for Agent 3.
        """
        engine_logger.info(
            f"AGENT 2: Starting Phase 2 for student '{student_id}' "
            f"on assignment '{assignment_id}'."
        )

        student_code = safe_read_file(student_path, ".c")
        if not student_code:
            engine_logger.error(
                f"AGENT 2: Failed to read student file: {student_path}"
            )
            return None

        skills = rag_pipeline.retrieve_micro_skills(assignment_id)
        if not skills:
            engine_logger.error(
                f"AGENT 2: No micro skills found for '{assignment_id}'. "
                f"Run Phase 1 first."
            )
            return None

        engine_logger.info(
            f"AGENT 2: Retrieved {len(skills)} micro skills. "
            f"Starting evaluation loop."
        )

        evaluated_skills = []
        for skill in skills:
            verdict = self._evaluate_skill_with_verification(
                skill=skill,
                student_code=student_code,
                assignment_id=assignment_id
            )
            evaluated_skills.append(verdict)

        engine_logger.info(
            f"AGENT 2: Phase 2 complete for student '{student_id}'. "
            f"{len(evaluated_skills)} skills evaluated."
        )

        return {
            "student_id":    student_id,
            "assignment_id": assignment_id,
            "skills":        evaluated_skills
        }


    # ── EVALUATE ONE SKILL WITH VERIFICATION LOOP ──────────────────
    def _evaluate_skill_with_verification(
        self,
        skill:         dict,
        student_code:  str,
        assignment_id: str
    ) -> dict:
        """
        Evaluates one micro skill against student code.
        Order: evaluate first → retrieve teacher ref → verify.
        Forces recommended_fix from teacher reference for every FAIL.
        """
        skill_rank = skill["rank"]
        skill_text = skill["text"]

        engine_logger.info(
            f"AGENT 2: Evaluating skill rank {skill_rank}: '{skill_text}'"
        )

        # STEP 1 — Produce initial verdict (no teacher ref)
        raw_verdict = self._evaluate_student_code(
            skill=skill,
            student_code=student_code,
            teacher_ref=None
        )

        # RETRIEVAL 2 — Get teacher reference AFTER initial evaluation
        teacher_ref = rag_pipeline.retrieve_teacher_reference(
            assignment_id=assignment_id,
            skill_rank=skill_rank
        )

        # VERIFICATION LOOP
        attempts = 0
        verified = False

        while attempts < MAX_REVERIFICATION_ATTEMPTS and not verified:
            attempts += 1

            if teacher_ref:
                verified = self._verify_verdict(
                    verdict=raw_verdict,
                    teacher_ref=teacher_ref,
                    student_code=student_code,
                    skill_text=skill_text
                )
            else:
                engine_logger.warning(
                    f"AGENT 2: No teacher reference for rank {skill_rank}. "
                    f"Accepting verdict without verification."
                )
                verified = True
                break

            if not verified:
                engine_logger.warning(
                    f"AGENT 2: Verdict mismatch for rank {skill_rank}. "
                    f"Re-evaluating (attempt {attempts}/{MAX_REVERIFICATION_ATTEMPTS})."
                )
                raw_verdict = self._evaluate_student_code(
                    skill=skill,
                    student_code=student_code,
                    teacher_ref=teacher_ref
                )

        if not verified:
            verified = True
            engine_logger.info(
                f"AGENT 2: Verifier inconclusive after {attempts} attempts "
                f"for rank {skill_rank}. Keeping latest verdict "
                f"'{raw_verdict.get('status')}'. Forcing verified=true. ✅"
            )

        # Post-process: enforce snippet size limits
        raw_verdict = self._enforce_snippet_limits(raw_verdict)

        # Post-process: force recommended_fix from teacher reference
        if raw_verdict.get("status") == "FAIL" and teacher_ref:
            raw_verdict["recommended_fix"] = teacher_ref.get("snippet", "")
            engine_logger.info(
                f"AGENT 2: Forced recommended_fix from teacher reference "
                f"for rank {skill_rank}. ✅"
            )

        engine_logger.info(
            f"AGENT 2: Skill rank {skill_rank} — "
            f"status: {raw_verdict.get('status', 'UNKNOWN')} | "
            f"verified: {verified}"
        )

        raw_verdict["verified"] = verified
        return raw_verdict


    # ── EVALUATE STUDENT CODE AGAINST ONE SKILL ────────────────────
    def _evaluate_student_code(
        self,
        skill:        dict,
        student_code: str,
        teacher_ref:  dict | None = None
    ) -> dict:
        """
        Calls LLM to evaluate student code against one micro skill.
        """
        skill_text   = skill["text"]
        skill_rank   = skill["rank"]
        skill_weight = skill["weight"]

        teacher_context = ""
        if teacher_ref:
            teacher_context = f"""
TEACHER REFERENCE (one correct implementation for this skill):
Line range: {teacher_ref.get('line_start', '?')} to {teacher_ref.get('line_end', '?')}
Code:
{teacher_ref.get('snippet', '')}

IMPORTANT: The teacher reference shows ONE valid approach.
The student may use a different but equally valid approach and still PASS.
When returning recommended_fix for a FAIL, copy the ENTIRE teacher reference
code above exactly as shown — do not truncate, do not extract one line only.
Do NOT include "Lines X-X:" or any prefix — return raw code only.
"""

        prompt = f"""
You are a Senior C Programming Instructor evaluating student C code.

YOU ARE EVALUATING ONLY THIS ONE SKILL IN COMPLETE ISOLATION:
Skill: {skill_text}

Ignore all other skills. Do not apply criteria from any other skill to this verdict.
Focus only on whether the student demonstrated THIS specific skill.

{teacher_context}

Student Code (with line numbers for reference):
{self._number_lines(student_code)}

EVALUATION RULES:
- PASS: Student's code correctly achieves the learning objective of THIS skill
        by ANY valid implementation — not necessarily identical to the teacher's.
- FAIL: Student's code is completely missing THIS skill, produces wrong output
        for this skill, or fails to demonstrate this concept in any recognizable form.
- Alternative valid implementations count as PASS — do NOT penalize different
  but correct approaches.
- If the skill names a SPECIFIC technique (e.g. "temporary variable", "scanf"),
  that exact technique must be present to PASS.
- Be fair — a student who achieves the correct result with a valid approach passes.

SNIPPET RULES (CRITICAL):
- student_snippet must contain ONLY the 2-{MAX_SNIPPET_LINES} lines most directly
  relevant to THIS skill — not the entire program.
- Do NOT include unrelated code like printf, scanf, or main() unless they ARE
  the skill being evaluated.
- For a loop-based skill, include only the loop and its immediately related lines.
- For an input skill like scanf, include only the input loop.
- line_start and line_end must tightly wrap ONLY the relevant lines.

- recommended_fix must be the FULL teacher reference snippet if FAIL,
  or null if PASS.
- Do not invent a fix — only use the teacher reference provided.

HARD CONSTRAINTS:
- Return ONLY a valid JSON object. No explanation. No markdown fences.
- Return format:
{{
  "skill":            "{skill_text}",
  "rank":             {skill_rank},
  "weight":           {skill_weight},
  "status":           "PASS" or "FAIL",
  "line_start":       <integer or null>,
  "line_end":         <integer or null>,
  "student_snippet":  "ONLY 2-{MAX_SNIPPET_LINES} lines directly relevant to this skill",
  "recommended_fix":  "full teacher reference snippet or null if PASS",
  "feedback":         "one sentence technical explanation of the verdict"
}}
"""
        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            agent_label="AGENT 2"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 2: Evaluation failed for rank {skill_rank} — "
                f"LLM returned no response."
            )
            return {
                "skill":           skill_text,
                "rank":            skill_rank,
                "weight":          skill_weight,
                "status":          "FAIL",
                "line_start":      None,
                "line_end":        None,
                "student_snippet": "",
                "recommended_fix": "",
                "feedback":        "Evaluation failed due to an internal error.",
                "verified":        False
            }

        try:
            verdict = json.loads(self._clean_json(content))

            if verdict.get("recommended_fix"):
                verdict["recommended_fix"] = self._strip_line_prefixes(
                    verdict["recommended_fix"]
                )

            return verdict

        except Exception as e:
            engine_logger.error(
                f"AGENT 2: JSON parsing failed for rank {skill_rank}: {e}"
            )
            return {
                "skill":           skill_text,
                "rank":            skill_rank,
                "weight":          skill_weight,
                "status":          "FAIL",
                "line_start":      None,
                "line_end":        None,
                "student_snippet": "",
                "recommended_fix": "",
                "feedback":        "Evaluation failed due to an internal error.",
                "verified":        False
            }


    # ── VERIFY VERDICT AGAINST TEACHER REFERENCE ──────────────────
    def _verify_verdict(
        self,
        verdict:      dict,
        teacher_ref:  dict,
        student_code: str,
        skill_text:   str
    ) -> bool:
        """
        Verifies one skill verdict against teacher reference.
        Returns True if confirmed correct, False if mismatch.
        """
        prompt = f"""
You are a Senior C Programming Instructor verifying ONE evaluation verdict.

YOU ARE VERIFYING ONLY THIS ONE SKILL IN COMPLETE ISOLATION:
Skill: {skill_text}

Do NOT consider any other skills. Do NOT apply criteria from other skills.
Your only job is to confirm whether the verdict for THIS specific skill is correct.

Student Verdict:
- Status: {verdict.get('status')}
- Student Snippet:
{verdict.get('student_snippet', '')}
- Feedback: {verdict.get('feedback', '')}

Teacher Reference (one correct implementation of THIS skill):
Line range: {teacher_ref.get('line_start', '?')} to {teacher_ref.get('line_end', '?')}
Code:
{teacher_ref.get('snippet', '')}

Student Full Code (for context only — evaluate THIS skill in isolation):
{student_code}

VERIFICATION RULES:
- The teacher reference shows ONE valid approach — the student may use a
  different valid approach and still deserve PASS.
- Confirm PASS if the student genuinely achieves the learning objective of
  THIS skill by any valid implementation.
- Confirm FAIL only if the student genuinely fails to demonstrate THIS skill
  concept — not because their approach differs from the teacher's.
- If the skill names a SPECIFIC technique (e.g. "temporary variable", "scanf"),
  verify that exact technique is present.
- If verdict status is correct but student_snippet points to wrong lines,
  return confirmed: false.
- If recommended_fix is missing or incomplete for a FAIL verdict,
  return confirmed: false.
- DO NOT reject a correct verdict just because the reasoning could be worded
  differently — if the STATUS is correct, confirm it.

HARD CONSTRAINTS:
- Answer ONLY with a valid JSON object. No markdown fences.
- Format: {{"confirmed": true}} or {{"confirmed": false, "reason": "specific issue"}}
"""
        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            agent_label="AGENT 2"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 2: Verification call returned no response for '{skill_text}'."
            )
            return False

        try:
            result    = json.loads(self._clean_json(content))
            confirmed = result.get("confirmed", False)

            if not confirmed:
                engine_logger.warning(
                    f"AGENT 2: Verdict not confirmed for '{skill_text}' — "
                    f"{result.get('reason', 'no reason given')}"
                )
            return confirmed

        except Exception as e:
            engine_logger.error(f"AGENT 2: Verification parsing failed: {e}")
            return False


    # ── SNIPPET SIZE ENFORCEMENT ───────────────────────────────────
    def _enforce_snippet_limits(self, verdict: dict) -> dict:
        """
        Python post-processing — guarantees snippet stays within
        MAX_SNIPPET_LINES lines.
        """
        snippet = verdict.get("student_snippet", "")
        if not snippet:
            return verdict

        lines = snippet.split("\n")
        separator = "\n"

        if len(lines) <= 1 and "\\n" in snippet:
            lines = snippet.split("\\n")
            separator = "\\n"

        engine_logger.info(
            f"AGENT 2: Snippet check for rank {verdict.get('rank', '?')} — "
            f"{len(lines)} lines detected, limit is {MAX_SNIPPET_LINES}."
        )

        if len(lines) <= MAX_SNIPPET_LINES:
            return verdict

        truncated_lines = lines[:MAX_SNIPPET_LINES]
        verdict["student_snippet"] = separator.join(truncated_lines)

        line_start = verdict.get("line_start")
        if line_start is not None:
            verdict["line_end"] = line_start + MAX_SNIPPET_LINES - 1

        engine_logger.info(
            f"AGENT 2: Python enforced snippet limit — "
            f"truncated from {len(lines)} to {MAX_SNIPPET_LINES} lines "
            f"for skill rank {verdict.get('rank', '?')}."
        )

        return verdict


    # ── LINE NUMBERING HELPER ──────────────────────────────────────
    @staticmethod
    def _number_lines(code: str) -> str:
        """Prepends line numbers to student code."""
        lines = code.splitlines()
        return "\n".join(
            f"{i + 1}: {line}"
            for i, line in enumerate(lines)
        )


    # ── LINE PREFIX STRIPPER ───────────────────────────────────────
    @staticmethod
    def _strip_line_prefixes(snippet: str) -> str:
        """
        Removes accidental line-number prefixes from LLM-generated snippets.
        Agent 2 owns its own copy — never calls Agent 1 functions directly.
        """
        cleaned_lines = []
        for line in snippet.splitlines():
            stripped = line.strip()
            match = re.match(
                r'^Lines?\s*\d+[\s\u2013\-–]*\d*\s*:\s*(.+)$',
                stripped,
                re.IGNORECASE
            )
            if match:
                stripped = match.group(1)
            elif stripped and stripped[0].isdigit():
                colon_pos = stripped.find(": ")
                if colon_pos != -1 and stripped[:colon_pos].strip().isdigit():
                    stripped = stripped[colon_pos + 2:]
            cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines)


    # ── JSON CLEANER ───────────────────────────────────────────────
    @staticmethod
    def _clean_json(content: str) -> str:
        """Strips markdown fences and whitespace from LLM JSON responses."""
        return content.replace("```json", "").replace("```", "").strip()


# ── GLOBAL INSTANCE (lazy) ─────────────────────────────────────────
agent2: Evaluator | None = None

def get_agent2() -> Evaluator:
    global agent2
    if agent2 is None:
        agent2 = Evaluator()
    return agent2
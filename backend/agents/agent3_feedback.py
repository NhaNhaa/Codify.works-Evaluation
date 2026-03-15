"""
backend/agents/agent3_feedback.py
Agent 3: Feedback Writer — Phase 3.
Runs per student.
Transforms verified technical verdicts from Agent 2 into
teacher-quality, pedagogical feedback.
Tone: encouraging mentor — warm and supportive.
Self-checks each skill feedback before finalizing.
Python sentence truncation guarantees length limits regardless of model.
Outputs both JSON and Markdown via formatter.
All LLM calls go through llm_client.call_llm_with_retry — auto rate limit retry.
"""

import json
import re
from openai import OpenAI
from backend.config.config import get_provider_config, get_model
from backend.config.constants import (
    HIGH_WEIGHT_THRESHOLD,
    STRUCTURE_HIGH_WEIGHT,
    STRUCTURE_LOW_WEIGHT,
    FEEDBACK_TONE,
    FEEDBACK_LANGUAGE,
    FEEDBACK_PASS_DEPTH,
    FEEDBACK_SUMMARY,
    FEEDBACK_PASS_MAX_SENTENCES,
    FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION,
    MAX_SELF_CHECK_ATTEMPTS
)
from backend.utils.logger import engine_logger
from backend.utils.formatter import build_output
from backend.utils.llm_client import call_llm_with_retry


class FeedbackWriter:
    """
    Agent 3 — Feedback Writer.
    Receives verified verdicts from Agent 2.
    Sorts by weight — highest first.
    Writes feedback per skill with self-check loop.
    Enforces sentence limits via Python post-processing.
    Produces final JSON + Markdown report.
    """

    def __init__(self):
        provider = get_provider_config()
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"]
        )
        self.model = get_model("agent3_feedback_writer")
        engine_logger.info("AGENT 3: FeedbackWriter initialized.")


    # ── ENTRY POINT ────────────────────────────────────────────────
    def run(self, evaluation_result: dict) -> dict | None:
        """
        Full Phase 3 pipeline.
        Returns final output dict with JSON truth and Markdown report.
        """
        student_id    = evaluation_result.get("student_id",    "unknown")
        assignment_id = evaluation_result.get("assignment_id", "unknown")
        skills        = evaluation_result.get("skills",        [])

        engine_logger.info(
            f"AGENT 3: Starting Phase 3 for student '{student_id}' "
            f"on assignment '{assignment_id}'."
        )

        if not skills:
            engine_logger.error(
                f"AGENT 3: No skills received for student '{student_id}'."
            )
            return None

        sorted_skills = sorted(
            skills,
            key=lambda s: s.get("weight", 0),
            reverse=True
        )

        enriched_skills = []
        for skill in sorted_skills:
            enriched = self._write_and_selfcheck_feedback(skill)
            enriched_skills.append(enriched)

        final_result = {
            "student_id":    student_id,
            "assignment_id": assignment_id,
            "skills":        enriched_skills
        }

        output = build_output(final_result)

        engine_logger.info(
            f"AGENT 3: Phase 3 complete for student '{student_id}'."
        )

        return output


    # ── WRITE AND SELF-CHECK FEEDBACK PER SKILL ────────────────────
    def _write_and_selfcheck_feedback(self, skill: dict) -> dict:
        """
        Writes feedback for one skill.
        Self-checks quality — max MAX_SELF_CHECK_ATTEMPTS attempts.
        Enforces sentence limits via Python AFTER LLM returns.
        """
        skill_text = skill.get("skill",  "Unknown Skill")
        status     = skill.get("status", "FAIL")
        weight     = skill.get("weight", 1)

        engine_logger.info(
            f"AGENT 3: Writing feedback for skill: '{skill_text}' "
            f"(weight: {weight}, status: {status})"
        )

        feedback = self._generate_feedback(skill)

        attempts = 0
        approved = False

        while attempts < MAX_SELF_CHECK_ATTEMPTS and not approved:
            attempts += 1
            approved  = self._selfcheck_feedback(
                skill_text=skill_text,
                status=status,
                weight=weight,
                feedback=feedback
            )

            if not approved:
                engine_logger.warning(
                    f"AGENT 3: Self-check failed for '{skill_text}'. "
                    f"Rewriting (attempt {attempts}/{MAX_SELF_CHECK_ATTEMPTS})."
                )
                feedback = self._generate_feedback(skill, previous=feedback)

        if not approved:
            engine_logger.warning(
                f"AGENT 3: Max self-check attempts reached for '{skill_text}'. "
                f"Accepting best feedback found."
            )

        feedback = self._enforce_sentence_limits(feedback, status, weight)

        skill["feedback"] = feedback
        return skill


    # ── PYTHON SENTENCE LIMIT ENFORCEMENT ──────────────────────────
    def _enforce_sentence_limits(
        self,
        feedback: str,
        status:   str,
        weight:   int
    ) -> str:
        """
        Python post-processing — guarantees sentence limits.
        This is the final safety net — no LLM dependency.
        """
        if status == "PASS":
            truncated = self._truncate_to_sentences(
                feedback, FEEDBACK_PASS_MAX_SENTENCES
            )
            if truncated != feedback:
                engine_logger.info(
                    f"AGENT 3: Python enforced PASS limit — "
                    f"truncated to {FEEDBACK_PASS_MAX_SENTENCES} sentences."
                )
            return truncated

        sections = self._split_into_sections(feedback)

        if weight >= HIGH_WEIGHT_THRESHOLD:
            expected_sections = len(STRUCTURE_HIGH_WEIGHT)
        else:
            expected_sections = len(STRUCTURE_LOW_WEIGHT)

        truncated_sections = []
        for i, section in enumerate(sections[:expected_sections]):
            truncated = self._truncate_to_sentences(
                section, FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION
            )
            truncated_sections.append(truncated)

        result = "\n\n".join(truncated_sections)

        if result != feedback:
            engine_logger.info(
                f"AGENT 3: Python enforced FAIL limit — "
                f"{expected_sections} sections × "
                f"{FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences each."
            )

        return result

    @staticmethod
    def _split_into_sections(text: str) -> list[str]:
        """Splits feedback text into sections by blank lines."""
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        if not sections:
            return [text.strip()]
        return sections

    @staticmethod
    def _truncate_to_sentences(text: str, max_sentences: int) -> str:
        """Truncates text to max sentences. Preserves code references with periods."""
        if not text or not text.strip():
            return text

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

        if len(sentences) <= max_sentences:
            return text.strip()

        return " ".join(sentences[:max_sentences]).strip()


    # ── GENERATE FEEDBACK FOR ONE SKILL ───────────────────────────
    def _generate_feedback(
        self,
        skill:    dict,
        previous: str = None
    ) -> str:
        """
        Calls LLM to generate teacher-quality feedback for one skill.
        Uses system message to enforce length limits before generation.
        """
        skill_text = skill.get("skill",           "Unknown Skill")
        status     = skill.get("status",           "FAIL")
        weight     = skill.get("weight",           1)
        snippet    = skill.get("student_snippet",  "")
        fix        = skill.get("recommended_fix",  "")
        line_start = skill.get("line_start")
        line_end   = skill.get("line_end")

        # ── Build system message ───────────────────────────────────
        if status == "PASS":
            system_message = (
                f"You are a {FEEDBACK_TONE} writing feedback for a C programming student. "
                f"You write in {FEEDBACK_LANGUAGE} only. "
                f"STRICT LENGTH RULE: Write EXACTLY {FEEDBACK_PASS_MAX_SENTENCES} sentences maximum. "
                f"Count every sentence. Stop after {FEEDBACK_PASS_MAX_SENTENCES} sentences. "
                f"Sentence 1: what the student did correctly with code reference. "
                f"Sentence 2: why it is technically correct. "
                f"Sentence 3: brief encouragement only if natural — skip if not needed. "
                f"No headers. No labels. No bullet points. Plain prose only. "
                f"No filler phrases. Every sentence must have technical content."
            )
        elif weight >= HIGH_WEIGHT_THRESHOLD:
            system_message = (
                f"You are a {FEEDBACK_TONE} writing feedback for a C programming student. "
                f"You write in {FEEDBACK_LANGUAGE} only. "
                f"STRICT LENGTH RULE: Write 3 paragraphs separated by a blank line. "
                f"Each paragraph must contain EXACTLY {FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} "
                f"sentences — no more, no fewer. Count every sentence in every paragraph. "
                f"Paragraph 1 ({FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences): "
                f"what the student did — reference their code and line numbers. "
                f"Paragraph 2 ({FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences): "
                f"why it is wrong — explain the technical consequence clearly. "
                f"Paragraph 3 ({FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences): "
                f"how to fix it — reference the recommended fix provided, do not invent a solution. "
                f"Do NOT write section labels or headers in your output. "
                f"No bullet points. No filler phrases. Every sentence must have technical content."
            )
        else:
            system_message = (
                f"You are a {FEEDBACK_TONE} writing feedback for a C programming student. "
                f"You write in {FEEDBACK_LANGUAGE} only. "
                f"STRICT LENGTH RULE: Write 2 paragraphs separated by a blank line. "
                f"Each paragraph must contain EXACTLY {FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} "
                f"sentences — no more, no fewer. Count every sentence in every paragraph. "
                f"Paragraph 1 ({FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences): "
                f"what the student did — reference their code and line numbers. "
                f"Paragraph 2 ({FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences): "
                f"how to fix it — reference the recommended fix provided, do not invent a solution. "
                f"Do NOT write section labels or headers in your output. "
                f"No bullet points. No filler phrases. Every sentence must have technical content."
            )

        # ── Build previous instruction if rewrite ──────────────────
        previous_instruction = ""
        if previous:
            previous_instruction = (
                f"\nThe previous attempt was REJECTED for being too long. Previous attempt:\n"
                f"{previous}\n"
                f"Rewrite it shorter. Count every sentence. "
                f"Each paragraph must have EXACTLY {FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} "
                f"sentences if FAIL, or {FEEDBACK_PASS_MAX_SENTENCES} total if PASS. "
                f"Do not exceed these limits under any circumstances."
            )

        line_ref = (
            f"Lines {line_start}–{line_end}"
            if line_start and line_end
            else "the relevant section"
        )

        # ── Build user message ─────────────────────────────────────
        user_message = f"""Micro Skill: {skill_text}
Status:      {status}
Weight:      {weight}/10
Location:    {line_ref}

Student Code Snippet:
{snippet if snippet else "Not provided"}

Recommended Fix:
{fix if fix else "N/A — student passed this skill"}
{previous_instruction}
Return ONLY the feedback text. No JSON. No labels. No markdown fences."""

        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.1,
            agent_label="AGENT 3"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 3: Feedback generation failed for '{skill_text}' — "
                f"LLM returned no response."
            )
            return "Feedback could not be generated for this skill."

        feedback = content.strip()
        engine_logger.info(
            f"AGENT 3: Feedback generated for skill: '{skill_text}'."
        )
        return feedback


    # ── SELF-CHECK FEEDBACK QUALITY ────────────────────────────────
    def _selfcheck_feedback(
        self,
        skill_text: str,
        status:     str,
        weight:     int,
        feedback:   str
    ) -> bool:
        """
        Agent 3 checks its own feedback quality.
        Returns True if approved, False if rewrite needed.
        """
        if status == "PASS":
            length_rule = (
                f"3. COUNT every sentence in the feedback. "
                f"PASS feedback must have {FEEDBACK_PASS_MAX_SENTENCES} sentences maximum. "
                f"If there are more than {FEEDBACK_PASS_MAX_SENTENCES} sentences — REJECT immediately."
            )
        else:
            length_rule = (
                f"3. COUNT every sentence in EACH paragraph of the feedback. "
                f"Each paragraph must have exactly {FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} sentences. "
                f"If any paragraph has more than {FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION} "
                f"sentences — REJECT immediately. This is the most important rule."
            )

        prompt = f"""
You are a Senior C Programming Instructor reviewing feedback quality.

Micro Skill: {skill_text}
Status:      {status}
Weight:      {weight}/10

Feedback to review:
{feedback}

Check ALL of the following quality criteria:
1. Is the tone warm and encouraging — not harsh or discouraging?
2. Is it concise — no padding, no filler phrases, every sentence has technical value?
{length_rule}
4. For FAIL — does it clearly describe what went wrong with a code reference?
5. For FAIL weight >= {HIGH_WEIGHT_THRESHOLD} — does it explain WHY it is wrong technically?
6. For FAIL — does it provide a fix without inventing one not from the teacher reference?
7. Does it contain ANY section labels like "WHAT YOU DID:" written directly in the text?
   If yes — REJECT immediately.
8. Would a struggling student understand this and know exactly how to improve?

HARD CONSTRAINTS:
- Criterion 3 (sentence count) is the MOST IMPORTANT — check it first.
- Answer ONLY with a valid JSON object. No markdown fences.
- Format: {{"approved": true}} or {{"approved": false, "reason": "specific issue"}}
"""
        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            agent_label="AGENT 3"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 3: Self-check returned no response for '{skill_text}'."
            )
            return True  # Accept on failure — don't block pipeline

        try:
            result   = json.loads(self._clean_json(content))
            approved = result.get("approved", False)

            if not approved:
                engine_logger.warning(
                    f"AGENT 3: Self-check rejected feedback for "
                    f"'{skill_text}': {result.get('reason', 'no reason given')}"
                )
            return approved

        except Exception as e:
            engine_logger.error(f"AGENT 3: Self-check parsing failed: {e}")
            return True  # Accept on failure — don't block pipeline


    # ── JSON CLEANER ───────────────────────────────────────────────
    @staticmethod
    def _clean_json(content: str) -> str:
        """Strips markdown fences and whitespace from LLM JSON responses."""
        return content.replace("```json", "").replace("```", "").strip()


# ── GLOBAL INSTANCE (lazy) ─────────────────────────────────────────
agent3: FeedbackWriter | None = None

def get_agent3() -> FeedbackWriter:
    global agent3
    if agent3 is None:
        agent3 = FeedbackWriter()
    return agent3
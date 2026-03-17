"""
backend/agents/agent3_feedback.py
Agent 3: Feedback Writer — Phase 3.
Runs per student.
Transforms verified technical verdicts from Agent 2 into
teacher-quality, pedagogical feedback.

Agent 3 writes ONLY the feedback text field.
It does NOT generate section headers or full report structure.
JSON remains the internal truth and Markdown is derived later by formatter.py.

All LLM calls go through llm_client.call_llm_with_retry — auto rate limit retry.
"""

import json
import re

from openai import OpenAI

from backend.config.config import get_provider_config, get_model
from backend.config.constants import (
    HIGH_WEIGHT_THRESHOLD,
    FEEDBACK_TONE,
    FEEDBACK_LANGUAGE,
    FEEDBACK_PASS_MAX_SENTENCES,
    FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION,
    MAX_SELF_CHECK_ATTEMPTS,
)
from backend.utils.logger import engine_logger
from backend.utils.formatter import build_output
from backend.utils.llm_client import call_llm_with_retry


class FeedbackWriter:
    """
    Agent 3 — Feedback Writer.
    Receives verified verdicts from Agent 2.
    Sorts by weight — highest first.
    Writes concise feedback text per skill with self-check loop.
    Enforces sentence limits via Python post-processing.
    Produces final JSON + Markdown report via formatter.
    """

    def __init__(self):
        provider = get_provider_config()
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"]
        )
        self.model = get_model("agent3_feedback_writer")
        engine_logger.info("AGENT 3: FeedbackWriter initialized.")

    def run(self, evaluation_result: dict) -> dict | None:
        """
        Full Phase 3 pipeline.
        Returns final output dict with JSON truth and Markdown report.
        """
        if not isinstance(evaluation_result, dict):
            engine_logger.error("AGENT 3: evaluation_result must be a dict.")
            return None

        student_id = evaluation_result.get("student_id", "unknown")
        assignment_id = evaluation_result.get("assignment_id", "unknown")
        skills = evaluation_result.get("skills", [])

        engine_logger.info(
            f"AGENT 3: Starting Phase 3 for student '{student_id}' "
            f"on assignment '{assignment_id}'."
        )

        if not isinstance(skills, list) or not skills:
            engine_logger.error(
                f"AGENT 3: No skills received for student '{student_id}'."
            )
            return None

        sorted_skills = sorted(
            skills,
            key=lambda skill: (-skill.get("weight", 0), skill.get("rank", 999))
        )

        enriched_skills = []
        for skill in sorted_skills:
            enriched_skill = self._write_and_selfcheck_feedback(dict(skill))
            enriched_skills.append(enriched_skill)

        final_result = {
            "student_id": student_id,
            "assignment_id": assignment_id,
            "skills": enriched_skills
        }

        output = build_output(final_result)

        engine_logger.info(
            f"AGENT 3: Phase 3 complete for student '{student_id}'."
        )

        return output

    def _write_and_selfcheck_feedback(self, skill: dict) -> dict:
        """
        Writes feedback for one skill.
        Self-checks quality — max MAX_SELF_CHECK_ATTEMPTS attempts.
        Enforces sentence limits via Python after LLM returns.
        """
        skill_text = skill.get("skill", skill.get("text", "Unknown Skill"))
        status = str(skill.get("status", "FAIL")).strip().upper()
        weight = skill.get("weight", 1)

        engine_logger.info(
            f"AGENT 3: Writing feedback for skill: '{skill_text}' "
            f"(weight: {weight}, status: {status})"
        )

        feedback = self._generate_feedback(skill)

        attempts = 0
        approved = False

        while attempts < MAX_SELF_CHECK_ATTEMPTS and not approved:
            attempts += 1

            approved = self._selfcheck_feedback(
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

        feedback = self._sanitize_feedback_precision(feedback, skill)
        feedback = self._enforce_sentence_limits(feedback, status)

        skill["feedback"] = feedback
        return skill

    def _generate_feedback(self, skill: dict, previous: str | None = None) -> str:
        """
        Calls LLM to generate concise pedagogical feedback text for one skill.
        Agent 3 writes only the feedback field.
        It does NOT generate sections, headers, or markdown structure.
        """
        deterministic_feedback = self._build_deterministic_feedback(skill)
        if deterministic_feedback is not None:
            engine_logger.info(
                f"AGENT 3: Deterministic feedback used for skill: "
                f"'{skill.get('skill', skill.get('text', 'Unknown Skill'))}'."
            )
            return deterministic_feedback

        skill_text = skill.get("skill", skill.get("text", "Unknown Skill"))
        status = str(skill.get("status", "FAIL")).strip().upper()
        weight = skill.get("weight", 1)
        snippet = skill.get("student_snippet", "")
        fix = skill.get("recommended_fix", "")
        technical_feedback = skill.get("feedback", "")
        line_start = skill.get("line_start")
        line_end = skill.get("line_end")

        line_ref = (
            f"Lines {line_start}-{line_end}"
            if line_start is not None and line_end is not None
            else "Relevant section"
        )

        if status == "PASS":
            max_sentences = FEEDBACK_PASS_MAX_SENTENCES
            task_rule = (
                "Write concise positive technical feedback. "
                "State what the student did correctly and why that local code pattern is correct."
            )
        else:
            max_sentences = FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION
            if weight >= HIGH_WEIGHT_THRESHOLD:
                task_rule = (
                    "Write concise technical corrective feedback. "
                    "State what is wrong and why it is technically wrong. "
                    "Do not include the fix itself because the formatter displays it separately."
                )
            else:
                task_rule = (
                    "Write concise technical corrective feedback. "
                    "State what is wrong briefly. "
                    "Do not include the fix itself because the formatter displays it separately."
                )

        previous_instruction = ""
        if previous:
            previous_instruction = (
                f"\nPrevious rejected attempt:\n{previous}\n"
                f"Rewrite it shorter and clearer. "
                f"Keep within {max_sentences} sentence(s)."
            )

        system_message = (
            f"You are a {FEEDBACK_TONE}. "
            f"You write in {FEEDBACK_LANGUAGE} only.\n\n"
            f"ROLE:\n"
            f"- Write pedagogical feedback for one C programming micro skill.\n"
            f"- Preserve the exact technical meaning of the technical verdict summary.\n"
            f"- Never broaden the claim beyond the isolated skill being evaluated.\n\n"
            f"OUTPUT FORMAT:\n"
            f"- Return ONLY plain feedback text.\n"
            f"- No JSON.\n"
            f"- No markdown fences.\n"
            f"- No section labels.\n"
            f"- No bullet points.\n\n"
            f"HARD CONSTRAINTS:\n"
            f"- Maximum {max_sentences} sentence(s).\n"
            f"- Every sentence must have technical value.\n"
            f"- Be warm, supportive, and precise.\n"
            f"- Refer to the student's code when possible.\n"
            f"- Do not invent any fix.\n"
            f"- Do not mention other skills.\n"
            f"- Do not claim the whole program is correct or incorrect.\n"
            f"- Do not claim the final array/output is correct unless that is explicitly supported.\n"
            f"- Do not say a value moved to the front/back unless the provided evidence clearly supports it.\n"
            f"- Evaluate only this one skill."
        )

        user_message = f"""Micro Skill: {skill_text}
Status: {status}
Weight: {weight}/10
Code Location: {line_ref}

Technical Verdict Summary From Agent 2:
{technical_feedback if technical_feedback else "Not provided"}

Student Code Snippet:
{snippet if snippet else "Not provided"}

Recommended Fix:
{fix if fix else "N/A — student passed this skill"}

Task:
{task_rule}
{previous_instruction}

Return ONLY the feedback text."""

        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            agent_label="AGENT 3"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 3: Feedback generation failed for '{skill_text}' — "
                f"LLM returned no response."
            )
            fallback = self._build_fallback_feedback(skill)
            return fallback

        feedback = content.strip()

        engine_logger.info(
            f"AGENT 3: Feedback generated for skill: '{skill_text}'."
        )
        return feedback

    def _build_deterministic_feedback(self, skill: dict) -> str | None:
        """
        Uses deterministic feedback for known high-risk skill families where
        wording drift can easily create inaccurate claims.
        """
        skill_text = skill.get("skill", skill.get("text", "Unknown Skill"))
        status = str(skill.get("status", "FAIL")).strip().upper()
        family = self._detect_skill_family(skill_text)

        if family == "shift_transform" and status == "PASS":
            return (
                "You correctly used the left-shift copy pattern inside the loop, "
                "and starting at i = 1 keeps arr[i - 1] within bounds. "
                "That demonstrates the required shift step for this skill."
            )

        if family == "shift_transform" and status == "FAIL":
            return (
                "This code does not clearly show the required left-shift copy step for "
                "this skill. The relevant lines should move each value left by copying "
                "arr[i] into arr[i - 1] in a valid loop."
            )

        if family == "shift_temp_safety" and status == "FAIL":
            return (
                "Your loop overwrites the original arr[0] before it is saved, so the "
                "value that should move to the last position is lost. "
                "A temporary variable is needed to preserve that original first element before shifting."
            )

        if family == "shift_temp_safety" and status == "PASS":
            return (
                "You preserved the original first element before shifting the array, "
                "so the value needed for the last position is not lost. "
                "That is the key data-safety step for this skill."
            )

        if family == "scanf_input" and status == "PASS":
            return (
                "You correctly used scanf with &arr[i] inside the loop so each input "
                "is stored directly in the correct array cell. "
                "Running the loop five times matches the requirement to read five integers."
            )

        if family == "scanf_input" and status == "FAIL":
            return (
                "This skill requires using scanf with &arr[i] to read each integer "
                "directly into the array. "
                "The current code does not show that input pattern clearly."
            )

        if family == "array_index" and status == "PASS":
            return (
                "You correctly used array index notation to access the intended array "
                "element for this operation. "
                "That demonstrates the required arr[i] access pattern for this skill."
            )

        if family == "array_index" and status == "FAIL":
            return (
                "This code does not clearly demonstrate the required array index access "
                "pattern for this skill. "
                "The relevant lines should use the correct arr[i] style access for the target element."
            )

        return None

    def _build_fallback_feedback(self, skill: dict) -> str:
        """
        Safe fallback feedback when Agent 3 cannot generate output.
        """
        deterministic_feedback = self._build_deterministic_feedback(skill)
        if deterministic_feedback is not None:
            return deterministic_feedback

        status = str(skill.get("status", "FAIL")).strip().upper()
        if status == "PASS":
            return "Your code correctly demonstrates this skill."
        return "This code does not fully demonstrate the required skill."

    def _sanitize_feedback_precision(self, feedback: str, skill: dict) -> str:
        """
        Python safety net for final wording precision.
        """
        if not feedback or not feedback.strip():
            return self._build_fallback_feedback(skill)

        cleaned = " ".join(feedback.strip().split())
        lowered = cleaned.lower()

        risky_phrases = [
            "whole program",
            "overall program",
            "entire program",
            "full rotation",
            "entire rotation",
            "final output is correct",
            "final array is correct",
            "preserved the last element",
            "wrapped correctly"
        ]

        if any(phrase in lowered for phrase in risky_phrases):
            deterministic_feedback = self._build_deterministic_feedback(skill)
            if deterministic_feedback is not None:
                return deterministic_feedback

        if "front" in lowered:
            skill_text = str(skill.get("skill", skill.get("text", ""))).lower()
            recommended_fix = str(skill.get("recommended_fix", ""))
            if "shift" in skill_text and "arr[4]" in recommended_fix:
                cleaned = re.sub(r"\bfront\b", "last position", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(
                    r"\bplaced at the front\b",
                    "placed in the last position",
                    cleaned,
                    flags=re.IGNORECASE
                )

        return cleaned

    def _enforce_sentence_limits(self, feedback: str, status: str) -> str:
        """
        Python post-processing — guarantees sentence limits.
        PASS uses FEEDBACK_PASS_MAX_SENTENCES.
        FAIL uses FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION because formatter
        shows feedback as the explanation text only.
        """
        if not feedback or not feedback.strip():
            return feedback

        if status == "PASS":
            limit = FEEDBACK_PASS_MAX_SENTENCES
        else:
            limit = FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION

        truncated = self._truncate_to_sentences(feedback, limit)

        if truncated != feedback:
            engine_logger.info(
                f"AGENT 3: Python enforced sentence limit — "
                f"truncated to {limit} sentence(s)."
            )

        return truncated

    @staticmethod
    def _truncate_to_sentences(text: str, max_sentences: int) -> str:
        """
        Truncates text to max_sentences using simple sentence boundaries.
        """
        if not text or not text.strip():
            return text

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

        if len(sentences) <= max_sentences:
            return text.strip()

        return " ".join(sentences[:max_sentences]).strip()

    @staticmethod
    def _detect_skill_family(skill_text: str) -> str:
        """
        Lightweight skill family detector for safer feedback wording.
        """
        lowered = str(skill_text).lower()

        if "temporary variable" in lowered or ("shift" in lowered and "avoid losing" in lowered):
            return "shift_temp_safety"

        if "shift" in lowered:
            return "shift_transform"

        if "scanf" in lowered:
            return "scanf_input"

        if "arr[i]" in lowered or "index" in lowered:
            return "array_index"

        return "other"

    def _selfcheck_feedback(
        self,
        skill_text: str,
        status: str,
        weight: int,
        feedback: str
    ) -> bool:
        """
        Agent 3 checks its own feedback quality.
        Returns True if approved, False if rewrite needed.
        """
        max_sentences = (
            FEEDBACK_PASS_MAX_SENTENCES
            if status == "PASS"
            else FEEDBACK_FAIL_MAX_SENTENCES_PER_SECTION
        )

        prompt = f"""
You are reviewing feedback quality for ONE C programming micro skill.

Micro Skill: {skill_text}
Status: {status}
Weight: {weight}/10

Feedback to review:
{feedback}

Check ALL criteria:
1. Is the tone warm and supportive?
2. Is it concise and technically useful?
3. Does it stay within {max_sentences} sentence(s)?
4. Does it avoid section labels, bullet points, and markdown formatting?
5. Does it focus only on this one skill?
6. Does it avoid claiming the whole program or final output is correct/incorrect?
7. For FAIL, does it explain the problem without inventing a fix?
8. For PASS, does it explain why the shown code pattern is correct?

HARD CONSTRAINTS:
- Answer ONLY with a valid JSON object.
- No markdown fences.
- Format: {{"approved": true}} or {{"approved": false, "reason": "specific issue"}}
"""
        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You review C programming feedback quality and return JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            agent_label="AGENT 3"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 3: Self-check returned no response for '{skill_text}'."
            )
            return True

        try:
            result = json.loads(self._clean_json(content))
            approved = result.get("approved", False)

            if not approved:
                engine_logger.warning(
                    f"AGENT 3: Self-check rejected feedback for "
                    f"'{skill_text}': {result.get('reason', 'no reason given')}"
                )

            return bool(approved)

        except Exception as e:
            engine_logger.error(f"AGENT 3: Self-check parsing failed: {e}")
            return True

    @staticmethod
    def _clean_json(content: str) -> str:
        """
        Strips markdown fences and whitespace from LLM JSON responses.
        """
        if not isinstance(content, str):
            return ""

        cleaned = content.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()


# ── GLOBAL INSTANCE (lazy) ─────────────────────────────────────────
agent3: FeedbackWriter | None = None


def get_agent3() -> FeedbackWriter:
    global agent3
    if agent3 is None:
        agent3 = FeedbackWriter()
    return agent3
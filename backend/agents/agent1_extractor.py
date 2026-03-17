"""
backend/agents/agent1_extractor.py
Agent 1: Skill Extractor — Phase 1.
Runs once per assignment.
Stores micro_skills + teacher_references into ChromaDB via rag_pipeline.

Step 2 uses skill_parser.py (Python — no LLM) for exact skill extraction.
LLM is only used for:
  - Generating skills from prose (Tier 3 only)
  - Finding MISSING skills when parser found < MIN_MICRO_SKILLS
  - Generating teacher reference snippets

All Python validation, deduplication, generic rejection, assignment-context filtering,
ranking, and weighting live in agent1_validators.py — imported as 'validators'.
All LLM calls go through llm_client.call_llm_with_retry — auto rate limit retry.
"""

import json

from openai import OpenAI

from backend.config.config import get_provider_config, get_model
from backend.config.constants import (
    MIN_MICRO_SKILLS,
    MAX_MICRO_SKILLS,
    SKILL_QUALITY_RULES,
)
from backend.rag.rag_pipeline import rag_pipeline
from backend.utils.logger import engine_logger
from backend.utils.security import safe_read_file
from backend.utils.skill_parser import extract_skills
from backend.utils.llm_client import call_llm_with_retry
from backend.agents import agent1_validators as validators


class SkillExtractor:
    """
    Agent 1 — Skill Extractor.
    Reads 3 assignment files sequentially.
    Extracts (Python) or generates (LLM) micro skills.
    Delegates validation, dedup, generic rejection, context filtering,
    and ranking to agent1_validators.
    Generates and stores teacher reference snippets via LLM.
    """

    def __init__(self):
        provider = get_provider_config()
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"]
        )
        self.model = get_model("agent1_skill_extractor")
        engine_logger.info("AGENT 1: SkillExtractor initialized.")

    def run(
        self,
        assignment_id: str,
        instructions_path: str,
        starter_path: str,
        teacher_path: str,
        force_regenerate: bool = False
    ) -> bool:
        """
        Full Phase 1 pipeline.
        Returns True if skills and references are stored successfully.
        """
        if not self._is_valid_assignment_id(assignment_id):
            return False

        engine_logger.info(
            f"AGENT 1: Starting Phase 1 for assignment '{assignment_id}'."
        )

        assignment_exists = rag_pipeline.assignment_exists(assignment_id)

        if assignment_exists and not force_regenerate:
            engine_logger.info(
                f"AGENT 1: Skills already exist for '{assignment_id}'. "
                f"Skipping Phase 1."
            )
            return True

        if assignment_exists and force_regenerate:
            engine_logger.info(
                f"AGENT 1: force_regenerate=True for '{assignment_id}'. "
                f"Existing assignment data will be replaced."
            )

        instructions = safe_read_file(instructions_path, ".md")
        starter = safe_read_file(starter_path, ".c")
        teacher = safe_read_file(teacher_path, ".c")

        if not all([instructions, starter, teacher]):
            engine_logger.error("AGENT 1: One or more input files failed validation.")
            return False

        engine_logger.info("AGENT 1: Step 2 — Extracting skills from instructions.md.")
        initial_skills, from_parser = self._get_initial_skills(instructions)

        if initial_skills is None:
            engine_logger.error("AGENT 1: instructions.md is invalid or empty. Aborting.")
            return False

        if len(initial_skills) > 0:
            engine_logger.info(
                f"AGENT 1: Step 2 complete — {len(initial_skills)} skills "
                f"({'parser-extracted' if from_parser else 'LLM-generated'})."
            )
        else:
            engine_logger.info(
                "AGENT 1: Step 2 — no initial skills survived validation. "
                "Cross-check will generate with code context."
            )

        if from_parser and len(initial_skills) >= MIN_MICRO_SKILLS:
            engine_logger.info(
                f"AGENT 1: Teacher wrote {len(initial_skills)} skills "
                f"(>= {MIN_MICRO_SKILLS}). Skipping LLM cross-check."
            )
            all_skills = list(initial_skills)
        else:
            engine_logger.info("AGENT 1: Steps 3+4 — Checking for missing skills.")
            all_skills = self._find_missing_skills(
                existing_skills=initial_skills,
                instructions=instructions,
                starter=starter,
                teacher=teacher
            )

        if len(all_skills) < MIN_MICRO_SKILLS:
            engine_logger.error(
                f"AGENT 1: Only {len(all_skills)} valid skills found "
                f"after all processing. Minimum required is "
                f"{MIN_MICRO_SKILLS}. Aborting."
            )
            return False

        engine_logger.info("AGENT 1: Steps 5+6 — Ranking and assigning weights (Python).")
        ranked_skills = validators.python_rank_and_weight(all_skills)

        if not ranked_skills:
            engine_logger.error("AGENT 1: Ranking and weight assignment failed.")
            return False

        engine_logger.info("AGENT 1: Step 7 — Validating skills.")
        self._log_final_skills(ranked_skills)

        engine_logger.info("AGENT 1: Step 8 — Generating teacher reference snippets.")
        teacher_refs = self._generate_teacher_references(
            skills=ranked_skills,
            teacher=teacher
        )

        if not teacher_refs:
            engine_logger.error("AGENT 1: Teacher reference generation failed.")
            return False

        if len(teacher_refs) != len(ranked_skills):
            engine_logger.error(
                f"AGENT 1: Teacher reference count mismatch. "
                f"Expected {len(ranked_skills)}, got {len(teacher_refs)}."
            )
            return False

        engine_logger.info("AGENT 1: Step 9 — Storing micro skills and teacher references.")

        skills_stored = rag_pipeline.store_micro_skills(
            skills=ranked_skills,
            assignment_id=assignment_id,
            force_regenerate=force_regenerate
        )

        if not skills_stored:
            engine_logger.error("AGENT 1: Failed to store micro skills in ChromaDB.")
            return False

        refs_stored = rag_pipeline.store_teacher_references(
            references=teacher_refs,
            assignment_id=assignment_id,
            force_regenerate=force_regenerate
        )

        if not refs_stored:
            engine_logger.error(
                "AGENT 1: Failed to store teacher references. "
                "Rolling back assignment data to avoid partial state."
            )
            rag_pipeline.clear_assignment(assignment_id)
            return False

        engine_logger.info(
            f"AGENT 1: Phase 1 complete for assignment '{assignment_id}'."
        )
        return True

    def _is_valid_assignment_id(self, assignment_id: str) -> bool:
        """
        Validates assignment_id before Phase 1 starts.
        """
        if not assignment_id or not str(assignment_id).strip():
            engine_logger.error("AGENT 1: assignment_id is missing or blank.")
            return False
        return True

    def _get_initial_skills(self, instructions: str) -> tuple[list[str] | None, bool]:
        """
        Step 2 — Two-path extraction:
        PATH A: Python parser extracts skills exactly as written.
        PATH B: LLM generates skills from prose (Tier 3 only).
        """
        parsed = extract_skills(instructions)

        if parsed is None:
            return None, False

        if len(parsed) > 0:
            engine_logger.info(
                f"AGENT 1: Step 2 — Python parser extracted "
                f"{len(parsed)} skills exactly. No LLM used."
            )
            return parsed, True

        engine_logger.info(
            "AGENT 1: Step 2 — No skill list found. "
            "Using LLM to generate skills from prose."
        )
        generated = self._generate_skills_from_prose(instructions)
        return generated, False

    def _generate_skills_from_prose(self, instructions: str) -> list[str] | None:
        """
        Called only when instructions.md contains no extractable list.
        Generates teacher-quality skills from prose description.
        Validates, deduplicates, rejects generic, and filters by assignment context.
        Returns [] if all skills are rejected — signals cross-check fallback.
        """
        quality_rules = "\n".join(f"- {rule}" for rule in SKILL_QUALITY_RULES)

        prompt = f"""
You are a Senior C Programming Instructor extracting Micro Skills
from an assignment description written in prose.

{validators.SKILL_FORMULA_PROMPT}

OUTPUT FORMAT:
- Return ONLY a valid JSON list of strings.
- No explanation.
- No markdown fences.

SKILL QUALITY RULES:
{quality_rules}

HARD CONSTRAINTS:
- Between {MIN_MICRO_SKILLS} and {MAX_MICRO_SKILLS} skills only.
- Every skill must start with one of: {', '.join(validators.VALID_SKILL_VERBS)}
- Every skill must reference specific C syntax, variable names, or techniques.
- Each skill must be between {validators.MIN_SKILL_WORDS} and {validators.MAX_SKILL_WORDS} words.
- Each skill must test ONE complete concept.
- Do NOT create two skills that would evaluate the same lines of code.
- Do NOT split the shift operation into two transform skills.
- If one skill already says the student must shift elements and handle the 1st cell,
  do NOT add another skill like "Handle wrapping the first element to last position".
- Boundary/wrap handling for the first element belongs inside the main shift skill,
  not as a second transform skill.
- Do NOT create separate skills for different array index notations when they are part of the same operation.
- Do NOT generate skills about basic language constructs unless they are the specific focus of this assignment.
- Output/display/formatting skills (printf, separators, commas, braces, formatting) are NOT valid
  unless instructions.md explicitly lists output/printing as a learning objective.
- Prefer core algorithm, data safety, and essential input skills.
- Safety skills must describe the pattern generically.
- If you cannot identify any evaluable programming skills, return: []

ASSIGNMENT DESCRIPTION:
{instructions}
"""
        result = self._call_llm_for_list(prompt, "prose skill generation")

        if result is None:
            return None

        if len(result) == 0:
            engine_logger.error(
                "AGENT 1: Step 2 — LLM could not identify any evaluable skills."
            )
            return None

        validated = validators.validate_and_fix_skills(result)
        if not validated:
            engine_logger.error(
                "AGENT 1: All LLM-generated skills failed formula validation."
            )
            return None

        deduplicated = validators.deduplicate_skills(validated)
        if not deduplicated:
            engine_logger.error(
                "AGENT 1: All LLM-generated skills were duplicates."
            )
            return None

        non_generic = validators.reject_generic_skills(deduplicated)
        if not non_generic:
            engine_logger.warning(
                "AGENT 1: All LLM-generated skills were generic. "
                "Returning empty list for cross-check fallback."
            )
            return []

        context_filtered = validators.filter_skills_by_assignment_context(
            skills=non_generic,
            instructions=instructions
        )
        context_filtered = validators.deduplicate_skills(context_filtered)

        if not context_filtered:
            engine_logger.warning(
                "AGENT 1: All LLM-generated skills were rejected by assignment context. "
                "Returning empty list for cross-check fallback."
            )
            return []

        return context_filtered

    def _find_missing_skills(
        self,
        existing_skills: list[str],
        instructions: str,
        starter: str,
        teacher: str
    ) -> list[str]:
        """
        Asks LLM one question: what core skills are missing?
        Validates, deduplicates, rejects generic, and filters by assignment context.
        """
        if len(existing_skills) >= MAX_MICRO_SKILLS:
            engine_logger.info(
                f"AGENT 1: Already at {MAX_MICRO_SKILLS} skills. "
                f"Skipping cross-check."
            )
            return list(existing_skills)

        spots_available = MAX_MICRO_SKILLS - len(existing_skills)

        concept_list = "\n".join(
            f"  {index + 1}. {skill}" for index, skill in enumerate(existing_skills)
        )

        existing_context = (
            f"A teacher has defined these micro skills for a C assignment:\n"
            f"{concept_list}\n\n"
            f"Review the starter code and teacher solution below.\n"
            f"Are there any CORE ALGORITHMIC skills missing that are NOT already\n"
            f"covered by any of the {len(existing_skills)} skills above?"
            if existing_skills else
            "No micro skills have been defined yet for this C assignment.\n\n"
            "Review the starter code and teacher solution below.\n"
            "Generate the CORE ALGORITHMIC skills needed to evaluate\n"
            "student submissions for this assignment."
        )

        prompt = f"""
You are a Senior C Programming Instructor.

{existing_context}

{validators.SKILL_FORMULA_PROMPT}

OUTPUT FORMAT:
- Return ONLY a valid JSON list of strings.
- No explanation.
- No markdown fences.

RULES:
- A skill is MISSING only if it tests a genuinely different concept not covered by any existing skill.
- Do NOT add a skill that overlaps with an existing skill.
- Do NOT add a skill that evaluates the same lines of code as an existing skill.
- Do NOT split the shift operation into two transform skills.
- If an existing skill already includes shifting plus handling the 1st cell,
  do NOT add a separate wrap/boundary skill.
- Only core algorithmic skills — not generic programming.
- Do NOT generate skills about basic language constructs unless they are the specific focus of this assignment.
- Output/display/formatting skills (printf, separators, commas, braces, formatting) are NOT valid
  unless instructions.md explicitly lists output/printing as a learning objective.
- Maximum {spots_available} new skills allowed.
- Every new skill must start with one of: {', '.join(validators.VALID_SKILL_VERBS)}
- Every new skill must be between {validators.MIN_SKILL_WORDS} and {validators.MAX_SKILL_WORDS} words.
- Safety skills must describe the pattern generically.
- If nothing is missing, return: []

STARTER CODE:
{starter}

TEACHER SOLUTION:
{teacher}

ASSIGNMENT INSTRUCTIONS:
{instructions}
"""
        new_skills = self._call_llm_for_list(prompt, "missing skills check")

        if not new_skills:
            engine_logger.info("AGENT 1: No missing skills found.")
            return list(existing_skills)

        validated_new = validators.validate_and_fix_skills(new_skills)
        if not validated_new:
            return list(existing_skills)

        deduplicated_new = validators.deduplicate_skills(validated_new)
        non_generic_new = validators.reject_generic_skills(deduplicated_new)
        context_filtered_new = validators.filter_skills_by_assignment_context(
            skills=non_generic_new,
            instructions=instructions
        )
        context_filtered_new = validators.deduplicate_skills(context_filtered_new)

        genuinely_new = []
        for new_skill in context_filtered_new:
            is_duplicate = False

            for existing in existing_skills:
                if validators.is_same_concept(existing, new_skill):
                    engine_logger.info(
                        f"AGENT 1: Rejected duplicate new skill: '{new_skill}' "
                        f"(overlaps with existing: '{existing}')"
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                for accepted in genuinely_new:
                    if validators.is_same_concept(accepted, new_skill):
                        engine_logger.info(
                            f"AGENT 1: Rejected duplicate new skill: '{new_skill}' "
                            f"(overlaps with accepted: '{accepted}')"
                        )
                        is_duplicate = True
                        break

            if not is_duplicate:
                genuinely_new.append(new_skill)
                engine_logger.info(f"AGENT 1: Accepted new skill: '{new_skill}'")

        result = list(existing_skills) + genuinely_new[:spots_available]
        result = validators.deduplicate_skills(result)

        engine_logger.info(
            f"AGENT 1: Cross-check complete — "
            f"{len(existing_skills)} original + {len(genuinely_new[:spots_available])} new = "
            f"{len(result)} total skills."
        )
        return result

    def _log_final_skills(self, skills: list[dict]) -> None:
        """
        Step 7 — logs final skills for audit.
        """
        for skill in skills:
            engine_logger.info(
                f"AGENT 1: Final skill — "
                f"rank={skill['rank']} weight={skill['weight']} "
                f"text='{skill['text']}'"
            )

        total = sum(skill["weight"] for skill in skills)
        engine_logger.info(
            f"AGENT 1: Step 7 — {len(skills)} skills validated. "
            f"Weight sum = {total}."
        )

    def _generate_teacher_references(
        self,
        skills: list[dict],
        teacher: str
    ) -> list[dict]:
        """
        Generates teacher reference snippet per micro skill
        from teacher_correction_code.c.
        Always PASS — teacher code is the gold standard.
        """
        numbered_teacher = validators.number_lines(teacher)
        references = []

        for skill in skills:
            prompt = f"""
You are a Senior C Programming Instructor analyzing correct C code.

Find the MINIMAL code block that implements this specific Micro Skill
in the teacher's solution.

OUTPUT FORMAT:
- Return ONLY a valid JSON object.
- No explanation.
- No markdown fences.

Micro Skill: {skill["text"]}

Teacher Solution Code (with line numbers):
{numbered_teacher}

HARD CONSTRAINTS:
- Return ONLY the lines that directly implement this skill.
- Do NOT include unrelated code unless inseparable from the skill.
- Do NOT return the entire program.
- The snippet must be complete enough for a student to replicate the fix.
- line_start and line_end must match the actual line numbers shown above.
- The snippet field must contain ONLY raw code — no line number prefixes, labels, or prose.
- Preserve original line breaks exactly as they appear in the source code.
- Format:
{{
  "snippet": "line1\\nline2\\nline3",
  "line_start": <integer>,
  "line_end": <integer>
}}
"""
            content = call_llm_with_retry(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                agent_label="AGENT 1"
            )

            if content is None:
                engine_logger.error(
                    f"AGENT 1: Teacher reference generation failed "
                    f"for rank {skill['rank']} — LLM returned no response."
                )
                return []

            try:
                ref_data = json.loads(validators.clean_json(content))

                snippet = validators.strip_line_prefixes(
                    ref_data.get("snippet", "")
                ).strip()

                line_start = ref_data.get("line_start", 0)
                line_end = ref_data.get("line_end", 0)

                if not snippet:
                    engine_logger.error(
                        f"AGENT 1: Empty teacher snippet for rank {skill['rank']}."
                    )
                    return []

                if not isinstance(line_start, int) or not isinstance(line_end, int):
                    engine_logger.error(
                        f"AGENT 1: Invalid teacher line numbers for rank {skill['rank']}."
                    )
                    return []

                if line_start <= 0 or line_end <= 0 or line_start > line_end:
                    engine_logger.error(
                        f"AGENT 1: Teacher line range invalid for rank {skill['rank']}: "
                        f"{line_start}-{line_end}"
                    )
                    return []

                references.append({
                    "rank": skill["rank"],
                    "snippet": snippet,
                    "line_start": line_start,
                    "line_end": line_end
                })

                engine_logger.info(
                    f"AGENT 1: Teacher reference generated for "
                    f"skill rank {skill['rank']}."
                )

            except Exception as e:
                engine_logger.error(
                    f"AGENT 1: Teacher reference parsing failed "
                    f"for rank {skill['rank']}: {e}"
                )
                return []

        return references

    def _call_llm_for_list(self, prompt: str, step_label: str) -> list[str] | None:
        """
        Calls LLM via retry wrapper and parses a JSON list response.
        Returns None on error, [] on empty result.
        """
        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            agent_label="AGENT 1"
        )

        if content is None:
            engine_logger.error(
                f"AGENT 1: LLM returned no response for {step_label}."
            )
            return None

        try:
            result = json.loads(validators.clean_json(content))

            if not isinstance(result, list):
                engine_logger.error(
                    f"AGENT 1: Expected list from {step_label}, "
                    f"got {type(result)}."
                )
                return None

            cleaned_result = []
            for item in result:
                if isinstance(item, str) and item.strip():
                    cleaned_result.append(item.strip())

            engine_logger.info(
                f"AGENT 1: {step_label} — {len(cleaned_result)} items returned."
            )
            return cleaned_result

        except Exception as e:
            engine_logger.error(f"AGENT 1: JSON parsing failed for {step_label}: {e}")
            return None


# ── GLOBAL INSTANCE (lazy) ─────────────────────────────────────────
agent1: SkillExtractor | None = None


def get_agent1() -> SkillExtractor:
    global agent1
    if agent1 is None:
        agent1 = SkillExtractor()
    return agent1
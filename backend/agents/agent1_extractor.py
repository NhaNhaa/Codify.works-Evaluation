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

All Python validation, ranking, deduplication, and generic rejection
logic lives in agent1_validators.py — imported as 'validators'.
All LLM calls go through llm_client.call_llm_with_retry — auto rate limit retry.
"""

import json
from openai import OpenAI
from backend.config.config import get_provider_config, get_model
from backend.config.constants import (
    MIN_MICRO_SKILLS,
    MAX_MICRO_SKILLS,
    SKILL_QUALITY_RULES
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
    Delegates validation, dedup, ranking to agent1_validators.
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


    # ── ENTRY POINT ────────────────────────────────────────────────
    def run(
        self,
        assignment_id:     str,
        instructions_path: str,
        starter_path:      str,
        teacher_path:      str,
        force_regenerate:  bool = False
    ) -> bool:
        """
        Full Phase 1 pipeline.
        Returns True if skills and references stored successfully.
        """
        engine_logger.info(
            f"AGENT 1: Starting Phase 1 for assignment '{assignment_id}'."
        )

        # STEP 1 — Check ChromaDB first
        if not force_regenerate and rag_pipeline.assignment_exists(assignment_id):
            engine_logger.info(
                f"AGENT 1: Skills already exist for '{assignment_id}'. "
                f"Skipping Phase 1."
            )
            return True

        # Read all 3 files safely
        instructions = safe_read_file(instructions_path, ".md")
        starter      = safe_read_file(starter_path,      ".c")
        teacher      = safe_read_file(teacher_path,       ".c")

        if not all([instructions, starter, teacher]):
            engine_logger.error("AGENT 1: One or more input files failed validation.")
            return False

        # STEP 2 — Extract or generate initial skills
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

        # STEPS 3+4 — Cross-check only when needed
        if from_parser and len(initial_skills) >= MIN_MICRO_SKILLS:
            engine_logger.info(
                f"AGENT 1: Teacher wrote {len(initial_skills)} skills "
                f"(>= {MIN_MICRO_SKILLS}). Skipping LLM cross-check — "
                f"trusting teacher's skills. ✅"
            )
            all_skills = list(initial_skills)
        else:
            engine_logger.info(
                "AGENT 1: Steps 3+4 — Checking for missing skills."
            )
            all_skills = self._find_missing_skills(
                existing_skills=initial_skills,
                instructions=instructions,
                starter=starter,
                teacher=teacher
            )

        # Validate minimum after all processing
        if len(all_skills) < MIN_MICRO_SKILLS:
            engine_logger.error(
                f"AGENT 1: Only {len(all_skills)} valid skills found "
                f"after all processing. Minimum required is "
                f"{MIN_MICRO_SKILLS}. Aborting."
            )
            return False

        # STEP 5+6 — Rank (Python) + assign weights (Python)
        engine_logger.info("AGENT 1: Steps 5+6 — Ranking and assigning weights (Python).")
        ranked_skills = validators.python_rank_and_weight(all_skills)
        if not ranked_skills:
            engine_logger.error("AGENT 1: Ranking and weight assignment failed.")
            return False

        # STEP 7 — Validation (simplified — log only, no LLM modification)
        engine_logger.info("AGENT 1: Step 7 — Validating skills.")
        self._log_final_skills(ranked_skills)

        # STEP 8 — Generate teacher reference snippets
        engine_logger.info("AGENT 1: Step 8 — Generating teacher reference snippets.")
        teacher_refs = self._generate_teacher_references(
            skills=ranked_skills,
            teacher=teacher
        )
        if not teacher_refs:
            engine_logger.error("AGENT 1: Teacher reference generation failed.")
            return False

        # STEP 9 — Store both collections in ChromaDB
        engine_logger.info("AGENT 1: Step 9 — Storing micro skills and teacher references.")
        skills_stored = rag_pipeline.store_micro_skills(
            skills=ranked_skills,
            assignment_id=assignment_id,
            force_regenerate=force_regenerate
        )
        refs_stored = rag_pipeline.store_teacher_references(
            references=teacher_refs,
            assignment_id=assignment_id,
            force_regenerate=force_regenerate
        )

        if skills_stored and refs_stored:
            engine_logger.info(
                f"AGENT 1: Phase 1 complete for assignment '{assignment_id}'."
            )
            return True

        engine_logger.error("AGENT 1: Failed to store data in ChromaDB.")
        return False


    # ── STEP 2: GET INITIAL SKILLS ─────────────────────────────────
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
                f"{len(parsed)} skills exactly. No LLM used. ✅"
            )
            return parsed, True

        engine_logger.info(
            "AGENT 1: Step 2 — No skill list found. "
            "Using LLM to generate skills from prose."
        )
        generated = self._generate_skills_from_prose(instructions)
        return generated, False


    # ── STEP 2 PATH B: LLM GENERATION FROM PROSE ──────────────────
    def _generate_skills_from_prose(self, instructions: str) -> list[str] | None:
        """
        Called only when instructions.md contains no extractable list.
        Generates teacher-quality skills from prose description.
        Validates, deduplicates, and rejects generic via validators.
        Returns [] if all skills rejected — signals cross-check fallback.
        """
        quality_rules = "\n".join(f"- {r}" for r in SKILL_QUALITY_RULES)

        prompt = f"""
You are a Senior C Programming Instructor extracting Micro Skills
from an assignment description written in prose (no skill list provided).

{validators.SKILL_FORMULA_PROMPT}

SKILL QUALITY RULES:
{quality_rules}

HARD CONSTRAINTS:
- Between {MIN_MICRO_SKILLS} and {MAX_MICRO_SKILLS} skills only.
- Every skill must start with one of: {', '.join(validators.VALID_SKILL_VERBS)}
- Every skill must reference specific C syntax, variable names, or
  techniques from the assignment.
- Each skill must be between {validators.MIN_SKILL_WORDS} and {validators.MAX_SKILL_WORDS} words.
- Each skill must test ONE complete concept — combine the operation
  and its edge case into a single skill.
- Do NOT create two skills that would evaluate the same lines of code.
- Do NOT create separate skills for different array index notations
  (arr[i-1], arr[i+1], arr[i]) when they are part of the same operation.
- Do NOT generate skills about basic language constructs (loops, variables,
  conditionals) unless they are the specific focus of this assignment.
- Safety skills (temp variable, data preservation) must describe the PATTERN
  generically — do NOT reference the same specific data element (e.g., "first
  element", "1st cell") that a transform skill already handles.
- If you cannot identify any evaluable programming skills, return: []
- Return ONLY a valid JSON list of strings. No explanation. No markdown fences.

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

        # Validate → Deduplicate → Reject generic (all Python)
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

        return non_generic


    # ── STEPS 3+4: FIND MISSING SKILLS ONLY ───────────────────────
    def _find_missing_skills(
        self,
        existing_skills: list[str],
        instructions:    str,
        starter:         str,
        teacher:         str
    ) -> list[str]:
        """
        Asks LLM ONE question: "What core skills are MISSING?"
        Validates, deduplicates, and rejects generic via validators.
        """
        if len(existing_skills) >= MAX_MICRO_SKILLS:
            engine_logger.info(
                f"AGENT 1: Already at {MAX_MICRO_SKILLS} skills. "
                f"Skipping cross-check."
            )
            return list(existing_skills)

        spots_available = MAX_MICRO_SKILLS - len(existing_skills)

        concept_list = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(existing_skills)
        )

        existing_context = (
            f"A teacher has defined these micro skills for a C assignment:\n"
            f"{concept_list}\n\n"
            f"Review the starter code and teacher solution below.\n"
            f"Are there any CORE ALGORITHMIC skills MISSING that are NOT already\n"
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

RULES:
- A skill is MISSING only if it tests a genuinely different concept
  not covered by ANY existing skill — even partially.
- Do NOT add a skill that overlaps with an existing skill.
  Example: if "Avoid losing data during shifting, using a temporary variable"
  exists, do NOT add "Use temp = arr[0] to preserve first element" — same concept.
- Do NOT add a skill that evaluates the same lines of code as an existing skill.
- Only core algorithmic skills — not generic programming (loops, variables).
- Do NOT generate skills about basic language constructs (loops, variables,
  conditionals) unless they are the specific focus of this assignment.
- Maximum {spots_available} new skills allowed.
- Every new skill must start with one of: {', '.join(validators.VALID_SKILL_VERBS)}
- Every new skill must be between {validators.MIN_SKILL_WORDS} and {validators.MAX_SKILL_WORDS} words.
- Safety skills (temp variable, data preservation) must describe the PATTERN
  generically — do NOT reference the same specific data element (e.g., "first
  element", "1st cell") that a transform skill already handles.

STARTER CODE:
{starter}

TEACHER SOLUTION:
{teacher}

ASSIGNMENT INSTRUCTIONS:
{instructions}

HARD CONSTRAINTS:
- If nothing is missing, return: []
- If something is missing, return ONLY the new skill texts as a JSON list.
- Do NOT include any existing skills in your response.
- Return ONLY a valid JSON list of strings. No explanation. No markdown fences.
"""
        new_skills = self._call_llm_for_list(prompt, "missing skills check")

        if not new_skills:
            engine_logger.info("AGENT 1: No missing skills found. ✅")
            return list(existing_skills)

        validated_new = validators.validate_and_fix_skills(new_skills)

        genuinely_new = []
        for new_skill in validated_new:
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
                engine_logger.info(
                    f"AGENT 1: Accepted new skill: '{new_skill}'"
                )

        non_generic_new = validators.reject_generic_skills(genuinely_new)

        result = list(existing_skills) + non_generic_new[:spots_available]
        engine_logger.info(
            f"AGENT 1: Cross-check complete — "
            f"{len(existing_skills)} original + {len(non_generic_new)} new = "
            f"{len(result)} total skills."
        )
        return result


    # ── STEP 7: LOG FINAL SKILLS (NO LLM) ─────────────────────────
    def _log_final_skills(self, skills: list[dict]) -> None:
        """Step 7 — logs final skills for audit."""
        for s in skills:
            engine_logger.info(
                f"AGENT 1: Final skill — "
                f"rank={s['rank']} weight={s['weight']} "
                f"text='{s['text']}'"
            )
        total = sum(s["weight"] for s in skills)
        engine_logger.info(
            f"AGENT 1: Step 7 — {len(skills)} skills validated. "
            f"Weight sum = {total}. ✅"
        )


    # ── STEP 8: GENERATE TEACHER REFERENCES ───────────────────────
    def _generate_teacher_references(
        self,
        skills:  list[dict],
        teacher: str
    ) -> list[dict]:
        """
        Generates teacher reference snippet per micro skill
        from teacher_correction_code.c.
        Always PASS — teacher code is the gold standard.
        """
        numbered_teacher = validators.number_lines(teacher)
        references       = []

        for skill in skills:
            prompt = f"""
You are a Senior C Programming Instructor analyzing correct C code.

Find the MINIMAL code block that implements this specific Micro Skill
in the teacher's solution. Return ONLY the lines directly relevant
to this skill — nothing more.

Micro Skill: {skill["text"]}

Teacher Solution Code (with line numbers):
{numbered_teacher}

HARD CONSTRAINTS:
- Return ONLY the lines that directly implement this skill.
- Do NOT include unrelated code (declarations, printf, main signature, etc.)
  unless they are inseparable from the skill implementation.
- Do NOT return the entire program — only the relevant code block.
- The snippet must be complete enough for a student to replicate the fix.
- line_start and line_end must match the actual line numbers shown above.
- The snippet field must contain ONLY raw code — no line number prefixes,
  no "Lines X-X:" text, no labels of any kind.
- PRESERVE the original line breaks exactly as they appear in the source code.
  Each statement must be on its own line — do NOT collapse multiple statements
  onto one line. Use \\n to represent line breaks in the JSON string value.
- Return ONLY a valid JSON object. No explanation. No markdown fences.
- Format:
{{
  "snippet":    "line1\\nline2\\nline3",
  "line_start": <integer>,
  "line_end":   <integer>
}}
"""
            content = call_llm_with_retry(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
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
                )

                references.append({
                    "rank":       skill["rank"],
                    "snippet":    snippet,
                    "line_start": ref_data.get("line_start", 0),
                    "line_end":   ref_data.get("line_end",   0)
                })
                engine_logger.info(
                    f"AGENT 1: Teacher reference generated for "
                    f"skill rank {skill['rank']}. ✅"
                )
            except Exception as e:
                engine_logger.error(
                    f"AGENT 1: Teacher reference parsing failed "
                    f"for rank {skill['rank']}: {e}"
                )
                return []

        return references


    # ── LLM HELPER — LIST RESPONSE ─────────────────────────────────
    def _call_llm_for_list(self, prompt: str, step_label: str) -> list[str] | None:
        """
        Calls LLM via retry wrapper and parses a JSON list response.
        Returns None on error, [] on empty result.
        """
        content = call_llm_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
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

            engine_logger.info(
                f"AGENT 1: {step_label} — {len(result)} items returned."
            )
            return result

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
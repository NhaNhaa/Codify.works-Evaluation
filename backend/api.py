"""
backend/api.py
FastAPI application — 11 endpoints + Swagger UI at /docs.
REST interface for teacher uploads and friend's AWS Node.js frontend.
All agents use lazy initialization — no crash on import.

Endpoint Organization:
  1-2: Upload files (setup)
  3:   Extract skills (Phase 1)
  4:   Evaluate student (Phase 2+3)
  5-6: View results and skills (read-only)
  7:   Health check (system)
  8-11: Delete operations (grouped — all destructive ops together)
"""

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agents.agent1_extractor import get_agent1
from backend.agents.agent2_evaluator import get_agent2
from backend.agents.agent3_feedback import get_agent3
from backend.config.config import DATA_INPUTS_PATH, DATA_OUTPUTS_PATH
from backend.rag.rag_pipeline import rag_pipeline
from backend.utils.logger import engine_logger


# ══════════════════════════════════════════════════════════════════
# SWAGGER TAG ORGANIZATION
# ══════════════════════════════════════════════════════════════════
TAGS_METADATA = [
    {
        "name": "1 — Upload",
        "description": "Upload assignment files and student submissions.",
    },
    {
        "name": "2 — Extract Skills",
        "description": "Run Agent 1 — extract micro skills from assignment files.",
    },
    {
        "name": "3 — Evaluate",
        "description": "Run Agent 2 + 3 — evaluate student code and generate feedback.",
    },
    {
        "name": "4 — View",
        "description": "Retrieve stored results and micro skills (read-only).",
    },
    {
        "name": "5 — Delete",
        "description": (
            "⚠️ **Destructive operations** — permanently removes files and/or ChromaDB data.\n\n"
            "| Endpoint | What It Deletes |\n"
            "|---|---|\n"
            "| `DELETE /results/{id}` | Output files only (JSON + MD) |\n"
            "| `DELETE /student/{id}` | Student input `.c` + all outputs |\n"
            "| `DELETE /skills/{id}` | ChromaDB skills + teacher refs |\n"
            "| `DELETE /assignment/{id}` | Input files + ChromaDB (everything) |"
        ),
    },
    {
        "name": "System",
        "description": "Health check and system status.",
    },
]

API_VERSION = "1.0.0"
ASSIGNMENT_FILE_NAMES = {
    "instructions": "instructions.md",
    "starter_code": "starter_code.c",
    "teacher_code": "teacher_correction_code.c",
}
RESULT_JSON_SUFFIX = "_feedback.json"
RESULT_MARKDOWN_SUFFIX = "_feedback.md"
STUDENT_FILE_EXTENSION = ".c"


# ══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="AutoEval-C — AI Evaluation Module",
    description=(
        "## AutoEval-C · Codify.works\n\n"
        "RAG-powered C code evaluation system with "
        "3 constrained AI agents. No scoring — qualitative feedback only.\n\n"
        "---\n\n"
        "### Workflow\n"
        "1. `POST /upload` → Upload assignment files\n"
        "2. `POST /upload-student` → Upload student submission\n"
        "3. `POST /extract-skills` → Agent 1 extracts micro skills\n"
        "4. `POST /evaluate` → Agent 2 evaluates · Agent 3 writes feedback\n"
        "5. `GET /results/{student_id}` → Get JSON + Markdown report\n\n"
        "---\n\n"
        "### Key Rules\n"
        "- Phase 1 must complete before Phase 2+3\n"
        "- Outputs: `JSON` (truth) + `Markdown` (display)\n"
        "- Recommended fix always from teacher reference in ChromaDB\n"
        "- All delete operations are grouped under **5 — Delete**"
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=TAGS_METADATA,
)


# ══════════════════════════════════════════════════════════════════
# CORS — Allow friend's Node.js AWS frontend
# ══════════════════════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST MODELS
# ══════════════════════════════════════════════════════════════════
class ExtractSkillsRequest(BaseModel):
    assignment_id: str = Field(
        ...,
        description="Unique assignment identifier",
        json_schema_extra={"example": "lab_01"},
    )
    force_regenerate: bool = Field(
        False,
        description="Set true to overwrite existing skills in ChromaDB",
        json_schema_extra={"example": False},
    )


class EvaluateRequest(BaseModel):
    assignment_id: str = Field(
        ...,
        description="Must match the assignment used in Phase 1",
        json_schema_extra={"example": "lab_01"},
    )
    student_id: str = Field(
        ...,
        description="Unique student identifier",
        json_schema_extra={"example": "student_01"},
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    message: str


# ══════════════════════════════════════════════════════════════════
#  TAG 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════
@app.post(
    "/upload",
    summary="Upload 3 assignment files",
    description=(
        "Upload `instructions.md`, `starter_code.c`, and "
        "`teacher_correction_code.c` for a new assignment.\n\n"
        "Files are saved to `data/inputs/{assignment_id}/`.\n\n"
        "**Next step:** `POST /extract-skills`"
    ),
    tags=["1 — Upload"],
    responses={
        200: {"description": "Files uploaded successfully"},
        400: {"description": "Invalid file type"},
        500: {"description": "Upload failed"},
    },
)
async def upload_files(
    assignment_id: str = Form(..., description="Unique assignment ID e.g. lab_01"),
    instructions: UploadFile = File(..., description="Assignment instructions (.md)"),
    starter_code: UploadFile = File(..., description="Starter code template (.c)"),
    teacher_code: UploadFile = File(..., description="Teacher correction code (.c)"),
):
    try:
        _validate_identifier(assignment_id, "assignment_id")
        _validate_extension(instructions.filename, ".md", "instructions")
        _validate_extension(starter_code.filename, ".c", "starter_code")
        _validate_extension(teacher_code.filename, ".c", "teacher_code")

        assignment_dir = _get_assignment_input_dir(assignment_id)
        assignment_dir.mkdir(parents=True, exist_ok=True)

        instructions_path = assignment_dir / ASSIGNMENT_FILE_NAMES["instructions"]
        starter_path = assignment_dir / ASSIGNMENT_FILE_NAMES["starter_code"]
        teacher_path = assignment_dir / ASSIGNMENT_FILE_NAMES["teacher_code"]

        with instructions_path.open("wb") as file_handle:
            file_handle.write(await instructions.read())
        with starter_path.open("wb") as file_handle:
            file_handle.write(await starter_code.read())
        with teacher_path.open("wb") as file_handle:
            file_handle.write(await teacher_code.read())

        engine_logger.info(
            f"API: Files uploaded for assignment '{assignment_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "assignment_id": assignment_id,
                "files_saved": {
                    "instructions": str(instructions_path),
                    "starter_code": str(starter_path),
                    "teacher_code": str(teacher_path),
                },
                "next_step": "Call POST /extract-skills to run Agent 1.",
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: File upload failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/upload-student",
    summary="Upload one student submission",
    description=(
        "Upload a single student's `.c` file for evaluation.\n"
        "Saved as `data/inputs/{assignment_id}/students/{student_id}.c`.\n\n"
        "**Next step:** `POST /evaluate`"
    ),
    tags=["1 — Upload"],
    responses={
        200: {"description": "Student file uploaded"},
        400: {"description": "Invalid file type — must be .c"},
        500: {"description": "Upload failed"},
    },
)
async def upload_student(
    assignment_id: str = Form(..., description="Assignment this student belongs to"),
    student_id: str = Form(..., description="Unique student ID e.g. student_01"),
    student_code: UploadFile = File(..., description="Student C code submission (.c)"),
):
    try:
        _validate_identifier(assignment_id, "assignment_id")
        _validate_identifier(student_id, "student_id")
        _validate_extension(student_code.filename, ".c", "student_code")

        students_dir = _get_assignment_students_dir(assignment_id)
        students_dir.mkdir(parents=True, exist_ok=True)

        student_path = students_dir / f"{student_id}{STUDENT_FILE_EXTENSION}"
        with student_path.open("wb") as file_handle:
            file_handle.write(await student_code.read())

        engine_logger.info(
            f"API: Student file uploaded — "
            f"student '{student_id}' assignment '{assignment_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "student_id": student_id,
                "assignment_id": assignment_id,
                "file_saved": str(student_path),
                "next_step": f"Call POST /evaluate with student_id='{student_id}'.",
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: Student upload failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════
#  TAG 2 — EXTRACT SKILLS
# ══════════════════════════════════════════════════════════════════
@app.post(
    "/extract-skills",
    summary="Run Agent 1 — extract micro skills (Phase 1)",
    description=(
        "Runs **once per assignment**. Skipped automatically if skills "
        "already exist in ChromaDB.\n\n"
        "**Agent 1 pipeline:** Read instructions → extract/generate skills → "
        "rank (Python) → assign weights (sum=10) → generate teacher refs → "
        "store in ChromaDB.\n\n"
        "Use `force_regenerate: true` to overwrite existing skills.\n\n"
        "**Next step:** `POST /evaluate`"
    ),
    tags=["2 — Extract Skills"],
    responses={
        200: {"description": "Skills extracted and stored"},
        404: {"description": "Assignment files not found"},
        500: {"description": "Extraction failed"},
    },
)
async def extract_skills(request: ExtractSkillsRequest):
    try:
        _validate_identifier(request.assignment_id, "assignment_id")

        assignment_dir = _get_assignment_input_dir(request.assignment_id)
        instructions_path = assignment_dir / ASSIGNMENT_FILE_NAMES["instructions"]
        starter_path = assignment_dir / ASSIGNMENT_FILE_NAMES["starter_code"]
        teacher_path = assignment_dir / ASSIGNMENT_FILE_NAMES["teacher_code"]

        missing_files = [
            str(file_path)
            for file_path in [instructions_path, starter_path, teacher_path]
            if not file_path.exists()
        ]
        if missing_files:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Assignment files not found. Upload them first via POST /upload. "
                    f"Missing: {missing_files}"
                ),
            )

        success = get_agent1().run(
            assignment_id=request.assignment_id,
            instructions_path=str(instructions_path),
            starter_path=str(starter_path),
            teacher_path=str(teacher_path),
            force_regenerate=request.force_regenerate,
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Skill extraction failed for '{request.assignment_id}'.",
            )

        skills = rag_pipeline.retrieve_micro_skills(request.assignment_id)

        engine_logger.info(
            f"API: Skills extracted for assignment '{request.assignment_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "assignment_id": request.assignment_id,
                "skills_count": len(skills),
                "skills": skills,
                "next_step": "Call POST /evaluate for each student.",
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: extract-skills failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════
#  TAG 3 — EVALUATE
# ══════════════════════════════════════════════════════════════════
@app.post(
    "/evaluate",
    summary="Run Agent 2 + 3 — evaluate student (Phase 2+3)",
    description=(
        "Runs **per student**.\n\n"
        "**Phase 2 (Agent 2):** Evaluate student code per skill → "
        "verify against teacher reference → force fix from ChromaDB.\n\n"
        "**Phase 3 (Agent 3):** Write pedagogical feedback → "
        "self-check quality → enforce sentence limits.\n\n"
        "**Prerequisites:** Phase 1 completed + student file uploaded.\n\n"
        "**Outputs:** `data/outputs/{assignment_id}/{student_id}_feedback.json` "
        "and `.md`"
    ),
    tags=["3 — Evaluate"],
    responses={
        200: {"description": "Evaluation complete — JSON + Markdown returned"},
        404: {"description": "Student file not found"},
        500: {"description": "Evaluation failed"},
    },
)
async def evaluate_student(request: EvaluateRequest):
    try:
        _validate_identifier(request.assignment_id, "assignment_id")
        _validate_identifier(request.student_id, "student_id")

        student_path = (
            _get_assignment_students_dir(request.assignment_id)
            / f"{request.student_id}{STUDENT_FILE_EXTENSION}"
        )

        if not student_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Student file not found: '{student_path}'. "
                    "Upload it first via POST /upload-student."
                ),
            )

        evaluation_result = get_agent2().run(
            assignment_id=request.assignment_id,
            student_id=request.student_id,
            student_path=str(student_path),
        )

        if not evaluation_result:
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed for student '{request.student_id}'.",
            )

        output = get_agent3().run(evaluation_result)

        if not output:
            raise HTTPException(
                status_code=500,
                detail=f"Feedback generation failed for '{request.student_id}'.",
            )

        outputs_dir = _get_assignment_output_dir(request.assignment_id)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        json_path = outputs_dir / f"{request.student_id}{RESULT_JSON_SUFFIX}"
        md_path = outputs_dir / f"{request.student_id}{RESULT_MARKDOWN_SUFFIX}"

        with json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(output["json"], file_handle, indent=2, ensure_ascii=False)

        with md_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(output["markdown"])

        engine_logger.info(
            f"API: Evaluation complete for student '{request.student_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "student_id": request.student_id,
                "assignment_id": request.assignment_id,
                "json_report": output["json"],
                "markdown_report": output["markdown"],
                "files_saved": {
                    "json": str(json_path),
                    "markdown": str(md_path),
                },
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: evaluate failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════
#  TAG 4 — VIEW (read-only)
# ══════════════════════════════════════════════════════════════════
@app.get(
    "/results/{student_id}",
    summary="Get student feedback report",
    description=(
        "Returns saved JSON + Markdown feedback for a student.\n\n"
        "Student must have been evaluated first via `POST /evaluate`.\n"
        "If the same student ID exists in multiple assignments, pass `assignment_id`."
    ),
    tags=["4 — View"],
    responses={
        200: {"description": "Report returned"},
        400: {"description": "Multiple reports found — specify assignment_id"},
        404: {"description": "No results found — run POST /evaluate first"},
    },
)
async def get_results(
    student_id: str,
    assignment_id: str | None = Query(
        default=None,
        description="Optional assignment_id to disambiguate results",
    ),
):
    try:
        _validate_identifier(student_id, "student_id")

        json_path, md_path, resolved_assignment_id = _resolve_result_paths(
            student_id=student_id,
            assignment_id=assignment_id,
        )

        if json_path is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No results found for student '{student_id}'. "
                    "Run POST /evaluate first."
                ),
            )

        with json_path.open("r", encoding="utf-8") as file_handle:
            json_report = json.load(file_handle)

        markdown_report = ""
        if md_path is not None and md_path.exists():
            with md_path.open("r", encoding="utf-8") as file_handle:
                markdown_report = file_handle.read()

        engine_logger.info(
            f"API: Results retrieved for student '{student_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "student_id": student_id,
                "assignment_id": resolved_assignment_id,
                "json_report": json_report,
                "markdown_report": markdown_report,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: get_results failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/skills/{assignment_id}",
    summary="View stored micro skills",
    description=(
        "Returns all micro skills in ChromaDB for an assignment, "
        "ordered by rank.\n\n"
        "Use to verify Phase 1 completed correctly before evaluating."
    ),
    tags=["4 — View"],
    responses={
        200: {"description": "Skills returned"},
        404: {"description": "No skills found — run POST /extract-skills first"},
    },
)
async def get_skills(assignment_id: str):
    try:
        _validate_identifier(assignment_id, "assignment_id")
        skills = rag_pipeline.retrieve_micro_skills(assignment_id)

        if not skills:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No skills found for assignment '{assignment_id}'. "
                    "Run POST /extract-skills first."
                ),
            )

        engine_logger.info(
            f"API: Skills retrieved for assignment '{assignment_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "assignment_id": assignment_id,
                "skills_count": len(skills),
                "skills": skills,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: get_skills failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════
#  SYSTEM
# ══════════════════════════════════════════════════════════════════
@app.get(
    "/health",
    summary="Health check",
    description="Returns system status. Use to verify the API is running.",
    tags=["System"],
    response_model=HealthResponse,
)
async def health_check():
    engine_logger.info("API: Health check called.")
    return JSONResponse(
        content={
            "status": "healthy",
            "version": API_VERSION,
            "message": "AutoEval-C API is running.",
        }
    )


# ══════════════════════════════════════════════════════════════════
#  TAG 5 — DELETE (all destructive operations grouped here)
# ══════════════════════════════════════════════════════════════════
@app.delete(
    "/results/{student_id}",
    summary="🗑️ Delete student OUTPUT files only",
    description=(
        "Removes evaluation output files only.\n\n"
        "If `assignment_id` is omitted, all matching output files for that "
        "student across assignments are deleted."
    ),
    tags=["5 — Delete"],
    responses={
        200: {"description": "Output files deleted"},
        404: {"description": "No results found for this student"},
        500: {"description": "Delete failed"},
    },
)
async def delete_results(
    student_id: str,
    assignment_id: str | None = Query(
        default=None,
        description="Optional assignment_id to target one assignment only",
    ),
):
    try:
        _validate_identifier(student_id, "student_id")

        deleted_files = _delete_result_files(
            student_id=student_id,
            assignment_id=assignment_id,
        )

        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"No results found for student '{student_id}'.",
            )

        engine_logger.info(
            f"API: Results deleted for student '{student_id}'. "
            f"Files removed: {deleted_files}"
        )

        return JSONResponse(
            content={
                "status": "success",
                "student_id": student_id,
                "assignment_id": assignment_id,
                "deleted_files": deleted_files,
                "message": (
                    f"Output files for '{student_id}' deleted. "
                    "Re-run POST /evaluate to regenerate."
                ),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: delete_results failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/student/{student_id}",
    summary="🗑️ Delete student INPUT + all OUTPUT files",
    description=(
        "Removes student input and output files.\n\n"
        "If `assignment_id` is omitted, all matching student files and outputs "
        "across assignments are deleted."
    ),
    tags=["5 — Delete"],
    responses={
        200: {"description": "Student data deleted"},
        404: {"description": "No files found for this student"},
        500: {"description": "Delete failed"},
    },
)
async def delete_student(
    student_id: str,
    assignment_id: str | None = Query(
        default=None,
        description="Optional assignment_id to target one assignment only",
    ),
):
    try:
        _validate_identifier(student_id, "student_id")

        deleted_files = []

        if assignment_id is not None:
            _validate_identifier(assignment_id, "assignment_id")
            student_input = (
                _get_assignment_students_dir(assignment_id)
                / f"{student_id}{STUDENT_FILE_EXTENSION}"
            )
            if student_input.exists():
                student_input.unlink()
                deleted_files.append(str(student_input))

            deleted_files.extend(
                _delete_result_files(student_id=student_id, assignment_id=assignment_id)
            )
        else:
            for student_input in Path(DATA_INPUTS_PATH).glob(
                f"*/students/{student_id}{STUDENT_FILE_EXTENSION}"
            ):
                if student_input.exists():
                    student_input.unlink()
                    deleted_files.append(str(student_input))

            deleted_files.extend(
                _delete_result_files(student_id=student_id, assignment_id=None)
            )

        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"No files found for student '{student_id}'.",
            )

        engine_logger.info(
            f"API: Student '{student_id}' deleted. "
            f"Files removed: {deleted_files}"
        )

        return JSONResponse(
            content={
                "status": "success",
                "student_id": student_id,
                "assignment_id": assignment_id,
                "deleted_files": deleted_files,
                "message": (
                    f"All data for '{student_id}' deleted. "
                    "Re-upload via POST /upload-student."
                ),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: delete_student failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/skills/{assignment_id}",
    summary="🗑️ Delete skills + teacher refs from ChromaDB",
    description=(
        "Clears micro skills and teacher references for an assignment "
        "from ChromaDB only.\n\n"
        "**Keeps:** Input files are NOT deleted.\n\n"
        "**After this:** Re-run `POST /extract-skills` with "
        "`force_regenerate: true`."
    ),
    tags=["5 — Delete"],
    responses={
        200: {"description": "ChromaDB data cleared"},
        500: {"description": "Clear failed"},
    },
)
async def delete_skills(assignment_id: str):
    try:
        _validate_identifier(assignment_id, "assignment_id")
        success = rag_pipeline.clear_assignment(assignment_id)

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear skills for '{assignment_id}'.",
            )

        engine_logger.info(
            f"API: Skills cleared for assignment '{assignment_id}'."
        )

        return JSONResponse(
            content={
                "status": "success",
                "assignment_id": assignment_id,
                "message": (
                    f"ChromaDB cleared for '{assignment_id}'. "
                    "Run POST /extract-skills to regenerate."
                ),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: delete_skills failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/assignment/{assignment_id}",
    summary="🗑️ Delete EVERYTHING for an assignment",
    description=(
        "Removes all assignment input files and ChromaDB data for one assignment.\n\n"
        "**Does NOT delete:** Student files or their outputs from other assignments."
    ),
    tags=["5 — Delete"],
    responses={
        200: {"description": "Assignment data deleted"},
        500: {"description": "Delete failed"},
    },
)
async def delete_assignment(assignment_id: str):
    try:
        _validate_identifier(assignment_id, "assignment_id")

        assignment_dir = _get_assignment_input_dir(assignment_id)
        deleted_files = []

        for filename in ASSIGNMENT_FILE_NAMES.values():
            file_path = assignment_dir / filename
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))

        if assignment_dir.exists() and not any(assignment_dir.iterdir()):
            assignment_dir.rmdir()

        chroma_cleared = rag_pipeline.clear_assignment(assignment_id)
        if not chroma_cleared:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear ChromaDB for '{assignment_id}'.",
            )

        engine_logger.info(
            f"API: Assignment '{assignment_id}' deleted. "
            f"Files removed: {deleted_files}"
        )

        return JSONResponse(
            content={
                "status": "success",
                "assignment_id": assignment_id,
                "deleted_files": deleted_files,
                "chromadb": "cleared",
                "message": (
                    f"All data for '{assignment_id}' deleted. "
                    "Upload new files via POST /upload."
                ),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        engine_logger.error(f"API: delete_assignment failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════
def _validate_extension(filename: str, expected_ext: str, field_name: str) -> None:
    """Validates uploaded file has the correct extension."""
    if not filename:
        raise HTTPException(
            status_code=400,
            detail=f"'{field_name}' file has no filename.",
        )

    suffix = Path(filename).suffix.lower()
    if suffix != expected_ext:
        raise HTTPException(
            status_code=400,
            detail=f"'{field_name}' must be a {expected_ext} file. Got: '{suffix}'.",
        )


def _validate_identifier(value: str, field_name: str) -> None:
    """Validates assignment_id and student_id style fields."""
    if not value or not str(value).strip():
        raise HTTPException(
            status_code=400,
            detail=f"'{field_name}' is missing or blank.",
        )


def _get_assignment_input_dir(assignment_id: str) -> Path:
    return Path(DATA_INPUTS_PATH) / assignment_id


def _get_assignment_students_dir(assignment_id: str) -> Path:
    return _get_assignment_input_dir(assignment_id) / "students"


def _get_assignment_output_dir(assignment_id: str) -> Path:
    return Path(DATA_OUTPUTS_PATH) / assignment_id


def _resolve_result_paths(
    student_id: str,
    assignment_id: str | None,
) -> tuple[Path | None, Path | None, str | None]:
    """
    Resolves result paths for a student.
    If assignment_id is omitted, returns the unique match or raises 400 if ambiguous.
    """
    if assignment_id is not None:
        _validate_identifier(assignment_id, "assignment_id")
        outputs_dir = _get_assignment_output_dir(assignment_id)
        json_path = outputs_dir / f"{student_id}{RESULT_JSON_SUFFIX}"
        md_path = outputs_dir / f"{student_id}{RESULT_MARKDOWN_SUFFIX}"
        if not json_path.exists():
            return None, None, assignment_id
        return json_path, md_path, assignment_id

    matches = list(Path(DATA_OUTPUTS_PATH).glob(f"*/{student_id}{RESULT_JSON_SUFFIX}"))

    if not matches:
        return None, None, None

    if len(matches) > 1:
        matched_assignments = [path.parent.name for path in matches]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple reports found for student '{student_id}'. "
                f"Specify assignment_id. Matches: {matched_assignments}"
            ),
        )

    json_path = matches[0]
    md_path = json_path.with_name(f"{student_id}{RESULT_MARKDOWN_SUFFIX}")
    return json_path, md_path, json_path.parent.name


def _delete_result_files(student_id: str, assignment_id: str | None) -> list[str]:
    """
    Deletes result files for one student.
    If assignment_id is omitted, deletes all matching result files across assignments.
    """
    deleted_files = []

    if assignment_id is not None:
        _validate_identifier(assignment_id, "assignment_id")
        targets = [
            _get_assignment_output_dir(assignment_id) / f"{student_id}{RESULT_JSON_SUFFIX}",
            _get_assignment_output_dir(assignment_id) / f"{student_id}{RESULT_MARKDOWN_SUFFIX}",
        ]
    else:
        targets = list(Path(DATA_OUTPUTS_PATH).glob(f"*/{student_id}{RESULT_JSON_SUFFIX}"))
        targets.extend(Path(DATA_OUTPUTS_PATH).glob(f"*/{student_id}{RESULT_MARKDOWN_SUFFIX}"))

    for file_path in targets:
        if file_path.exists():
            file_path.unlink()
            deleted_files.append(str(file_path))

    return deleted_files
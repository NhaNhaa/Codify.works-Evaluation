"""
backend/tests/test_12_api.py
Tests for api.py — all endpoints.
Uses FastAPI TestClient — no real server needed.
Agents are fully mocked — no API keys required.
Run with: python -m pytest backend/tests/test_12_api.py -v
"""

import json
import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app


# ── TEST CLIENT ────────────────────────────────────────────────────
client = TestClient(app)


# ── SHARED MOCK DATA ───────────────────────────────────────────────
MOCK_SKILLS = [
    {"text": "Use scanf to read integers", "rank": 1, "weight": 4},
    {"text": "Shift elements using arr[i]", "rank": 2, "weight": 3},
    {"text": "Avoid losing data using temp", "rank": 3, "weight": 2},
    {"text": "Access array using arr[i]",   "rank": 4, "weight": 1},
]

MOCK_EVALUATION = {
    "student_id":    "student_01",
    "assignment_id": "lab_01",
    "skills": [
        {
            "skill":           "Use scanf to read integers",
            "rank":            1,
            "weight":          4,
            "status":          "PASS",
            "line_start":      5,
            "line_end":        5,
            "student_snippet": "scanf('%d', &arr[i]);",
            "recommended_fix": None,
            "feedback":        "Well done!",
            "verified":        True
        }
    ]
}

MOCK_OUTPUT = {
    "json":     MOCK_EVALUATION,
    "markdown": "# Evaluation Report — student_01\n---\n### ✅ Skill 1"
}


# ══════════════════════════════════════════════════════════════════
# 1. GET /health
# ══════════════════════════════════════════════════════════════════

def test_health_check():
    """Should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"]  == "healthy"
    assert data["version"] == "1.0.0"


# ══════════════════════════════════════════════════════════════════
# 2. POST /upload
# ══════════════════════════════════════════════════════════════════

def test_upload_files_success(tmp_path):
    """Should save all 3 required files and return success."""
    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)):
        response = client.post(
            "/upload",
            data={"assignment_id": "lab_01"},
            files={
                "instructions": ("instructions.md", BytesIO(b"# Assignment"), "text/markdown"),
                "starter_code": ("starter_code.c",  BytesIO(b"int main(){}"),  "text/plain"),
                "teacher_code": ("teacher_code.c",  BytesIO(b"int main(){}"),  "text/plain"),
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"]        == "success"
    assert data["assignment_id"] == "lab_01"
    assert "next_step" in data


def test_upload_files_wrong_extension(tmp_path):
    """Should return 400 when wrong file extension uploaded."""
    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)):
        response = client.post(
            "/upload",
            data={"assignment_id": "lab_01"},
            files={
                "instructions": ("instructions.txt", BytesIO(b"content"), "text/plain"),
                "starter_code": ("starter_code.c",   BytesIO(b"int main(){}"), "text/plain"),
                "teacher_code": ("teacher_code.c",   BytesIO(b"int main(){}"), "text/plain"),
            }
        )
    assert response.status_code == 400


# ══════════════════════════════════════════════════════════════════
# 3. POST /upload-student
# ══════════════════════════════════════════════════════════════════

def test_upload_student_success(tmp_path):
    """Should save student file and return success."""
    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)):
        response = client.post(
            "/upload-student",
            data={
                "assignment_id": "lab_01",
                "student_id":    "student_01"
            },
            files={
                "student_code": ("student_01.c", BytesIO(b"int main(){}"), "text/plain")
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"]     == "success"
    assert data["student_id"] == "student_01"
    assert "next_step" in data


def test_upload_student_wrong_extension(tmp_path):
    """Should return 400 for non-.c student file."""
    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)):
        response = client.post(
            "/upload-student",
            data={
                "assignment_id": "lab_01",
                "student_id":    "student_01"
            },
            files={
                "student_code": ("student_01.txt", BytesIO(b"content"), "text/plain")
            }
        )
    assert response.status_code == 400


# ══════════════════════════════════════════════════════════════════
# 4. POST /extract-skills
# ══════════════════════════════════════════════════════════════════

def test_extract_skills_success(tmp_path):
    """Should call Agent 1 and return skills on success."""
    # Create dummy input files
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "instructions.md").write_text("# Assignment")
    (tmp_path / "starter_code.c").write_text("int main(){}")
    (tmp_path / "teacher_correction_code.c").write_text("int main(){}")

    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)), \
         patch("backend.api.get_agent1") as mock_get_agent1, \
         patch("backend.api.rag_pipeline") as mock_rag:

        mock_agent1 = MagicMock()
        mock_agent1.run.return_value = True
        mock_get_agent1.return_value = mock_agent1
        mock_rag.retrieve_micro_skills.return_value = MOCK_SKILLS

        response = client.post(
            "/extract-skills",
            json={"assignment_id": "lab_01", "force_regenerate": False}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"]       == "success"
    assert data["skills_count"] == 4
    assert "next_step" in data


def test_extract_skills_agent_failure(tmp_path):
    """Should return 500 when Agent 1 fails."""
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "instructions.md").write_text("# Assignment")
    (tmp_path / "starter_code.c").write_text("int main(){}")
    (tmp_path / "teacher_correction_code.c").write_text("int main(){}")

    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)), \
         patch("backend.api.get_agent1") as mock_get_agent1:

        mock_agent1 = MagicMock()
        mock_agent1.run.return_value = False
        mock_get_agent1.return_value = mock_agent1

        response = client.post(
            "/extract-skills",
            json={"assignment_id": "lab_01", "force_regenerate": False}
        )

    assert response.status_code == 500


# ══════════════════════════════════════════════════════════════════
# 5. POST /evaluate
# ══════════════════════════════════════════════════════════════════

def test_evaluate_success(tmp_path):
    """Should run Agent 2 + 3 and return JSON + Markdown."""
    students_dir = tmp_path / "students"
    students_dir.mkdir(parents=True)
    (students_dir / "student_01.c").write_text("int main(){}")

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)

    with patch("backend.api.DATA_INPUTS_PATH",  str(tmp_path)), \
         patch("backend.api.DATA_OUTPUTS_PATH", str(outputs_dir)), \
         patch("backend.api.get_agent2") as mock_get_agent2, \
         patch("backend.api.get_agent3") as mock_get_agent3:

        mock_agent2 = MagicMock()
        mock_agent2.run.return_value = MOCK_EVALUATION
        mock_get_agent2.return_value = mock_agent2

        mock_agent3 = MagicMock()
        mock_agent3.run.return_value = MOCK_OUTPUT
        mock_get_agent3.return_value = mock_agent3

        response = client.post(
            "/evaluate",
            json={"assignment_id": "lab_01", "student_id": "student_01"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"]          == "success"
    assert "json_report"           in data
    assert "markdown_report"       in data
    assert "files_saved"           in data


def test_evaluate_student_file_not_found(tmp_path):
    """Should return 404 when student file does not exist."""
    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)):
        response = client.post(
            "/evaluate",
            json={"assignment_id": "lab_01", "student_id": "ghost_student"}
        )
    assert response.status_code == 404


def test_evaluate_agent2_failure(tmp_path):
    """Should return 500 when Agent 2 fails."""
    students_dir = tmp_path / "students"
    students_dir.mkdir(parents=True)
    (students_dir / "student_01.c").write_text("int main(){}")

    with patch("backend.api.DATA_INPUTS_PATH", str(tmp_path)), \
         patch("backend.api.get_agent2") as mock_get_agent2:

        mock_agent2 = MagicMock()
        mock_agent2.run.return_value = None
        mock_get_agent2.return_value = mock_agent2

        response = client.post(
            "/evaluate",
            json={"assignment_id": "lab_01", "student_id": "student_01"}
        )

    assert response.status_code == 500


# ══════════════════════════════════════════════════════════════════
# 6. GET /results/{student_id}
# ══════════════════════════════════════════════════════════════════

def test_get_results_success(tmp_path):
    """Should return JSON and Markdown report for existing student."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    json_path = outputs_dir / "student_01_feedback.json"
    md_path   = outputs_dir / "student_01_feedback.md"
    json_path.write_text(json.dumps(MOCK_EVALUATION))
    md_path.write_text("# Evaluation Report")

    with patch("backend.api.DATA_OUTPUTS_PATH", str(outputs_dir)):
        response = client.get("/results/student_01")

    assert response.status_code == 200
    data = response.json()
    assert data["status"]     == "success"
    assert data["student_id"] == "student_01"
    assert "json_report"      in data
    assert "markdown_report"  in data


def test_get_results_not_found(tmp_path):
    """Should return 404 for student with no saved results."""
    with patch("backend.api.DATA_OUTPUTS_PATH", str(tmp_path)):
        response = client.get("/results/nonexistent_student")
    assert response.status_code == 404


# ══════════════════════════════════════════════════════════════════
# 7. GET /skills/{assignment_id}
# ══════════════════════════════════════════════════════════════════

def test_get_skills_success():
    """Should return skills when ChromaDB has them."""
    with patch("backend.api.rag_pipeline") as mock_rag:
        mock_rag.retrieve_micro_skills.return_value = MOCK_SKILLS
        response = client.get("/skills/lab_01")

    assert response.status_code == 200
    data = response.json()
    assert data["status"]       == "success"
    assert data["skills_count"] == 4


def test_get_skills_not_found():
    """Should return 404 when no skills in ChromaDB."""
    with patch("backend.api.rag_pipeline") as mock_rag:
        mock_rag.retrieve_micro_skills.return_value = []
        response = client.get("/skills/unknown_assignment")

    assert response.status_code == 404


# ══════════════════════════════════════════════════════════════════
# 8. DELETE /skills/{assignment_id}
# ══════════════════════════════════════════════════════════════════

def test_delete_skills_success():
    """Should clear skills and return success."""
    with patch("backend.api.rag_pipeline") as mock_rag:
        mock_rag.clear_assignment.return_value = True
        response = client.delete("/skills/lab_01")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "message"      in data


def test_delete_skills_failure():
    """Should return 500 when clear operation fails."""
    with patch("backend.api.rag_pipeline") as mock_rag:
        mock_rag.clear_assignment.return_value = False
        response = client.delete("/skills/lab_01")

    assert response.status_code == 500


# ── MAIN RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
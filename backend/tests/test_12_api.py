import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend import api as api_module


class FakeAgent1:
    def run(self, assignment_id, instructions_path, starter_path, teacher_path, force_regenerate=False):
        return True


class FakeAgent2:
    def run(self, assignment_id, student_id, student_path):
        return {
            "student_id": student_id,
            "assignment_id": assignment_id,
            "skills": [
                {
                    "skill": "Use scanf to read 5 integers into an array",
                    "rank": 1,
                    "weight": 4,
                    "status": "PASS",
                    "line_start": 1,
                    "line_end": 2,
                    "student_snippet": 'scanf("%d", &arr[i]);',
                    "recommended_fix": "",
                    "feedback": "Correct input pattern.",
                    "verified": True,
                }
            ],
        }


class FakeAgent3:
    def run(self, evaluation_result):
        return {
            "json": evaluation_result,
            "markdown": "# Mock Report",
        }


class FakeRAGPipeline:
    def __init__(self):
        self.skills = [
            {"text": "Use scanf to read 5 integers into an array", "rank": 1, "weight": 4},
            {"text": "Avoid losing data during shifting, using a temporary variable", "rank": 2, "weight": 3},
            {"text": "Be able to shift elements, using arr[i-1] and handle the 1st cell", "rank": 3, "weight": 2},
            {"text": "Be able to access array elements using index arr[i]", "rank": 4, "weight": 1},
        ]

    def retrieve_micro_skills(self, assignment_id):
        return self.skills

    def clear_assignment(self, assignment_id):
        return True


def build_client(monkeypatch, tmp_path):
    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"

    monkeypatch.setattr(api_module, "DATA_INPUTS_PATH", str(inputs_dir))
    monkeypatch.setattr(api_module, "DATA_OUTPUTS_PATH", str(outputs_dir))
    monkeypatch.setattr(api_module, "get_agent1", lambda: FakeAgent1())
    monkeypatch.setattr(api_module, "get_agent2", lambda: FakeAgent2())
    monkeypatch.setattr(api_module, "get_agent3", lambda: FakeAgent3())
    monkeypatch.setattr(api_module, "rag_pipeline", FakeRAGPipeline())

    return TestClient(api_module.app), inputs_dir, outputs_dir


def test_upload_files_saves_into_assignment_scoped_directory(monkeypatch, tmp_path):
    client, inputs_dir, _ = build_client(monkeypatch, tmp_path)

    response = client.post(
        "/upload",
        data={"assignment_id": "lab_01"},
        files={
            "instructions": ("instructions.md", b"# Instructions"),
            "starter_code": ("starter_code.c", b"int main(void){return 0;}"),
            "teacher_code": ("teacher_code.c", b"int main(void){return 0;}"),
        },
    )

    assert response.status_code == 200
    assert (inputs_dir / "lab_01" / "instructions.md").exists()
    assert (inputs_dir / "lab_01" / "starter_code.c").exists()
    assert (inputs_dir / "lab_01" / "teacher_correction_code.c").exists()


def test_upload_student_saves_into_assignment_students_directory(monkeypatch, tmp_path):
    client, inputs_dir, _ = build_client(monkeypatch, tmp_path)

    response = client.post(
        "/upload-student",
        data={"assignment_id": "lab_01", "student_id": "student_01"},
        files={"student_code": ("student.c", b'int main(void){printf("ok");}')},
    )

    assert response.status_code == 200
    assert (inputs_dir / "lab_01" / "students" / "student_01.c").exists()


def test_extract_skills_uses_assignment_specific_input_files(monkeypatch, tmp_path):
    client, inputs_dir, _ = build_client(monkeypatch, tmp_path)

    assignment_dir = inputs_dir / "lab_01"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    (assignment_dir / "instructions.md").write_text("# Instructions", encoding="utf-8")
    (assignment_dir / "starter_code.c").write_text("int main(void){return 0;}", encoding="utf-8")
    (assignment_dir / "teacher_correction_code.c").write_text("int main(void){return 0;}", encoding="utf-8")

    response = client.post(
        "/extract-skills",
        json={"assignment_id": "lab_01", "force_regenerate": False},
    )

    assert response.status_code == 200
    assert response.json()["assignment_id"] == "lab_01"
    assert response.json()["skills_count"] == 4


def test_evaluate_student_saves_output_into_assignment_output_directory(monkeypatch, tmp_path):
    client, inputs_dir, outputs_dir = build_client(monkeypatch, tmp_path)

    student_dir = inputs_dir / "lab_01" / "students"
    student_dir.mkdir(parents=True, exist_ok=True)
    (student_dir / "student_01.c").write_text("int main(void){return 0;}", encoding="utf-8")

    response = client.post(
        "/evaluate",
        json={"assignment_id": "lab_01", "student_id": "student_01"},
    )

    assert response.status_code == 200
    assert (outputs_dir / "lab_01" / "student_01_feedback.json").exists()
    assert (outputs_dir / "lab_01" / "student_01_feedback.md").exists()


def test_get_results_requires_assignment_id_when_student_exists_in_multiple_assignments(monkeypatch, tmp_path):
    client, _, outputs_dir = build_client(monkeypatch, tmp_path)

    for assignment_id in ["lab_01", "lab_02"]:
        assignment_output_dir = outputs_dir / assignment_id
        assignment_output_dir.mkdir(parents=True, exist_ok=True)
        (assignment_output_dir / "student_01_feedback.json").write_text(
            json.dumps({"student_id": "student_01", "assignment_id": assignment_id}),
            encoding="utf-8",
        )

    response = client.get("/results/student_01")

    assert response.status_code == 400
    assert "Specify assignment_id" in response.json()["detail"]


def test_delete_student_without_assignment_id_removes_files_across_assignments(monkeypatch, tmp_path):
    client, inputs_dir, outputs_dir = build_client(monkeypatch, tmp_path)

    for assignment_id in ["lab_01", "lab_02"]:
        student_dir = inputs_dir / assignment_id / "students"
        student_dir.mkdir(parents=True, exist_ok=True)
        (student_dir / "student_01.c").write_text("int main(void){return 0;}", encoding="utf-8")

        output_dir = outputs_dir / assignment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "student_01_feedback.json").write_text("{}", encoding="utf-8")
        (output_dir / "student_01_feedback.md").write_text("# report", encoding="utf-8")

    response = client.delete("/student/student_01")

    assert response.status_code == 200
    assert not list(inputs_dir.glob("*/students/student_01.c"))
    assert not list(outputs_dir.glob("*/student_01_feedback.json"))
    assert not list(outputs_dir.glob("*/student_01_feedback.md"))
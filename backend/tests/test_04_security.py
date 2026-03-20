from backend.utils import security as security_module


def test_validate_file_input_accepts_valid_c_file(tmp_path):
    file_path = tmp_path / "student.c"
    file_path.write_text("int main(){return 0;}", encoding="utf-8")

    assert security_module.validate_file_input(str(file_path), ".c") is True


def test_validate_file_input_rejects_blank_path():
    assert security_module.validate_file_input("", ".c") is False


def test_validate_file_input_rejects_wrong_extension(tmp_path):
    file_path = tmp_path / "instructions.txt"
    file_path.write_text("hello", encoding="utf-8")

    assert security_module.validate_file_input(str(file_path), ".md") is False


def test_validate_file_input_accepts_expected_extension_without_dot(tmp_path):
    file_path = tmp_path / "starter.c"
    file_path.write_text("int main(){return 0;}", encoding="utf-8")

    assert security_module.validate_file_input(str(file_path), "c") is True


def test_validate_file_input_rejects_empty_file(tmp_path):
    file_path = tmp_path / "empty.c"
    file_path.write_text("", encoding="utf-8")

    assert security_module.validate_file_input(str(file_path), ".c") is False


def test_validate_all_input_files_accepts_complete_valid_set(tmp_path):
    instructions = tmp_path / "instructions.md"
    starter = tmp_path / "starter_code.c"
    teacher = tmp_path / "teacher_correction_code.c"
    student = tmp_path / "student_01.c"

    instructions.write_text("# Instructions", encoding="utf-8")
    starter.write_text("int main(){return 0;}", encoding="utf-8")
    teacher.write_text("int main(){return 0;}", encoding="utf-8")
    student.write_text("int main(){return 0;}", encoding="utf-8")

    assert security_module.validate_all_input_files(
        str(instructions),
        str(starter),
        str(teacher),
        str(student),
    ) is True


def test_validate_all_input_files_rejects_missing_student_when_provided(tmp_path):
    instructions = tmp_path / "instructions.md"
    starter = tmp_path / "starter_code.c"
    teacher = tmp_path / "teacher_correction_code.c"
    missing_student = tmp_path / "student_01.c"

    instructions.write_text("# Instructions", encoding="utf-8")
    starter.write_text("int main(){return 0;}", encoding="utf-8")
    teacher.write_text("int main(){return 0;}", encoding="utf-8")

    assert security_module.validate_all_input_files(
        str(instructions),
        str(starter),
        str(teacher),
        str(missing_student),
    ) is False


def test_is_content_safe_logs_warning_for_prohibited_keyword(monkeypatch):
    warning_messages = []

    def fake_warning(message):
        warning_messages.append(message)

    monkeypatch.setattr(security_module.engine_logger, "warning", fake_warning)

    result = security_module.is_content_safe("system('ls');", "student.c")

    assert result is True
    assert any("system(" in message for message in warning_messages)


def test_safe_read_file_returns_content_for_valid_file(tmp_path):
    file_path = tmp_path / "instructions.md"
    file_path.write_text("# Title", encoding="utf-8")

    content = security_module.safe_read_file(str(file_path), ".md")

    assert content == "# Title"


def test_safe_read_file_returns_none_for_whitespace_only_file(tmp_path):
    file_path = tmp_path / "instructions.md"
    file_path.write_text("   \n\t  ", encoding="utf-8")

    content = security_module.safe_read_file(str(file_path), ".md")

    assert content is None
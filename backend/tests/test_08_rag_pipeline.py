from backend.rag import rag_pipeline as rag_pipeline_module


class FakeEmbedder:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.last_texts = None

    def embed_texts(self, texts):
        self.last_texts = texts
        return self.embeddings


class FakeChromaClient:
    def __init__(self):
        self.assignment_exists_value = False
        self.store_micro_skills_called_with = None
        self.store_teacher_references_called_with = None
        self.retrieve_micro_skills_value = []
        self.retrieve_teacher_reference_value = None
        self.clear_assignment_value = True

    def assignment_exists(self, assignment_id):
        return self.assignment_exists_value

    def store_micro_skills(self, skills, assignment_id, embeddings, force_regenerate=False):
        self.store_micro_skills_called_with = {
            "skills": skills,
            "assignment_id": assignment_id,
            "embeddings": embeddings,
            "force_regenerate": force_regenerate,
        }
        return True

    def retrieve_micro_skills(self, assignment_id):
        return self.retrieve_micro_skills_value

    def store_teacher_references(self, references, assignment_id, embeddings, force_regenerate=False):
        self.store_teacher_references_called_with = {
            "references": references,
            "assignment_id": assignment_id,
            "embeddings": embeddings,
            "force_regenerate": force_regenerate,
        }
        return True

    def retrieve_teacher_reference(self, assignment_id, skill_rank):
        return self.retrieve_teacher_reference_value

    def clear_assignment(self, assignment_id):
        return self.clear_assignment_value


def test_store_micro_skills_rejects_weight_sum_mismatch(monkeypatch):
    pipeline = rag_pipeline_module.RAGPipeline()

    fake_embedder = FakeEmbedder([[0.1], [0.2]])
    fake_chroma = FakeChromaClient()

    monkeypatch.setattr(rag_pipeline_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(rag_pipeline_module, "get_chroma_client", lambda: fake_chroma)

    skills = [
        {"text": "Skill A", "rank": 1, "weight": 4},
        {"text": "Skill B", "rank": 2, "weight": 3},
    ]

    result = pipeline.store_micro_skills(skills, "assignment_1")

    assert result is False
    assert fake_chroma.store_micro_skills_called_with is None


def test_store_micro_skills_passes_embeddings_to_chroma(monkeypatch):
    pipeline = rag_pipeline_module.RAGPipeline()

    fake_embedder = FakeEmbedder([[0.1], [0.2], [0.3], [0.4]])
    fake_chroma = FakeChromaClient()

    monkeypatch.setattr(rag_pipeline_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(rag_pipeline_module, "get_chroma_client", lambda: fake_chroma)

    skills = [
        {"text": "Skill A", "rank": 1, "weight": 4},
        {"text": "Skill B", "rank": 2, "weight": 3},
        {"text": "Skill C", "rank": 3, "weight": 2},
        {"text": "Skill D", "rank": 4, "weight": 1},
    ]

    result = pipeline.store_micro_skills(skills, "assignment_2", force_regenerate=True)

    assert result is True
    assert fake_embedder.last_texts == ["Skill A", "Skill B", "Skill C", "Skill D"]
    assert fake_chroma.store_micro_skills_called_with["assignment_id"] == "assignment_2"
    assert fake_chroma.store_micro_skills_called_with["embeddings"] == [[0.1], [0.2], [0.3], [0.4]]
    assert fake_chroma.store_micro_skills_called_with["force_regenerate"] is True


def test_store_micro_skills_rejects_embedding_count_mismatch(monkeypatch):
    pipeline = rag_pipeline_module.RAGPipeline()

    fake_embedder = FakeEmbedder([[0.1], [0.2]])
    fake_chroma = FakeChromaClient()

    monkeypatch.setattr(rag_pipeline_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(rag_pipeline_module, "get_chroma_client", lambda: fake_chroma)

    skills = [
        {"text": "Skill A", "rank": 1, "weight": 4},
        {"text": "Skill B", "rank": 2, "weight": 3},
        {"text": "Skill C", "rank": 3, "weight": 2},
        {"text": "Skill D", "rank": 4, "weight": 1},
    ]

    result = pipeline.store_micro_skills(skills, "assignment_3")

    assert result is False
    assert fake_chroma.store_micro_skills_called_with is None


def test_store_teacher_references_rejects_invalid_reference_payload(monkeypatch):
    pipeline = rag_pipeline_module.RAGPipeline()

    fake_embedder = FakeEmbedder([[0.1]])
    fake_chroma = FakeChromaClient()

    monkeypatch.setattr(rag_pipeline_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(rag_pipeline_module, "get_chroma_client", lambda: fake_chroma)

    references = [
        {"rank": 1, "snippet": "", "line_start": 3, "line_end": 3},
    ]

    result = pipeline.store_teacher_references(references, "assignment_4")

    assert result is False
    assert fake_chroma.store_teacher_references_called_with is None


def test_retrieve_teacher_reference_rejects_invalid_skill_rank(monkeypatch):
    pipeline = rag_pipeline_module.RAGPipeline()

    fake_chroma = FakeChromaClient()
    monkeypatch.setattr(rag_pipeline_module, "get_chroma_client", lambda: fake_chroma)

    result = pipeline.retrieve_teacher_reference("assignment_5", 0)

    assert result is None


def test_assignment_exists_and_clear_assignment_delegate_to_chroma(monkeypatch):
    pipeline = rag_pipeline_module.RAGPipeline()

    fake_chroma = FakeChromaClient()
    fake_chroma.assignment_exists_value = True

    monkeypatch.setattr(rag_pipeline_module, "get_chroma_client", lambda: fake_chroma)

    assert pipeline.assignment_exists("assignment_6") is True
    assert pipeline.clear_assignment("assignment_6") is True
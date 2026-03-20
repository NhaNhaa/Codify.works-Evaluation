from backend.rag.chroma_client import ChromaClient


class FakeCollection:
    def __init__(self):
        self.records = []

    def get(self, where=None):
        matched = []

        for record in self.records:
            metadata = record["metadata"]

            if where is None:
                matched.append(record)
                continue

            if "$and" in where:
                conditions = where["$and"]
                is_match = all(
                    metadata.get(list(condition.keys())[0]) == list(condition.values())[0]
                    for condition in conditions
                )
            else:
                is_match = all(metadata.get(key) == value for key, value in where.items())

            if is_match:
                matched.append(record)

        return {
            "ids": [record["id"] for record in matched],
            "documents": [record["document"] for record in matched],
            "metadatas": [record["metadata"] for record in matched],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for record_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            existing_index = next(
                (index for index, record in enumerate(self.records) if record["id"] == record_id),
                None,
            )

            payload = {
                "id": record_id,
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

            if existing_index is None:
                self.records.append(payload)
            else:
                self.records[existing_index] = payload

    def delete(self, ids):
        self.records = [record for record in self.records if record["id"] not in ids]


def build_client():
    client = ChromaClient.__new__(ChromaClient)
    client.client = object()
    client.skills_collection = FakeCollection()
    client.references_collection = FakeCollection()
    return client


def test_validate_skill_payload_rejects_weight_sum_mismatch():
    client = build_client()

    skills = [
        {"text": "Skill A", "rank": 1, "weight": 4},
        {"text": "Skill B", "rank": 2, "weight": 3},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    assert client._validate_skill_payload(skills, embeddings, "assignment_1") is False


def test_store_micro_skills_rejects_existing_assignment_without_force_regenerate():
    client = build_client()

    existing_skills = [
        {"text": "Old skill", "rank": 1, "weight": 5},
        {"text": "Old skill 2", "rank": 2, "weight": 3},
        {"text": "Old skill 3", "rank": 3, "weight": 2},
    ]
    existing_embeddings = [[0.1], [0.2], [0.3]]

    assert client.store_micro_skills(
        skills=existing_skills,
        assignment_id="assignment_1",
        embeddings=existing_embeddings,
        force_regenerate=False,
    ) is True

    new_skills = [
        {"text": "New skill A", "rank": 1, "weight": 4},
        {"text": "New skill B", "rank": 2, "weight": 3},
        {"text": "New skill C", "rank": 3, "weight": 2},
        {"text": "New skill D", "rank": 4, "weight": 1},
    ]
    new_embeddings = [[0.5], [0.6], [0.7], [0.8]]

    assert client.store_micro_skills(
        skills=new_skills,
        assignment_id="assignment_1",
        embeddings=new_embeddings,
        force_regenerate=False,
    ) is False


def test_store_teacher_references_force_regenerate_does_not_delete_micro_skills():
    client = build_client()

    skills = [
        {"text": "Skill A", "rank": 1, "weight": 4},
        {"text": "Skill B", "rank": 2, "weight": 3},
        {"text": "Skill C", "rank": 3, "weight": 2},
        {"text": "Skill D", "rank": 4, "weight": 1},
    ]
    skill_embeddings = [[0.1], [0.2], [0.3], [0.4]]

    references_v1 = [
        {"rank": 1, "snippet": "temp = a;", "line_start": 1, "line_end": 1},
        {"rank": 2, "snippet": "a = b;", "line_start": 2, "line_end": 2},
        {"rank": 3, "snippet": "b = temp;", "line_start": 3, "line_end": 3},
        {"rank": 4, "snippet": "printf(\"done\");", "line_start": 4, "line_end": 4},
    ]
    reference_embeddings_v1 = [[0.11], [0.22], [0.33], [0.44]]

    references_v2 = [
        {"rank": 1, "snippet": "temp = arr[0];", "line_start": 5, "line_end": 5},
        {"rank": 2, "snippet": "arr[0] = arr[1];", "line_start": 6, "line_end": 6},
        {"rank": 3, "snippet": "arr[1] = temp;", "line_start": 7, "line_end": 7},
        {"rank": 4, "snippet": "printf(\"ok\");", "line_start": 8, "line_end": 8},
    ]
    reference_embeddings_v2 = [[0.55], [0.66], [0.77], [0.88]]

    assert client.store_micro_skills(
        skills=skills,
        assignment_id="assignment_2",
        embeddings=skill_embeddings,
        force_regenerate=False,
    ) is True

    assert client.store_teacher_references(
        references=references_v1,
        assignment_id="assignment_2",
        embeddings=reference_embeddings_v1,
        force_regenerate=False,
    ) is True

    assert client.store_teacher_references(
        references=references_v2,
        assignment_id="assignment_2",
        embeddings=reference_embeddings_v2,
        force_regenerate=True,
    ) is True

    retrieved_skills = client.retrieve_micro_skills("assignment_2")
    assert len(retrieved_skills) == 4

    retrieved_reference = client.retrieve_teacher_reference("assignment_2", 1)
    assert retrieved_reference["snippet"] == "temp = arr[0];"


def test_retrieve_micro_skills_returns_rank_sorted_results():
    client = build_client()

    skills = [
        {"text": "Skill rank 3", "rank": 3, "weight": 2},
        {"text": "Skill rank 1", "rank": 1, "weight": 4},
        {"text": "Skill rank 2", "rank": 2, "weight": 3},
        {"text": "Skill rank 4", "rank": 4, "weight": 1},
    ]
    embeddings = [[0.1], [0.2], [0.3], [0.4]]

    assert client.store_micro_skills(
        skills=skills,
        assignment_id="assignment_3",
        embeddings=embeddings,
        force_regenerate=False,
    ) is True

    result = client.retrieve_micro_skills("assignment_3")

    assert [item["rank"] for item in result] == [1, 2, 3, 4]
    assert [item["text"] for item in result] == [
        "Skill rank 1",
        "Skill rank 2",
        "Skill rank 3",
        "Skill rank 4",
    ]


def test_retrieve_teacher_reference_returns_expected_payload():
    client = build_client()

    references = [
        {"rank": 1, "snippet": "temp = a;", "line_start": 10, "line_end": 10},
        {"rank": 2, "snippet": "a = b;", "line_start": 11, "line_end": 11},
        {"rank": 3, "snippet": "b = temp;", "line_start": 12, "line_end": 12},
        {"rank": 4, "snippet": "printf(\"ok\");", "line_start": 13, "line_end": 13},
    ]
    embeddings = [[0.1], [0.2], [0.3], [0.4]]

    assert client.store_teacher_references(
        references=references,
        assignment_id="assignment_4",
        embeddings=embeddings,
        force_regenerate=False,
    ) is True

    result = client.retrieve_teacher_reference("assignment_4", 3)

    assert result == {
        "snippet": "b = temp;",
        "rank": 3,
        "line_start": 12,
        "line_end": 12,
        "status": "PASS",
    }
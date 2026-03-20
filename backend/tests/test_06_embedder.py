from backend.rag import embedder as embedder_module


class FakeEmbeddings:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        data = []
        for text in texts:
            text_length = float(len(text))
            word_count = float(len(text.split()))
            data.append([text_length, word_count])
        return FakeEmbeddings(data)


def test_embed_texts_returns_embeddings_for_valid_input(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embedder_module.Embedder()
    result = embedder.embed_texts(["Use scanf to read input", "Avoid data loss"])

    assert len(result) == 2
    assert all(isinstance(vector, list) for vector in result)
    assert all(len(vector) == 2 for vector in result)


def test_embed_texts_rejects_invalid_item_without_silent_dropping(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embedder_module.Embedder()
    result = embedder.embed_texts(["Valid text", "   ", "Another valid text"])

    assert result == []


def test_embed_single_returns_one_vector(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embedder_module.Embedder()
    result = embedder.embed_single("Use temporary variable")

    assert isinstance(result, list)
    assert len(result) == 2


def test_embed_single_returns_empty_for_invalid_text(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embedder_module.Embedder()
    result = embedder.embed_single("   ")

    assert result == []


def test_get_embedder_returns_singleton(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    monkeypatch.setattr(embedder_module, "_embedder_instance", None)

    first = embedder_module.get_embedder()
    second = embedder_module.get_embedder()

    assert first is second
from langchain_core.documents import Document

from src.vectordb import store as vector_store


class RecordingStore:
    def __init__(self):
        self.deletions = []
        self.insertions = []

    def delete(self, **kwargs):
        self.deletions.append(kwargs)

    def add_documents(self, documents, ids):
        self.insertions.append((documents, ids))


def test_reindex_replaces_only_matching_doc_id(monkeypatch):
    fake = RecordingStore()
    monkeypatch.setattr(vector_store, "get_store", lambda *args, **kwargs: fake)
    docs = [Document(page_content="updated", metadata={"document_id": "doc-a", "source": "a.txt"})]
    assert vector_store.index_documents(docs) == 1
    assert fake.deletions == [{"where": {"document_id": {"$eq": "doc-a"}}}]
    assert len(fake.insertions) == 1
    assert fake.insertions[0][0] == docs


def test_configurable_collection_and_storage_path(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_store, "Chroma", lambda **kwargs: kwargs)
    store = vector_store.get_store(tmp_path / "custom-index", collection_name="resume_docs", load_embeddings=False)
    assert store["collection_name"] == "resume_docs"
    assert store["persist_directory"] == str(tmp_path / "custom-index")

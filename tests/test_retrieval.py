from langchain_core.documents import Document

from src.retrieval import search


class FakeStore:
    def __init__(self):
        self.kwargs = None

    def similarity_search(self, query, **kwargs):
        self.kwargs = kwargs
        return [
            Document(page_content="Cats use whiskers to sense their surroundings.", metadata={"document_id": "a", "source": "cats.txt"}),
            Document(page_content="Dogs have an excellent sense of smell.", metadata={"document_id": "b", "source": "dogs.txt"}),
        ]


def test_retrieval_applies_document_filter_and_hybrid_order(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(search, "get_store", lambda *args, **kwargs: store)
    docs, metrics = search.retrieve("whiskers cats", top_k=1, document_ids=["a"], hybrid=True)
    assert docs[0].metadata["source"] == "cats.txt"
    assert store.kwargs["filter"] == {"document_id": {"$eq": "a"}}
    assert metrics["candidate_count"] == 2


def test_empty_query_short_circuits():
    docs, metrics = search.retrieve("  ")
    assert docs == []
    assert metrics["candidate_count"] == 0

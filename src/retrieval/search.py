"""Retrieve relevant passages from the selected documents."""

from __future__ import annotations

import re
from time import perf_counter

from langchain_core.documents import Document

from src.vectordb.store import get_store

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def _filter_for_documents(document_ids: list[str] | None) -> dict | None:
    ids = [value for value in (document_ids or []) if value]
    if not ids:
        return None
    if len(ids) == 1:
        return {"document_id": {"$eq": ids[0]}}
    return {"document_id": {"$in": ids}}


def retrieve(query: str, *, top_k: int = 5, document_ids: list[str] | None = None, hybrid: bool = False,
             rerank: bool = False, persist_directory=None, collection_name: str | None = None) -> tuple[list[Document], dict[str, float]]:
    """Return passages and basic timing/count metrics."""
    if not query.strip():
        return [], {"retrieval_ms": 0.0, "candidate_count": 0}
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    started = perf_counter()
    store = get_store(persist_directory, collection_name=collection_name)
    candidate_k = min(max(top_k * 4, top_k), 40) if hybrid or rerank else top_k
    kwargs = {"k": candidate_k}
    where = _filter_for_documents(document_ids)
    if where:
        kwargs["filter"] = where
    candidates = store.similarity_search(query, **kwargs)

    if hybrid and candidates:
        query_terms = set(TOKEN_RE.findall(query.lower()))
        # Combine Chroma's rank with a lightweight keyword overlap score.
        ranked = []
        for rank, doc in enumerate(candidates):
            terms = set(TOKEN_RE.findall(doc.page_content.lower()))
            lexical = len(query_terms & terms) / max(len(query_terms), 1)
            semantic_rank = 1 / (rank + 1)
            ranked.append((0.65 * semantic_rank + 0.35 * lexical, doc))
        candidates = [doc for _, doc in sorted(ranked, key=lambda item: item[0], reverse=True)]

    if rerank and candidates:
        from sentence_transformers import CrossEncoder

        model = _get_reranker()
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = model.predict(pairs)
        candidates = [doc for _, doc in sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)]

    results = candidates[:top_k]
    return results, {"retrieval_ms": round((perf_counter() - started) * 1000, 1), "candidate_count": len(candidates)}


_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return _RERANKER

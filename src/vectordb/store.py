"""Operations on DocuMind's own persistent Chroma collection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.embeddings.model import get_embeddings


def get_store(persist_directory: Path | None = None, model_name: str | None = None, *, collection_name: str | None = None, load_embeddings: bool = True) -> Chroma:
    directory = persist_directory or settings.persist_directory
    directory.mkdir(parents=True, exist_ok=True)
    embedding_function = get_embeddings(model_name or settings.embedding_model) if load_embeddings else None
    return Chroma(collection_name=collection_name or settings.collection_name, persist_directory=str(directory), embedding_function=embedding_function)


def index_documents(documents: list[Document], persist_directory: Path | None = None, collection_name: str | None = None) -> int:
    if not documents:
        return 0
    store = get_store(persist_directory, collection_name=collection_name)
    # Re-indexing a source replaces its previous chunks, limited to this collection/document ID.
    document_ids = {doc.metadata.get("document_id") for doc in documents if doc.metadata.get("document_id")}
    for document_id in document_ids:
        store.delete(where={"document_id": {"$eq": document_id}})
    ids = []
    for i, doc in enumerate(documents):
        fingerprint = hashlib.sha256(f"{doc.metadata.get('document_id', '')}:{doc.metadata.get('page', '')}:{doc.metadata.get('row', '')}:{doc.page_content}".encode()).hexdigest()
        ids.append(fingerprint)
    store.add_documents(documents, ids=ids)
    return len(documents)


def delete_document(document_id: str, persist_directory: Path | None = None, collection_name: str | None = None) -> None:
    """Delete only chunks whose document_id belongs to this application document."""
    if not document_id:
        raise ValueError("A document ID is required.")
    get_store(persist_directory, collection_name=collection_name, load_embeddings=False).delete(where={"document_id": {"$eq": document_id}})


def list_documents(persist_directory: Path | None = None, collection_name: str | None = None) -> list[dict]:
    store = get_store(persist_directory, collection_name=collection_name, load_embeddings=False)
    records = store.get(include=["metadatas"])
    unique: dict[str, dict] = {}
    for metadata in records.get("metadatas", []):
        if metadata:
            doc_id = metadata.get("document_id")
            if doc_id:
                current = unique.setdefault(doc_id, {"document_id": doc_id, "source": metadata.get("source", "Unknown"), "file_type": metadata.get("file_type", "unknown"), "chunks": 0})
                current["chunks"] += 1
    result = sorted(unique.values(), key=lambda item: (item["source"].lower(), item["document_id"]))
    duplicate_sources: dict[str, int] = {}
    for item in result:
        duplicate_sources[item["source"]] = duplicate_sources.get(item["source"], 0) + 1
    for item in result:
        item["label"] = f"{item['source']} · {item['document_id'][:6]}" if duplicate_sources[item["source"]] > 1 else item["source"]
    return result

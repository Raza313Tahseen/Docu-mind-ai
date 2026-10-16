import pytest
from langchain_core.documents import Document

from src.chunking.splitter import split_documents


def test_chunks_preserve_metadata_and_limit_size():
    source = Document(page_content="word " * 500, metadata={"source": "a.txt", "page": 3})
    chunks = split_documents([source], chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 300 for chunk in chunks)
    assert all(chunk.metadata["source"] == "a.txt" and chunk.metadata["page"] == 3 for chunk in chunks)


def test_invalid_chunk_settings_are_rejected():
    with pytest.raises(ValueError):
        split_documents([], chunk_size=50)
    with pytest.raises(ValueError):
        split_documents([], chunk_size=100, chunk_overlap=100)

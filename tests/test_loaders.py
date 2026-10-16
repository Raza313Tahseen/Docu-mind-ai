import json
import io

import pytest

from src.ingestion.loaders import load_uploaded_file, load_webpage


def test_load_text_includes_source_metadata():
    docs = load_uploaded_file("notes.txt", b"A note about retrieval.")
    assert docs[0].page_content == "A note about retrieval."
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["document_id"]


def test_load_csv_creates_one_document_per_data_row():
    docs = load_uploaded_file("people.csv", b"name,role\nAda,engineer\n")
    assert len(docs) == 1
    assert "name: Ada" in docs[0].page_content
    assert docs[0].metadata["row"] == 2


def test_load_docx_extracts_paragraphs():
    from docx import Document as WordDocument

    buffer = io.BytesIO()
    document = WordDocument()
    document.add_paragraph("A DOCX test passage.")
    document.save(buffer)
    docs = load_uploaded_file("guide.docx", buffer.getvalue())
    assert "A DOCX test passage." in docs[0].page_content
    assert docs[0].metadata["file_type"] == "docx"


def test_load_json_and_reject_invalid_or_empty_inputs():
    docs = load_uploaded_file("record.json", json.dumps({"title": "RAG"}).encode())
    assert '"title": "RAG"' in docs[0].page_content
    with pytest.raises(ValueError, match="Unsupported"):
        load_uploaded_file("image.png", b"data")
    with pytest.raises(ValueError, match="empty"):
        load_uploaded_file("empty.txt", b"")


def test_webpage_url_validation_happens_before_network():
    with pytest.raises(ValueError, match="http"):
        load_webpage("localhost")

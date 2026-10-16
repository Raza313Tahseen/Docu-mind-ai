"""Load supported files and webpages into LangChain Documents."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".json", ".txt"}


def _doc_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]


def _text_documents(text: str, source: str, kind: str) -> list[Document]:
    if not text.strip():
        return []
    return [Document(page_content=text.strip(), metadata={"source": source, "document_id": _doc_id(source), "file_type": kind})]


def load_uploaded_file(name: str, content: bytes) -> list[Document]:
    """Parse one uploaded file. Invalid or empty inputs produce a useful ValueError."""
    path = Path(name)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")
    if not content:
        raise ValueError("The uploaded file is empty.")
    source = path.name
    document_id = _doc_id(f"{source}:{hashlib.sha256(content).hexdigest()}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": source, "document_id": document_id, "file_type": "pdf", "page": page_number}))
        if not documents:
            raise ValueError("No extractable text was found. This may be a scanned PDF; OCR is not included.")
        return documents

    if suffix == ".docx":
        from docx import Document as WordDocument

        word = WordDocument(io.BytesIO(content))
        text = "\n".join(p.text for p in word.paragraphs if p.text.strip())
        for table in word.tables:
            text += "\n" + "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        documents = _text_documents(text, source, "docx")
    elif suffix == ".csv":
        decoded = content.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(decoded)))
        if not rows:
            documents = []
        else:
            header = rows[0]
            documents = [
                Document(page_content=" | ".join(f"{key}: {value}" for key, value in zip(header, row)), metadata={"source": source, "document_id": _doc_id(source), "file_type": "csv", "row": row_number})
                for row_number, row in enumerate(rows[1:], start=2) if any(cell.strip() for cell in row)
            ]
    elif suffix == ".json":
        value = json.loads(content.decode("utf-8-sig"))
        documents = _text_documents(json.dumps(value, ensure_ascii=False, indent=2), source, "json")
    else:
        documents = _text_documents(content.decode("utf-8-sig"), source, "txt")

    if not documents:
        raise ValueError("No readable text or rows were found in this file.")
    for document in documents:
        document.metadata["document_id"] = document_id
    return documents


def load_webpage(url: str) -> list[Document]:
    """Fetch a public webpage and keep its URL as source metadata."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete http:// or https:// URL.")
    from langchain_community.document_loaders import WebBaseLoader

    docs = WebBaseLoader(web_paths=(url.strip(),), requests_kwargs={"timeout": 15}).load()
    source_id = _doc_id(url.strip())
    result = [Document(page_content=d.page_content, metadata={**d.metadata, "source": url.strip(), "document_id": source_id, "file_type": "webpage"}) for d in docs if d.page_content.strip()]
    if not result:
        raise ValueError("The webpage returned no readable text.")
    return result

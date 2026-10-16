"""Configuration shared by the application modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    # Streamlit Community Cloud secrets are exposed through st.secrets.
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings; persistent files stay inside this project's ignored data dir."""

    persist_directory: Path = ROOT_DIR / ".chroma"
    collection_name: str = "documind_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = _secret("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
    groq_api_key: str | None = _secret("GROQ_API_KEY")


settings = Settings()

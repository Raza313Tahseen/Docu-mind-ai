"""Configuration shared by the application modules."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_KEY_SETUP_MESSAGE = (
    "To enable answer generation, configure GROQ_API_KEY. For local development, "
    "add it to the project's .env file. On Streamlit Community Cloud, add "
    "GROQ_API_KEY as a root-level value in App settings → Secrets, or set it as "
    "an environment variable, then restart or redeploy the app."
)
MISSING_GROQ_KEY_MESSAGE = f"Answer generation is unavailable because GROQ_API_KEY was not found. {GROQ_KEY_SETUP_MESSAGE}"

_UNSET = object()


def _secret(
    name: str,
    default: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    secrets=_UNSET,
) -> str | None:
    """Read a non-empty setting from the environment, then Streamlit secrets.

    Environment variables include values loaded from the local .env file and
    values configured directly in a hosting environment. Streamlit secrets are
    read at call time so settings are not frozen before the app runtime is ready.
    """
    environment = os.environ if environ is None else environ
    value = environment.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()

    if secrets is _UNSET:
        try:
            import streamlit as st

            secrets = st.secrets
        except ImportError:
            secrets = {}
        except FileNotFoundError:
            # Streamlit raises StreamlitSecretNotFoundError when no secrets file
            # is configured (the exception subclasses FileNotFoundError).
            secrets = {}

    try:
        value = secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        value = default
    return value.strip() if isinstance(value, str) and value.strip() else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings; persistent files stay inside this project's ignored data dir."""

    persist_directory: Path = ROOT_DIR / ".chroma"
    collection_name: str = "documind_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def groq_model(self) -> str:
        return _secret("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL

    @property
    def groq_api_key(self) -> str | None:
        return _secret("GROQ_API_KEY")


settings = Settings()

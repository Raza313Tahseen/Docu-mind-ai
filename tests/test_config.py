import sys
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src import config
from src.llm import answer


def test_secret_prefers_environment_over_streamlit_secrets():
    value = config._secret(
        "GROQ_API_KEY",
        environ={"GROQ_API_KEY": " environment-key "},
        secrets={"GROQ_API_KEY": "cloud-key"},
    )

    assert value == "environment-key"


def test_secret_reads_streamlit_secret_when_environment_is_empty():
    value = config._secret(
        "GROQ_API_KEY",
        environ={},
        secrets={"GROQ_API_KEY": " cloud-key "},
    )

    assert value == "cloud-key"


def test_secret_uses_default_for_missing_or_blank_values():
    value = config._secret(
        "GROQ_MODEL",
        "default-model",
        environ={"GROQ_MODEL": "  "},
        secrets={"GROQ_MODEL": " "},
    )

    assert value == "default-model"


def test_settings_resolve_api_key_and_model_dynamically(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-one")
    monkeypatch.setenv("GROQ_MODEL", "test-model-one")
    assert config.settings.groq_api_key == "test-key-one"
    assert config.settings.groq_model == "test-model-one"

    monkeypatch.setenv("GROQ_API_KEY", "test-key-two")
    monkeypatch.setenv("GROQ_MODEL", "test-model-two")
    assert config.settings.groq_api_key == "test-key-two"
    assert config.settings.groq_model == "test-model-two"


def test_settings_read_streamlit_secrets_at_runtime(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(secrets={"GROQ_API_KEY": "cloud-test-key", "GROQ_MODEL": "cloud-test-model"}),
    )

    assert config.settings.groq_api_key == "cloud-test-key"
    assert config.settings.groq_model == "cloud-test-model"


def test_generate_answer_uses_configured_key_and_separate_model(monkeypatch):
    calls = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            calls["client"] = kwargs

        def invoke(self, messages):
            calls["messages"] = messages
            return SimpleNamespace(content="Grounded answer.")

    monkeypatch.setattr(answer, "settings", SimpleNamespace(groq_api_key="test-key", groq_model="test-model"))
    monkeypatch.setitem(sys.modules, "langchain_groq", SimpleNamespace(ChatGroq=FakeChatGroq))
    docs = [Document(page_content="Evidence passage.", metadata={"source": "sample.txt"})]

    result, _elapsed_ms = answer.generate_answer("What is the answer?", docs)

    assert result == "Grounded answer."
    assert calls["client"] == {"model": "test-model", "temperature": 0, "api_key": "test-key"}


def test_missing_key_message_covers_local_and_streamlit_cloud(monkeypatch):
    monkeypatch.setattr(answer, "settings", SimpleNamespace(groq_api_key=None, groq_model="test-model"))

    with pytest.raises(RuntimeError) as exc_info:
        answer.generate_answer("Question", [])

    message = str(exc_info.value)
    assert ".env" in message
    assert "Streamlit Community Cloud" in message
    assert "App settings" in message

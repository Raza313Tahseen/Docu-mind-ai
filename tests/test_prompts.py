import pytest
from langchain_core.documents import Document

from src.prompts.grounded import build_answer_messages, format_context


def test_prompt_includes_source_labels_and_history():
    docs = [Document(page_content="The model uses MiniLM.", metadata={"source": "readme.txt", "page": 2})]
    messages = build_answer_messages("Which model?", docs, [{"role": "user", "content": "What is this app?"}])
    rendered = "\n".join(content for _, content in messages)
    assert "[Source 1: readme.txt, page/row 2]" in rendered
    assert "What is this app?" in rendered
    assert "only the provided source passages" in messages[0][1]


def test_prompt_requires_sources():
    with pytest.raises(ValueError, match="source passage"):
        build_answer_messages("question", [])


def test_format_context_empty():
    assert format_context([]) == ""

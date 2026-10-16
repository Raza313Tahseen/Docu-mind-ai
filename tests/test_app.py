from streamlit.testing.v1 import AppTest


def test_empty_collection_explains_how_to_start(tmp_path):
    app = AppTest.from_file("app.py", default_timeout=60).run()
    app.text_input[0].set_value("app_test_empty_collection").run()
    app.text_input[1].set_value(str(tmp_path / "chroma")).run()
    assert not app.exception
    assert any("No documents are indexed" in item.value for item in app.warning)
    assert len(app.chat_input) == 1

    app.chat_input[0].set_value("How are you?").run()
    assert not app.exception
    assert any("collection has no indexed documents" in item.value for item in app.markdown)

"""DocuMind AI — Document Assistant."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="DocuMind AI — Document Assistant", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { background: #101116; font-family: 'DM Sans', sans-serif; }
.stApp, [data-testid="stAppViewContainer"] { color: #f1f1f4; }
[data-testid="stHeader"] { background: rgba(16,17,22,.92); }
section[data-testid="stSidebar"] { background: #191a20; border-right: 1px solid #2b2c34; }
section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
.block-container { max-width: 940px; padding-top: 3.1rem; padding-bottom: 7rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: #f5f5f7; }
.brand { display:flex; align-items:center; gap:.7rem; margin: .15rem 0 1.8rem; }
.brand-mark { display:grid; place-items:center; width:38px; height:38px; border-radius:12px; background:linear-gradient(135deg,#ff8a65,#fa5e58); font-size:1.15rem; }
.brand-name { font-family:'Manrope',sans-serif; font-size:1.05rem; font-weight:800; color:#f7f7f9; }
.config-label { color:#f0f0f3; font-weight:700; font-size:1.12rem; margin:.2rem 0 .8rem; }
.main-title { text-align:center; font-family:'Manrope',sans-serif; font-size:2.6rem; font-weight:800; letter-spacing:-.055em; margin:.25rem 0 .3rem; color:#fafafd; }
.main-subtitle { text-align:center; color:#9b9ca8; font-size:.95rem; margin:0 auto 2rem; }
[data-testid="stChatMessage"] { background:#1b1c23; border:1px solid #292a33; border-radius:17px; padding:.75rem 1rem; }
[data-testid="stChatInput"] { background:#1b1c23; border:1px solid #34353e; border-radius:16px; }
[data-testid="stChatInput"] textarea { color:#f1f1f4; }
[data-testid="stChatInput"] textarea::placeholder { color:#9697a3; }
div[data-testid="stExpander"] { border:1px solid #30313a; border-radius:12px; background:#1b1c23; }
div[data-testid="stFileUploader"] { border-color:#34353e; }
[data-testid="stSelectboxVirtualDropdown"] { max-height:200px !important; overflow-y:auto !important; }
.stCaption, [data-testid="stCaptionContainer"] { color:#9b9ca8; }
.stButton > button[kind="primary"] { border:0; background:linear-gradient(100deg,#ff8864,#f65e58); color:white; font-weight:700; }
div[data-testid="stMetric"] { background:#1b1c23; border:1px solid #30313a; padding:.75rem; border-radius:12px; }
hr { border-color:#30313a; }
</style>
""", unsafe_allow_html=True)

try:
    from src.chunking.splitter import split_documents
    from src.config import ROOT_DIR, settings
    from src.ingestion.loaders import load_uploaded_file, load_webpage
    from src.llm.answer import contextualize_question, generate_answer
    from src.retrieval.search import retrieve
    from src.vectordb.store import delete_document, index_documents, list_documents
except ImportError as exc:
    st.error("Some project packages are missing. In the project terminal run `python -m pip install -r requirements.txt`, then restart Streamlit.")
    st.code(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_status" not in st.session_state:
    st.session_state.index_status = "Add a document to start a conversation."
if "index_errors" not in st.session_state:
    st.session_state.index_errors = []

with st.sidebar:
    st.markdown("<div class='brand'><div class='brand-mark'>📚</div><div class='brand-name'>DocuMind AI</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='config-label'>RAG configuration</div>", unsafe_allow_html=True)
    doc_type = st.selectbox("Document type", ["All supported", "PDF", "DOCX", "CSV", "JSON", "TXT", "Webpage"], key="doc_type_input")
    collection_name = st.text_input("Collection name", value=settings.collection_name, key="collection_name_input", help="A separate Chroma collection for this workspace.")
    chroma_path_input = st.text_input("Chroma path", value="./.chroma", key="chroma_path_input", help="Persistent local folder for the vector index.")
    chroma_path = Path(chroma_path_input).expanduser()
    if not chroma_path.is_absolute():
        chroma_path = (ROOT_DIR / chroma_path).resolve()
    model_name = st.text_input("Groq chat model", value=settings.groq_model, key="groq_model_input")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5, key="retrieved_chunks")

    with st.expander("Advanced retrieval", expanded=False):
        chunk_size = st.slider("Chunk size", min_value=300, max_value=1800, value=900, step=100, key="chunk_size")
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=150, step=25, key="chunk_overlap")
        hybrid = st.toggle("Hybrid keyword + semantic search", value=False, key="hybrid_search")
        rerank = st.toggle("Rerank passages", value=False, key="rerank_passages", help="Downloads a second local model on first use.")

    st.divider()
    with st.expander("Add documents", expanded=False):
        extensions = {"PDF": ["pdf"], "DOCX": ["docx"], "CSV": ["csv"], "JSON": ["json"], "TXT": ["txt"]}
        allowed_types = extensions.get(doc_type, ["pdf", "docx", "csv", "json", "txt"])
        uploads = st.file_uploader("Upload files", type=allowed_types, accept_multiple_files=True, disabled=doc_type == "Webpage", key="document_upload")
        st.caption("Selecting a file only stages it. Click **Index uploaded files** to extract, embed, and save it for chat.")
        if st.button("Index uploaded files", type="primary", use_container_width=True, disabled=not uploads or not collection_name.strip()):
            total = 0
            errors = []
            progress = st.progress(0, text="Preparing documents…")
            for i, uploaded in enumerate(uploads):
                progress.progress(i / len(uploads), text=f"Reading and embedding {uploaded.name}… first use may download the local model")
                try:
                    docs = load_uploaded_file(uploaded.name, uploaded.getvalue())
                    chunks = split_documents(docs, chunk_size, chunk_overlap)
                    total += index_documents(chunks, chroma_path, collection_name.strip())
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")
                progress.progress((i + 1) / len(uploads), text=f"Processed {i + 1} of {len(uploads)} files")
            st.session_state.index_status = f"Indexed {total} chunks." + (f" {len(errors)} file(s) had errors." if errors else "")
            st.session_state.index_errors = errors
            st.rerun()

        webpage_url = st.text_input("Or add a webpage URL", placeholder="https://example.com/article", key="webpage_url")
        if st.button("Fetch webpage", use_container_width=True, disabled=not webpage_url.strip() or not collection_name.strip()):
            try:
                docs = load_webpage(webpage_url)
                chunks = split_documents(docs, chunk_size, chunk_overlap)
                count = index_documents(chunks, chroma_path, collection_name.strip())
                st.session_state.index_status = f"Indexed {count} webpage chunks."
                st.session_state.index_errors = []
                st.success(st.session_state.index_status)
                st.rerun()
            except Exception as exc:
                st.session_state.index_errors = [str(exc)]
                st.error(f"Webpage indexing failed: {exc}")

    for index_error in st.session_state.index_errors:
        st.error(f"Indexing error: {index_error}")

    if not settings.groq_api_key:
        st.info("Add GROQ_API_KEY to `.env` to enable answer generation.")

namespace = f"{chroma_path}|{collection_name.strip()}"
if st.session_state.get("active_namespace") != namespace:
    st.session_state.active_namespace = namespace
    st.session_state.messages = []
    st.session_state.index_errors = []

try:
    all_documents = list_documents(chroma_path, collection_name.strip()) if collection_name.strip() else []
    documents = [item for item in all_documents if doc_type == "All supported" or item["file_type"] == doc_type.lower()]
except Exception as exc:
    documents = []
    st.sidebar.error(f"Could not open this collection: {exc}")

st.markdown("<div class='main-title'>DocuMind AI</div><div class='main-subtitle'>Advanced RAG-Based Document Intelligence<br>Ask a question about the documents in your collection.</div>", unsafe_allow_html=True)

with st.sidebar:
    with st.expander(f"Indexed sources · {len(documents)}", expanded=bool(documents)):
        if documents:
            source_options = {item["label"]: item["document_id"] for item in documents}
            source_widget_key = f"selected_sources_{abs(hash(namespace))}_{doc_type}"
            selected_sources = st.multiselect("Search in", list(source_options), default=list(source_options), key=source_widget_key, label_visibility="collapsed")
            selected_ids = [source_options[label] for label in selected_sources]
            st.caption(st.session_state.index_status)
            for item in documents:
                title = f"{item['label']} · {item['chunks']} chunks"
                with st.expander(title):
                    st.caption(f"Type: {item['file_type']} · ID: {item['document_id'][:10]}…")
                    if st.button("Remove from index", key=f"remove_{abs(hash(namespace))}_{item['document_id']}"):
                        try:
                            delete_document(item["document_id"], chroma_path, collection_name.strip())
                            st.session_state.index_status = f"Removed {item['source']} from this collection."
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not remove document: {exc}")
        else:
            selected_ids = []
            st.caption("No sources match this document type yet.")

if not st.session_state.messages:
    st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)
    if not documents:
        st.warning("No documents are indexed in this collection yet. Selecting a file only stages it: open **Add documents** in the sidebar and click **Index uploaded files**. Check for an indexing error above if the source count stays at 0.")
    elif not selected_ids:
        st.info("Select at least one source in **Indexed sources** in the sidebar to enable search.")
    else:
        st.caption(st.session_state.index_status)

if documents and selected_ids and not settings.groq_api_key:
    st.info("Search can find matching passages, but answer generation is unavailable until you add `GROQ_API_KEY` to `.env` and restart Streamlit.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander(f"Sources and passages · {len(item['sources'])}"):
                for source in item["sources"]:
                    st.markdown(f"**{source['label']} · {source['source']}**")
                    st.caption(source["passage"])

if not documents:
    input_placeholder = "Ask a question — I’ll help you connect your documents"
elif not selected_ids:
    input_placeholder = "Ask a question, then select a source in the sidebar"
else:
    input_placeholder = "Ask a question about your indexed documents"

prompt = st.chat_input(input_placeholder)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        sources = []
        passages = []
        metrics = None
        answer_ms = 0
        if not documents:
            answer = "I received your question, but this collection has no indexed documents yet. Open **Add documents** in the sidebar, upload a file, and click **Index uploaded files**. Then ask me again."
        elif not selected_ids:
            answer = "I received your question, but no sources are selected. Expand **Indexed sources** in the sidebar and select at least one document."
        else:
            try:
                with st.spinner("Searching your selected documents and preparing an answer…"):
                    history = st.session_state.messages[:-1]
                    retrieval_query = contextualize_question(prompt, history, model_name) if history else prompt
                    passages, metrics = retrieve(retrieval_query, top_k=top_k, document_ids=selected_ids, hybrid=hybrid, rerank=rerank,
                                                 persist_directory=chroma_path, collection_name=collection_name.strip())
                    if not passages:
                        answer = "I couldn't find enough information in the selected sources to answer that. Try selecting a different document or adding relevant material."
                    elif not settings.groq_api_key:
                        answer = "Relevant passages were found, but answer generation needs a Groq API key. Add GROQ_API_KEY to your local `.env` file and restart the app."
                    else:
                        answer, answer_ms = generate_answer(prompt, passages, history=history, model_name=model_name)
                for index, doc in enumerate(passages, 1):
                    location = doc.metadata.get("page") or doc.metadata.get("row")
                    suffix = f" · page/row {location}" if location else ""
                    passage = doc.page_content[:900] + ("…" if len(doc.page_content) > 900 else "")
                    sources.append({"label": f"Source {index}", "source": f"{doc.metadata.get('source', 'Unknown')}{suffix}", "passage": passage})
            except Exception as exc:
                answer = f"I couldn't complete that request: {exc}"
        st.markdown(answer)
        if sources:
            with st.expander(f"Sources and passages · {len(sources)} retrieved"):
                for source in sources:
                    st.markdown(f"**{source['label']} · {source['source']}**")
                    st.caption(source["passage"])
        if metrics:
            st.caption(f"Retrieval {metrics['retrieval_ms']} ms · Answer {answer_ms} ms · {len(passages)} passages")
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

if st.session_state.messages and st.button("Clear chat", key="clear_chat"):
    st.session_state.messages = []
    st.rerun()

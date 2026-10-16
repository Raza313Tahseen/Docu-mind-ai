# DocuMind AI

DocuMind AI is a document question-answering app. Add documents or a public webpage, ask a question in plain language, and get an answer based on relevant passages from the sources you selected. Answers include source passages so you can check where the information came from.

## The problem it solves

Useful information is often spread across reports, spreadsheets, notes, and webpages. Finding a specific detail can mean opening files one by one and searching manually. DocuMind AI brings those sources into one searchable collection and lets you ask questions across them. It is designed to make document research faster while keeping answers tied to source material.

## What it can do

- Read PDF, DOCX, CSV, JSON, and TXT files, and fetch readable public webpages.
- Split content into overlapping passages while keeping source details such as PDF pages and CSV rows.
- Create document embeddings locally with `sentence-transformers/all-MiniLM-L6-v2` and store them in a persistent Chroma index.
- Search selected sources using semantic search, with optional keyword blending and cross-encoder reranking.
- Generate answers from retrieved passages using a Groq chat model. The app asks the model to cite sources and say when the evidence is insufficient.
- Use recent messages to support follow-up questions during the current chat session.
- Show retrieved passages and basic retrieval and answer timings, and remove individual sources from the index.

## How it works

```text
Files or webpage
       ↓
Text extraction and source metadata
       ↓
Overlapping passages → local embeddings → persistent Chroma index
                                          ↓
Question → retrieve passages → Groq answer with source references
```

The main code is organized by responsibility:

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and chat flow |
| `src/ingestion/` | File and webpage text extraction |
| `src/chunking/` | Passage splitting and metadata handling |
| `src/embeddings/` | Local embedding model setup |
| `src/vectordb/` | Persistent Chroma storage and source management |
| `src/retrieval/` | Semantic search, keyword blending, and optional reranking |
| `src/prompts/` | Grounded answer instructions and source formatting |
| `src/llm/` | Groq answer generation and follow-up question handling |
| `src/evaluation/` | Sample questions and lightweight evaluation metrics |

## Requirements

- Python 3.11
- A Groq API key to generate answers
- Internet access on first run to download the local embedding model; internet is also needed for Groq and webpage fetching

The embedding model runs locally on CPU. Its first download can take time and disk space. Enabling reranking downloads an additional model on first use.

## Install and run on Windows

Open PowerShell in the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Groq settings:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_MODEL` is optional; the app uses `openai/gpt-oss-120b` by default. Without `GROQ_API_KEY`, you can still index documents and retrieve relevant passages, but the app cannot generate an answer.

Start the app:

```powershell
python -m streamlit run app.py
```

## Use the app

1. Open **Add documents** in the sidebar and upload one or more supported files, or enter a public webpage URL.
2. For uploaded files, click **Index uploaded files**. Selecting files only stages them; indexing extracts text, creates embeddings, and saves the passages.
3. Choose which indexed sources to search under **Indexed sources**. You can adjust the number of retrieved passages and advanced retrieval options in the sidebar.
4. Ask a question in the chat. Review the displayed source passages to verify the answer.
5. Use **Clear chat** to clear the conversation. This keeps the indexed sources. Use **Remove from index** on a source to delete that document's passages from the collection.

The local vector index is stored in `.chroma/` by default and persists between runs. The collection name and storage path can be changed in the sidebar.

## Evaluation

`src/evaluation/samples.json` contains starter questions and expected facts. After answering those questions against a document with the expected information, save results as a JSON array, for example:

```json
[
  {
    "question": "Where should I put the Groq API key?",
    "retrieved_passages": ["Set GROQ_API_KEY in your local .env file."],
    "answer": "Put the Groq API key in the local .env file."
  }
]
```

Run the evaluator with:

```powershell
python -m src.evaluation.run_eval eval_results.json
```

It reports lexical indicators for evidence coverage, answer relevance, and faithfulness coverage. These metrics are a lightweight aid, not a benchmark or proof of answer quality; inspect the retrieved passages and answers as well.

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository. Keep `.env`, `.venv/`, `.chroma/`, and uploaded files out of Git; `.gitignore` excludes local environment and index data.
2. Create a Streamlit Community Cloud app using the repository, branch, and `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add the Groq settings using TOML syntax:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```

4. Streamlit installs dependencies from `requirements.txt`. The embedding model needs memory and may take time to download and load.

The hosted filesystem may be cleared during redeployments or restarts, so the local `.chroma/` index is not durable in the cloud. Re-upload and re-index documents when needed. Do not use a public demo for sensitive documents.

## Limitations and privacy

- Scanned or image-only PDFs need OCR and may produce no extractable text.
- Large documents can take a while to process on CPU.
- Webpage fetching works with readable public pages; authenticated, blocked, or JavaScript-heavy sites may not load.
- Hybrid retrieval is a lightweight keyword blend, not a full BM25 search index. Reranking can improve ordering but adds CPU work.
- The project uses one local collection by default and is intended as a single-user demo, not a multi-user service.
- Document text is sent to Groq when answer generation is requested. Embeddings and the Chroma index are created locally on the machine running the app.
- Never commit your API key. This project has not been security-audited for confidential data.

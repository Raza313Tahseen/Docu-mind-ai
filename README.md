# DocuMind AI

DocuMind AI is a local-first document assistant. Add files or a public webpage, ask a question in plain language, and get an answer based on relevant passages from your selected sources.

## The problem it solves

Important details are often scattered across reports, spreadsheets, notes, and webpages. Finding one answer can mean opening and searching each source by hand. DocuMind AI brings those sources into one searchable collection so you can ask questions across them and inspect the passages behind each answer.

## How it works

The app extracts text, splits it into overlapping passages, creates embeddings on your machine, and stores them in a local Chroma database. When you ask a question, DocuMind retrieves relevant passages and sends those passages to Groq to generate a cited answer.

```text
PDF / DOCX / CSV / JSON / TXT / webpage
                  │
                  ▼
       Extract text and source details
                  │
                  ▼
       Split into overlapping passages
                  │
                  ▼
 Local sentence-transformer embeddings
                  │
                  ▼
      Persistent Chroma collection
                  │
Question ──► retrieve selected passages ──► Groq ──► cited answer
```

The project is local-first, but it is not fully offline: document parsing, embeddings, and the vector index run locally; Groq is used over the network for answer generation. Internet access is also needed to fetch webpages and download the embedding model the first time.

## Features

- Upload PDF, DOCX, CSV, JSON, and TXT files, or index a readable public webpage.
- Keep source metadata, including PDF page numbers and CSV row numbers.
- Configure chunk size and overlap before indexing.
- Search only the sources you select.
- Use semantic search, optional keyword blending, and optional cross-encoder reranking.
- Generate answers grounded in retrieved passages, with source references and an insufficient-evidence response.
- Ask follow-up questions using the current chat context.
- View source passages and basic retrieval and answer timings.
- Remove a source from the index or clear the chat without deleting indexed documents.

## Tech stack

| Technology | Role |
| --- | --- |
| Python 3.11 | Application language |
| Streamlit 1.50 | Web interface and chat experience |
| LangChain 0.3 | Document objects, text splitting, integrations, and model calls |
| Sentence Transformers 5.1 | Local embeddings and optional local reranking |
| `all-MiniLM-L6-v2` | Default embedding model; runs locally on CPU |
| Chroma 1.1 | Persistent vector storage in `.chroma/` |
| Groq | Hosted answer generation; default model is `openai/gpt-oss-120b` |
| PyPDF, python-docx, Python CSV/JSON | File text extraction |
| WebBaseLoader, Requests, Beautiful Soup | Public webpage loading and parsing |

Exact Python package versions are pinned in [`requirements.txt`](requirements.txt).

## Project structure

```text
Docu-mind-ai/
├── app.py                   # Streamlit application
├── requirements.txt         # Pinned Python dependencies
├── src/
│   ├── chunking/            # Split text and preserve metadata
│   ├── embeddings/          # Local Hugging Face embedding model
│   ├── evaluation/          # Sample questions and evaluation helpers
│   ├── ingestion/           # File and webpage loaders
│   ├── llm/                 # Groq answers and follow-up question rewriting
│   ├── prompts/             # Grounded answer prompt and source formatting
│   ├── retrieval/           # Semantic, hybrid, and reranked retrieval
│   └── vectordb/            # Persistent Chroma operations
└── tests/                   # Unit tests
```

## Requirements

- Python 3.11
- A Groq API key for answer generation
- Internet access for the initial embedding-model download, Groq requests, and webpage fetching

Document indexing and passage retrieval can run without a Groq key. The first indexing run downloads the embedding model; enabling reranking downloads an additional model. Both models run locally on CPU after download.

## Install and run

From the project directory, create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Groq settings:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_MODEL` is optional; the app uses `openai/gpt-oss-120b` by default. The model must be available to your Groq account. The app can also read these values from Streamlit secrets when deployed.

Run the app:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Use the app

1. Expand **Add documents** in the sidebar.
2. Upload files and click **Index uploaded files**, or enter a public webpage URL and click **Fetch webpage**. Uploading a file only stages it; it must be indexed before it can be searched.
3. Under **Indexed sources**, select the sources you want to search.
4. Ask a question in the chat and review the displayed source passages.
5. Use **Clear chat** to clear the current conversation. To remove indexed content, choose **Remove from index** for that source.

The default Chroma index is saved in `.chroma/` and persists between local runs. The collection name and storage path can be changed in the sidebar.

## Retrieval and answer flow

1. The loader extracts text and source metadata from each supported file or webpage.
2. The text splitter creates overlapping passages and carries their metadata forward.
3. The local embedding model converts passages into vectors, which Chroma stores with the source text and metadata.
4. The question is matched against the selected sources. Hybrid keyword blending and cross-encoder reranking are optional.
5. The retrieved passages and recent chat context are sent to the configured Groq model. The prompt asks it to cite sources and report when the evidence is insufficient.

## Evaluation

Starter questions and expected evidence are in `src/evaluation/samples.json`. Save manually captured answers and passages in a JSON file such as:

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

```bash
python -m src.evaluation.run_eval eval_results.json
```

It reports lexical evidence coverage, answer relevance, and faithfulness coverage. These lightweight indicators are not a benchmark or proof of answer quality; review the passages and answers as well.

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository. `.gitignore` excludes `.env`, virtual environments, local Chroma data, uploads, and logs.
2. Create a Streamlit Community Cloud app using the repository and `app.py` as the entry point.
3. Add the Groq values under **Settings → Secrets** in TOML format:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```

4. Streamlit installs packages from `requirements.txt`. The embedding model can need substantial memory and time to download and load.

Files on a hosted filesystem may be cleared on restart or redeployment. Treat the hosted `.chroma/` index as temporary and re-upload and re-index documents when needed.

## Limitations and privacy

- Scanned or image-only PDFs need OCR; OCR is not included.
- Large files can take time to embed on CPU.
- Webpage loading is for readable public pages; authenticated, blocked, and JavaScript-heavy sites may not load.
- Hybrid retrieval is a lightweight keyword blend, not a full BM25 index. Reranking uses extra CPU and downloads another model on first use.
- The app uses a local collection and is intended as a single-user project, not a multi-user service.
- When you request an answer, retrieved document passages and recent chat messages are sent to Groq. Embeddings and the Chroma index are created locally when running the app on your own machine.
- Keep `.env` private and never commit your API key. This project has not been security-audited for confidential data.

## Run tests

From the project root:

```bash
python -m pytest
```

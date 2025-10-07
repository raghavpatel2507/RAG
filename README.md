# RAG — Retrieval-Augmented Generation Toolkit

Lightweight Retrieval-Augmented Generation (RAG) starter project using LangChain, SentenceTransformers for embeddings, FAISS for vector search, and a Groq-powered LLM as an example. This repository provides utilities to load documents (PDF, TXT, CSV, DOCX, XLSX, JSON), chunk and embed them, build or load a FAISS index, and run simple retrieval + summarization flows.

This README explains how to set up the environment, ingest your documents, build a vectorstore, and run the example search/summarize script.

## Contents

- `main.py` — example runner that demonstrates loading documents, loading vectorstore and running a RAG query.
- `src/` — core modules:
  - `Data_loader.py` — document loaders for PDF, TXT, CSV, DOCX, XLSX, JSON.
  - `embedding.py` — chunking and embedding pipeline (SentenceTransformers).
  - `vectorstore.py` — FAISS-backed vector store utilities (build/save/load/query).
  - `search.py` — a small RAG-style wrapper that uses the vectorstore and an LLM to summarize retrieved context.
- `data/` — example data folder (place your documents here).
- `requirements.txt` — Python dependencies.

## Features

- Multi-format document ingestion (PDF, TXT, CSV, Excel, Word, JSON).
- Text chunking using LangChain text splitter.
- Embeddings via `sentence-transformers` (default: `all-MiniLM-L6-v2`).
- FAISS vector index persistence (index + metadata saved to disk).
- Simple RAG search + summarization using a Groq chat LLM (example).

## Requirements

- Python 3.8+ recommended.
- Linux / macOS / Windows (FAISS binary differs on platform; repository uses `faiss-cpu` in `requirements.txt`).

Install dependencies:

```bash
pip install uv
uv init
uv venv
source .venv/bin/activate
uv add -r requirements.txt
```

Notes:
- If you have a GPU and want FAISS GPU support, replace `faiss-cpu` with the appropriate `faiss-gpu` package and follow FAISS install instructions for your environment.
- `sentence-transformers` will download model weights on first run; ensure you have internet access.

## Environment variables

This project loads environment variables via `python-dotenv` in `src/search.py`. Create a `.env` file in the project root with (example):

```
# Groq API key (example LLM provider used in the project)
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gemma2-9b-it
PERSIST_DIR=faiss_store
```

Edit `src/search.py` or provide values via environment if you want to change defaults. The current `RAGSearch` implementation sets `groq_api_key` to an empty string in code, so ensure you set it in `.env` or modify the file to pass the key programmatically.

## Data layout

Place your documents under the `data/` directory. The loader will recursively search for supported file types:

- `data/pdf/*.pdf`
- `data/text_files/*.txt`
- `data/*.csv`, `data/**/*.xlsx`, `data/**/*.docx`, `data/**/*.json`

Example project already includes some PDFs and text files under `data/` for testing.

## Quick usage

1. Build vector store from your documents (first time):

	- Option A: run the `main.py` example which contains code that builds/loads the vectorstore.

```bash
python main.py
```

	- Option B: run a minimal script to build the index (you can adapt this snippet):

```python
from src.Data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

docs = load_all_documents("data")
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)
```

This saves `faiss.index` and `metadata.pkl` under the `faiss_store` (or the folder you set via `PERSIST_DIR`).

2. Query and summarize (example via `main.py`):

```bash
python main.py
# or use the RAGSearch wrapper in code
```

`src/search.py` demonstrates how retrieved chunks are concatenated into context and then passed to the LLM for summarization.

## Example: programmatic query

Use the `RAGSearch` class from `src/search.py` for simple retrieval + summarization:

```python
from src.search import RAGSearch

rag = RAGSearch(persist_dir="faiss_store")
summary = rag.search_and_summarize("What is the HR policy?", top_k=3)
print(summary)
```

Adjust `top_k`, embedding/LLM model names and `persist_dir` as needed.

## Troubleshooting

- FAISS errors (index not found): ensure you have built the vectorstore first — `faiss.index` and `metadata.pkl` must exist in the configured `persist_dir`.
- Model download / memory issues: larger embedding models require more RAM. The default `all-MiniLM-L6-v2` is small and suitable for development.
- LLM integration: `src/search.py` uses `langchain_groq.ChatGroq`. You must provide a valid API key and confirm the package name (the code imports `langchain_groq` — ensure the package exists and is installed).
- Missing loaders: `langchain_community.document_loaders` is used; ensure the dependency is installed and compatible with your LangChain version.

If you hit errors, run the example loader to inspect which files are discovered:

```bash
python -c "from src.Data_loader import load_all_documents; docs=load_all_documents('data'); print(len(docs))"
```

## Development notes & next steps

- Add CLI argument parsing to `main.py` to accept queries, `persist_dir`, and rebuild flags.
- Add unit tests for `Data_loader`, `EmbeddingPipeline`, and `FaissVectorStore`.
- Add a small web UI or API endpoint for interactive queries.
- Consider swapping FAISS for Chroma/Weaviate if you want more features like metadata filtering and persistent remote vector DB.

## Contributing

Contributions are welcome. Please open an issue to discuss changes before creating pull requests.


- Implement a CLI wrapper for `main.py` (argparse/typer).
- Add a small unit test or two for the loader/vectorstore.

Tell me which of those you'd like next and I'll implement it.


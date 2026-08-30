# TFG RAG: Naive vs Hybrid Retrieval-Augmented Generation

This repository contains a complete Python project for a TFG that implements, compares, and evaluates two RAG systems:

- **Naive RAG**: LangChain + FAISS dense retrieval
- **Advanced Hybrid RAG**: LangChain + ChromaDB dense retrieval + BM25 lexical retrieval + fusion

## Objective

Provide a production-like but academically clean setup where you only need to:

1. Install dependencies
2. Fill `.env` with your API keys and model names
3. Add documents to `data/raw/`
4. Run ingestion, QA, and evaluation scripts

## Architecture Overview

- **Ingestion**: load `.txt`, `.md`, and `.pdf`; clean text; chunk with overlap; preserve metadata
- **Naive pipeline**: dense embeddings + FAISS + prompt grounding + LLM answer
- **Hybrid pipeline**: ChromaDB dense retrieval + BM25 + weighted/RRF fusion + optional reranker interface
- **Evaluation**: build RAGAS datasets and generate CSV/JSON reports

## Folder Structure

```text
repo/
  README.md
  requirements.txt
  .env.example
  .gitignore
  pyproject.toml
  config/
    settings.yaml
    prompts.yaml
  data/
    raw/
    processed/
    eval/
  outputs/
    indexes/
    runs/
    reports/
    logs/
  src/
  scripts/
  tests/
```

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill required values:

```bash
cp .env.example .env
```

Required for generation/indexing with OpenAI-compatible LangChain classes:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`

Optional runtime overrides and tracking:

- `CHROMA_PERSIST_DIR`
- `FAISS_INDEX_DIR`
- `DATA_RAW_DIR`
- `DATA_PROCESSED_DIR`
- `TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

## Add Documents

Place source material in:

- `data/raw/*.txt`
- `data/raw/*.md`
- `data/raw/*.pdf`

## Ingestion

```bash
python scripts/ingest_naive.py
python scripts/ingest_hybrid.py
```

## Run QA Pipelines

```bash
python scripts/run_without_rag.py "Your question here"
python scripts/run_naive_rag.py "Your question here"
python scripts/run_hybrid_rag.py "Your question here"
```

The first command is the baseline without retrieval, while the next two show the difference introduced by RAG. Runs are saved as JSON in `outputs/runs/`.

## Evaluation with RAGAS

Prepare a CSV in `data/eval/` with at least:

- `question`
- `ground_truth`

Then run:

```bash
python scripts/evaluate_naive.py outputs/runs/naive_<timestamp>.json data/eval/ground_truth.csv
python scripts/evaluate_hybrid.py outputs/runs/hybrid_<timestamp>.json data/eval/ground_truth.csv
```

Reports are saved to `outputs/reports/` as CSV and JSON.

## Compare Naive vs Hybrid

```bash
python scripts/compare_results.py outputs/reports/naive_eval.csv outputs/reports/hybrid_eval.csv
```

## Testing

```bash
pytest -q
```

## Troubleshooting

- **Missing API key/model**: ensure `.env` is filled correctly.
- **No documents loaded**: verify files exist under `data/raw/`.
- **PDF ingestion error**: install `pypdf`.
- **RAGAS import errors**: confirm compatible `ragas` dependencies.

## Future Work

- Add reranker models (cross-encoder)
- Add experiment tracking dashboards
- Add richer multilingual preprocessing
- Extend support for additional LLM/embedding providers

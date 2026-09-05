from __future__ import annotations

import time
from pathlib import Path

from src.config.loader import load_prompts, load_settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.document_loader import TextChunk, load_documents
from src.ingestion.preprocessing import preprocess_documents
from src.utils.io import dump_json, dump_jsonl_atomic, load_jsonl
from src.utils.logger import get_logger


class BaseRAGPipeline:
    pipeline_name = "base"

    def __init__(self) -> None:
        self.settings = load_settings()
        self.prompts = load_prompts()
        self.logger = get_logger(self.pipeline_name, self.settings.logs_dir)

    def load_chunks(self) -> list[TextChunk]:
        """Build the processed corpus from raw documents (offline ingestion)."""
        self.logger.info("[INGESTION 1/4] Loading source documents from %s", self.settings.raw_data_dir)
        docs = load_documents(self.settings.raw_data_dir)
        self.logger.info("[INGESTION 2/4] Loaded %s source documents; preprocessing text", len(docs))
        preprocessed = preprocess_documents(docs)
        self.logger.info("[INGESTION 3/4] Splitting documents into chunks (size=%s, overlap=%s)", self.settings.chunk_size, self.settings.chunk_overlap)
        chunks = chunk_documents(preprocessed, self.settings.chunk_size, self.settings.chunk_overlap)
        processed_path = Path(self.settings.processed_data_dir) / "chunks.jsonl"
        dump_jsonl_atomic(
            processed_path,
            [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks],
        )
        self.logger.info("[INGESTION 4/4] Wrote %s chunks to %s", len(chunks), processed_path)
        return chunks

    def load_processed_chunks(self) -> list[TextChunk]:
        """Load the immutable processed corpus without touching raw documents."""
        processed_path = Path(self.settings.processed_data_dir) / "chunks.jsonl"
        rows = load_jsonl(processed_path)
        chunks: list[TextChunk] = []
        for row in rows:
            text = row.get("text")
            metadata = row.get("metadata")
            if not isinstance(text, str) or not isinstance(metadata, dict):
                raise ValueError(f"Invalid processed chunk record in {processed_path}")
            chunks.append(TextChunk(text=text, metadata=dict(metadata)))
        self.logger.info("[QUERY 1/8] Loaded %s processed chunks from %s", len(chunks), processed_path)
        return chunks

    def save_run(self, payload: dict) -> Path:
        runs_dir = Path(self.settings.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        file_path = runs_dir / f"{self.pipeline_name}_{int(time.time())}.json"
        dump_json(file_path, payload)
        return file_path

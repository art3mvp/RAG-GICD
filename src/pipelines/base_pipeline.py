from __future__ import annotations

import time
from pathlib import Path

from src.config.loader import load_prompts, load_settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.document_loader import TextChunk, load_documents
from src.ingestion.preprocessing import preprocess_documents
from src.utils.io import dump_json, dump_jsonl
from src.utils.logger import get_logger


class BaseRAGPipeline:
    pipeline_name = "base"

    def __init__(self) -> None:
        self.settings = load_settings()
        self.prompts = load_prompts()
        self.logger = get_logger(self.pipeline_name, self.settings.logs_dir)

    def load_chunks(self) -> list[TextChunk]:
        docs = load_documents(self.settings.raw_data_dir)
        preprocessed = preprocess_documents(docs)
        chunks = chunk_documents(preprocessed, self.settings.chunk_size, self.settings.chunk_overlap)
        processed_path = Path(self.settings.processed_data_dir) / "chunks.jsonl"
        dump_jsonl(
            processed_path,
            [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks],
        )
        self.logger.info("Loaded %s raw docs and generated %s chunks", len(docs), len(chunks))
        return chunks

    def save_run(self, payload: dict) -> Path:
        runs_dir = Path(self.settings.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        file_path = runs_dir / f"{self.pipeline_name}_{int(time.time())}.json"
        dump_json(file_path, payload)
        return file_path

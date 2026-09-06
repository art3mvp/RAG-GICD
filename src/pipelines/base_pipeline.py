from __future__ import annotations

import time
import uuid
from pathlib import Path

from src.config.loader import load_prompts, load_settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.document_loader import TextChunk, load_documents
from src.ingestion.preprocessing import preprocess_documents
from src.utils.io import dump_json, dump_jsonl_atomic, load_jsonl
from src.utils.logger import get_logger
from src.utils.artifacts import chunks_hash
from src.utils.hashing import stable_hash


class BaseRAGPipeline:
    pipeline_name = "base"

    def __init__(self) -> None:
        self.settings = load_settings()
        self.prompts = load_prompts()
        self.logger = get_logger(self.pipeline_name, self.settings.logs_dir)

    def _start_run(self, operation: str) -> None:
        self._log("Run started | operation=%s", operation)

    def _log(self, message: str, *args) -> None:
        self.logger.info(message, *args)

    def _log_configuration(self) -> None:
        self._log(
            "Configuration | chunking_strategy=%s chunk_size=%s chunk_overlap=%s top_k=%s embedding_model=%s chat_model=%s prompt_version=%s",
            self.settings.chunking_strategy,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.top_k,
            self.settings.openai_embedding_model,
            self.settings.openai_model,
            self.prompt_version,
        )

    @property
    def prompt_version(self) -> str:
        prompt_text = "\n".join(self.prompts.get(key, "") for key in ("system_prompt", "qa_prompt"))
        return stable_hash(prompt_text)[:12]

    @staticmethod
    def build_citations(results) -> list[dict]:
        return [
            {
                "rank": rank,
                "source": item.source,
                "chunk_id": item.metadata.get("chunk_id"),
                "score": item.score,
            }
            for rank, item in enumerate(results, start=1)
        ]

    @staticmethod
    def add_citations(answer: str, citations: list[dict]) -> str:
        if not citations:
            return answer
        references = "; ".join(
            f"[{citation['rank']}] {citation['source']} (chunk_id={citation['chunk_id']}, score={citation['score']:.4f})"
            for citation in citations
        )
        return f"{answer}\n\nFuentes: {references}"

    def load_chunks(self) -> list[TextChunk]:
        """Build the processed corpus from raw documents (offline ingestion)."""
        self._start_run("ingestion")
        self._log(
            "Ingestion configuration | segmentation_strategy=%s chunk_size=%s chunk_overlap=%s",
            self.settings.chunking_strategy,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        started = time.perf_counter()
        self._log("Loading source documents")
        docs = load_documents(self.settings.raw_data_dir)
        self._log("Source documents loaded | document_count=%s", len(docs))
        preprocessed = preprocess_documents(docs)
        self._log(
            "Segmenting documents | strategy=%s chunk_size=%s chunk_overlap=%s",
            self.settings.chunking_strategy,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        chunks = chunk_documents(preprocessed, self.settings.chunk_size, self.settings.chunk_overlap, self.settings.chunking_strategy)
        processed_path = Path(self.settings.processed_data_dir) / "chunks.jsonl"
        dump_jsonl_atomic(
            processed_path,
            [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks],
        )
        self._log(
            "Processed corpus written | chunk_count=%s corpus_hash=%s output=%s duration_ms=%.1f",
            len(chunks),
            chunks_hash(chunks),
            processed_path,
            (time.perf_counter() - started) * 1000,
        )
        return chunks

    def load_processed_chunks(self) -> list[TextChunk]:
        """Load the immutable processed corpus without touching raw documents."""
        processed_path = Path(self.settings.processed_data_dir) / "chunks.jsonl"
        started = time.perf_counter()
        self._log("Loading processed corpus")
        rows = load_jsonl(processed_path)
        chunks: list[TextChunk] = []
        for row in rows:
            text = row.get("text")
            metadata = row.get("metadata")
            if not isinstance(text, str) or not isinstance(metadata, dict):
                raise ValueError(f"Invalid processed chunk record in {processed_path}")
            chunks.append(TextChunk(text=text, metadata=dict(metadata)))
        self._log(
            "Processed corpus loaded | chunk_count=%s duration_ms=%.1f",
            len(chunks),
            (time.perf_counter() - started) * 1000,
        )
        return chunks

    def save_run(self, payload: dict) -> Path:
        runs_dir = Path(self.settings.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        file_path = runs_dir / f"{self.pipeline_name}_{time.time_ns()}_{uuid.uuid4().hex[:8]}.json"
        dump_json(file_path, payload)
        self._log("Run saved | output=%s", file_path)
        return file_path

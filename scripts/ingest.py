from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embeddings.embedding_factory import create_embeddings
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.bm25_retriever import BM25Retriever
from src.stores.chroma_store import ChromaStore
from src.utils.artifacts import (
    BM25_FILENAME,
    DENSE_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    build_dense_manifest,
    build_manifest,
    chunks_hash,
    save_manifest,
)
from src.utils.logger import get_logger


def main() -> None:
    pipeline = BaseRAGPipeline()
    logger = get_logger("ingest", pipeline.settings.logs_dir)
    chunks = pipeline.load_chunks()
    if not chunks:
        print("No documents found in data/raw")
        return

    corpus_version = chunks_hash(chunks)[:12]
    embedding_started = time.perf_counter()
    embeddings = create_embeddings(pipeline.settings)
    model_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    embedding_dimension = getattr(embeddings, "dimensions", None) or model_dimensions.get(
        pipeline.settings.openai_embedding_model,
        "unknown",
    )
    logger.info(
        "Embedding model ready | model=%s dimension=%s duration_ms=%.1f corpus_version=%s",
        pipeline.settings.openai_embedding_model,
        embedding_dimension,
        (time.perf_counter() - embedding_started) * 1000,
        corpus_version,
    )
    store = ChromaStore(pipeline.settings.chroma_persist_dir)
    chroma_started = time.perf_counter()
    store.build(chunks, embeddings)
    logger.info(
        "Chroma index built | vectors_inserted=%s duration_ms=%.1f corpus_version=%s errors=0 retries=0",
        len(chunks),
        (time.perf_counter() - chroma_started) * 1000,
        corpus_version,
    )

    index_dir = Path(pipeline.settings.dense_index_dir)
    bm25_started = time.perf_counter()
    bm25 = BM25Retriever(k1=pipeline.settings.bm25_k1, b=pipeline.settings.bm25_b)
    bm25.build(chunks)
    hybrid_dir = Path(pipeline.settings.hybrid_index_dir)
    bm25.save(hybrid_dir / BM25_FILENAME)
    logger.info(
        "BM25 index built | documents_indexed=%s duration_ms=%.1f corpus_version=%s errors=0 retries=0",
        len(chunks),
        (time.perf_counter() - bm25_started) * 1000,
        corpus_version,
    )
    save_manifest(
        index_dir / DENSE_MANIFEST_FILENAME,
        build_dense_manifest(chunks, pipeline.settings, store.collection_name),
    )
    save_manifest(
        index_dir / MANIFEST_FILENAME,
        build_manifest(chunks, pipeline.settings, store.collection_name),
    )
    logger.info("Ingestion completed | chunk_count=%s indexes=chroma,bm25", len(chunks))


if __name__ == "__main__":
    main()
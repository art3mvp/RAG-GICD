from __future__ import annotations

from pathlib import Path
import sys

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
    save_manifest,
)
from src.utils.logger import get_logger


def main() -> None:
    pipeline = BaseRAGPipeline()
    logger = get_logger("ingest", pipeline.settings.logs_dir)
    logger.info("[INGESTION START] Rebuilding shared RAG artifacts from raw documents")
    chunks = pipeline.load_chunks()
    if not chunks:
        print("No documents found in data/raw")
        return

    embeddings = create_embeddings(pipeline.settings)
    store = ChromaStore(pipeline.settings.chroma_persist_dir)
    store.build(chunks, embeddings)

    index_dir = Path(pipeline.settings.dense_index_dir)
    bm25 = BM25Retriever(k1=pipeline.settings.bm25_k1, b=pipeline.settings.bm25_b)
    bm25.build(chunks)
    hybrid_dir = Path(pipeline.settings.hybrid_index_dir)
    bm25.save(hybrid_dir / BM25_FILENAME)
    save_manifest(
        index_dir / DENSE_MANIFEST_FILENAME,
        build_dense_manifest(chunks, pipeline.settings, store.collection_name),
    )
    save_manifest(
        hybrid_dir / MANIFEST_FILENAME,
        build_manifest(chunks, pipeline.settings, store.collection_name),
    )
    logger.info("[INGESTION COMPLETE] Published %s chunks into shared Chroma + BM25", len(chunks))
    print(f"Indexed {len(chunks)} chunks into shared Chroma and BM25")


if __name__ == "__main__":
    main()
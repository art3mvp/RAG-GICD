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
    MANIFEST_FILENAME,
    build_manifest,
    save_manifest,
)
from src.utils.logger import get_logger


def main() -> None:
    pipeline = BaseRAGPipeline()
    logger = get_logger("ingest_hybrid", pipeline.settings.logs_dir)
    logger.info("[INGESTION START] Rebuilding hybrid indexes from raw documents")
    chunks = pipeline.load_chunks()
    if not chunks:
        print("No documents found in data/raw")
        return

    logger.info("[INDEX 1/4] Creating dense embedding model: %s", pipeline.settings.openai_embedding_model)
    embeddings = create_embeddings(pipeline.settings)
    store = ChromaStore(pipeline.settings.chroma_persist_dir)
    logger.info("[INDEX 2/4] Rebuilding dense Chroma index at %s", store.persist_dir)
    store.build(chunks, embeddings)
    logger.info("[INDEX 2/4] Dense index rebuilt with %s chunks", len(chunks))

    logger.info("[INDEX 3/4] Building lexical inverted BM25 index")
    bm25 = BM25Retriever(k1=pipeline.settings.bm25_k1, b=pipeline.settings.bm25_b)
    bm25.build(chunks)
    index_dir = Path(pipeline.settings.hybrid_index_dir)
    bm25.save(index_dir / BM25_FILENAME)
    logger.info("[INDEX 3/4] BM25 index saved to %s", index_dir / BM25_FILENAME)
    logger.info("[INDEX 4/4] Publishing hybrid manifest")
    manifest = build_manifest(chunks, pipeline.settings, store.collection_name)
    save_manifest(index_dir / MANIFEST_FILENAME, manifest)
    logger.info("[INGESTION COMPLETE] Published %s chunks into dense Chroma + lexical BM25", len(chunks))


if __name__ == "__main__":
    main()

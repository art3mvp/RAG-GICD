from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embeddings.embedding_factory import create_embeddings
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.bm25_retriever import BM25Retriever
from src.stores.chroma_store import ChromaStore


def main() -> None:
    pipeline = BaseRAGPipeline()
    chunks = pipeline.load_chunks()
    if not chunks:
        print("No documents found in data/raw")
        return

    embeddings = create_embeddings(pipeline.settings)
    store = ChromaStore(pipeline.settings.chroma_persist_dir)
    store.build(chunks, embeddings)
    store.save()

    bm25 = BM25Retriever(k1=pipeline.settings.bm25_k1, b=pipeline.settings.bm25_b)
    bm25.build(chunks)
    print(f"Indexed {len(chunks)} chunks into Chroma + BM25")


if __name__ == "__main__":
    main()

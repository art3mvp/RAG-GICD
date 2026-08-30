from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embeddings.embedding_factory import create_embeddings
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.stores.faiss_store import FAISSStore


def main() -> None:
    pipeline = BaseRAGPipeline()
    chunks = pipeline.load_chunks()
    if not chunks:
        print("No documents found in data/raw")
        return

    embeddings = create_embeddings(pipeline.settings)
    store = FAISSStore(pipeline.settings.faiss_index_dir)
    store.build(chunks, embeddings)
    store.save()
    print(f"Indexed {len(chunks)} chunks into FAISS")


if __name__ == "__main__":
    main()

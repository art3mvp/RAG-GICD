from __future__ import annotations

from pathlib import Path

from src.embeddings.embedding_factory import create_embeddings
from src.llm.llm_factory import create_chat_llm
from src.llm.prompt_builder import build_messages
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.dense_retriever import DenseRetriever
from src.stores.chroma_store import ChromaStore
from src.utils.artifacts import DENSE_MANIFEST_FILENAME, load_manifest, validate_dense_manifest


class NaiveRAGPipeline(BaseRAGPipeline):
    pipeline_name = "naive"

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = None
        self.llm = None
        self.store = ChromaStore(self.settings.chroma_persist_dir)

    def prepare_index(self, chunks) -> None:
        manifest_path = Path(self.settings.dense_index_dir) / DENSE_MANIFEST_FILENAME
        manifest = load_manifest(manifest_path)
        validate_dense_manifest(manifest, chunks, self.settings, self.store.collection_name)
        if not self.store.exists():
            raise RuntimeError(f"Chroma index not found: {self.store.persist_dir}")
        self.store.load(self.embeddings)
        self.store.validate(
            expected_count=len(chunks),
            expected_chunk_ids={str(chunk.metadata.get("chunk_id")) for chunk in chunks},
        )
        self.logger.info("Loaded and validated naive Chroma index")

    def run(self, question: str) -> dict:
        chunks = self.load_processed_chunks()
        if not chunks:
            return {"question": question, "answer": "I do not know.", "retrieved": []}

        self.embeddings = create_embeddings(self.settings)
        self.llm = create_chat_llm(self.settings)
        self.prepare_index(chunks)
        retriever = DenseRetriever(self.store, method_name="chroma_dense")
        retrieved = retriever.retrieve(question, self.settings.top_k)
        messages = build_messages(question, retrieved, self.prompts)
        answer = self.llm.invoke(messages).content

        payload = {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "retrieved": [item.__dict__ for item in retrieved],
        }
        output_file = self.save_run(payload)
        self.logger.info("Run saved to %s", output_file)
        return payload

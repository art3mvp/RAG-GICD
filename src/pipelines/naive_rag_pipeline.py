from __future__ import annotations

import time
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
        started = time.perf_counter()
        manifest_path = Path(self.settings.dense_index_dir) / DENSE_MANIFEST_FILENAME
        self._log("Loading dense index manifest")
        manifest = load_manifest(manifest_path)
        validate_dense_manifest(manifest, chunks, self.settings, self.store.collection_name)
        self._log("Dense index manifest validated | chunk_count=%s", len(chunks))
        if not self.store.exists():
            raise RuntimeError(f"Chroma index not found: {self.store.persist_dir}")
        self._log("Loading dense Chroma index")
        self.store.load(self.embeddings)
        self.store.validate(
            expected_count=len(chunks),
            expected_chunk_ids={str(chunk.metadata.get("chunk_id")) for chunk in chunks},
        )
        self._log(
            "Dense Chroma index loaded and validated | chunk_count=%s duration_ms=%.1f",
            len(chunks),
            (time.perf_counter() - started) * 1000,
        )

    def run(self, question: str) -> dict:
        self._start_run("naive_query")
        self._log_configuration()
        chunks = self.load_processed_chunks()
        if not chunks:
            self._log("Processed corpus is empty | returning fallback answer")
            return {"question": question, "answer": "I do not know.", "retrieved": []}

        self.embeddings = create_embeddings(self.settings)
        self.llm = create_chat_llm(self.settings)
        self.prepare_index(chunks)
        retriever = DenseRetriever(self.store, method_name="chroma_dense")
        retrieval_started = time.perf_counter()
        self._log("Retrieving dense candidates | top_k=%s", self.settings.top_k)
        retrieved = retriever.retrieve(question, self.settings.top_k)
        self._log(
            "Dense retrieval completed | result_count=%s duration_ms=%.1f",
            len(retrieved),
            (time.perf_counter() - retrieval_started) * 1000,
        )
        messages = build_messages(question, retrieved, self.prompts)
        generation_started = time.perf_counter()
        self._log("Generating answer | context_chunks=%s", len(retrieved))
        raw_answer = self.llm.invoke(messages).content
        self._log("Answer generated | duration_ms=%.1f", (time.perf_counter() - generation_started) * 1000)
        citations = self.build_citations(retrieved)
        answer = self.add_citations(raw_answer, citations)

        payload = {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "retrieved": [item.__dict__ for item in retrieved],
            "citations": citations,
            "prompt_version": self.prompt_version,
        }
        output_file = self.save_run(payload)
        self._log("Naive query completed | retrieved_count=%s output=%s", len(retrieved), output_file)
        return payload

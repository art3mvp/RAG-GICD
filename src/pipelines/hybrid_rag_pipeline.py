from __future__ import annotations

import time
from pathlib import Path

from src.embeddings.embedding_factory import create_embeddings
from src.llm.llm_factory import create_chat_llm
from src.llm.prompt_builder import build_messages
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.reranker import BaseReranker, get_reranker
from src.stores.chroma_store import ChromaStore
from src.utils.artifacts import (
    BM25_FILENAME,
    MANIFEST_FILENAME,
    load_manifest,
    validate_manifest,
)


class HybridRAGPipeline(BaseRAGPipeline):
    pipeline_name = "hybrid"

    def __init__(self, reranker: BaseReranker | None = None) -> None:
        super().__init__()
        self.embeddings = None
        self.llm = None
        self.store = ChromaStore(self.settings.chroma_persist_dir)
        self.bm25 = None
        self.reranker = reranker or get_reranker(self.settings)

    def prepare_indexes(self, chunks) -> None:
        """Load and validate offline-built artifacts; never build them online."""
        started = time.perf_counter()
        index_dir = Path(self.settings.dense_index_dir)
        self._log("Loading hybrid manifest")
        manifest = load_manifest(index_dir / MANIFEST_FILENAME)
        validate_manifest(manifest, chunks, self.settings, self.store.collection_name)
        self._log("Hybrid manifest validated | corpus_and_index_settings_match=true")
        bm25_path = Path(self.settings.hybrid_index_dir) / manifest["artifacts"].get("bm25", BM25_FILENAME)
        self._log("Loading lexical BM25 index")
        self.bm25 = BM25Retriever.load(bm25_path)
        if self.bm25.parameters != (self.settings.bm25_k1, self.settings.bm25_b):
            raise RuntimeError("BM25 artifact parameters do not match the manifest")
        bm25_chunks = self.bm25.chunks
        if len(bm25_chunks) != len(chunks) or {
            str(chunk.metadata.get("chunk_id")) for chunk in bm25_chunks
        } != {str(chunk.metadata.get("chunk_id")) for chunk in chunks}:
            raise RuntimeError("BM25 artifact does not match the processed chunks")
        if not self.store.exists():
            raise RuntimeError(f"Chroma index not found: {self.store.persist_dir}")
        self._log("Loading dense Chroma index")
        self.store.load(self.embeddings)
        self.store.validate(
            expected_count=len(chunks),
            expected_chunk_ids={str(chunk.metadata.get("chunk_id")) for chunk in chunks},
        )
        self._log(
            "Dense and lexical indexes validated | chunk_count=%s duration_ms=%.1f",
            len(chunks),
            (time.perf_counter() - started) * 1000,
        )

    def run(self, question: str) -> dict:
        self._start_run("hybrid_query")
        self._log_configuration()
        if self.settings.hybrid_fusion_strategy == "weighted":
            self._log(
                "Hybrid retrieval configuration | fusion_strategy=weighted dense_weight=%s bm25_weight=%s bm25_k1=%s bm25_b=%s reranker_enabled=%s reranker_model=%s",
                self.settings.hybrid_dense_weight,
                self.settings.hybrid_bm25_weight,
                self.settings.bm25_k1,
                self.settings.bm25_b,
                self.settings.reranker_enabled,
                self.settings.reranker_model,
            )
        else:
            self._log(
                "Hybrid retrieval configuration | fusion_strategy=rrf formula=1/(rrf_k+rank) rrf_k=%s bm25_k1=%s bm25_b=%s reranker_enabled=%s reranker_model=%s",
                self.settings.rrf_k,
                self.settings.bm25_k1,
                self.settings.bm25_b,
                self.settings.reranker_enabled,
                self.settings.reranker_model,
            )
        chunks = self.load_processed_chunks()
        if not chunks:
            self._log("Processed corpus is empty | returning fallback answer")
            return {"question": question, "answer": "I do not know.", "retrieved": []}

        self.embeddings = create_embeddings(self.settings)
        self.llm = create_chat_llm(self.settings)
        self.prepare_indexes(chunks)
        dense = DenseRetriever(self.store, method_name="chroma_dense")
        if self.bm25 is None:
            raise RuntimeError("BM25 index was not loaded")
        hybrid = HybridRetriever(dense, self.bm25, logger=self.logger)

        # Retrieve candidate pool (initial_k instead of top_k)
        candidate_k = self.settings.initial_k if self.settings.reranker_enabled else self.settings.top_k

        if self.settings.hybrid_fusion_strategy == "weighted":
            self._log("Retrieving and fusing candidates | strategy=weighted dense_weight=%s bm25_weight=%s candidate_k=%s", self.settings.hybrid_dense_weight, self.settings.hybrid_bm25_weight, candidate_k)
            retrieved = hybrid.retrieve_weighted(
                question,
                top_k=candidate_k,
                dense_weight=self.settings.hybrid_dense_weight,
                bm25_weight=self.settings.hybrid_bm25_weight,
            )
        else:
            self._log("Retrieving and fusing candidates | strategy=rrf candidate_k=%s", candidate_k)
            retrieved = hybrid.retrieve_rrf(question, top_k=candidate_k, k_constant=self.settings.rrf_k)

        # Rerank and truncate down to top_k
        self._log("Reranking input prepared | rerank_input_k=%s rerank_top_k=%s", len(retrieved), self.settings.top_k)
        rerank_load_started = time.perf_counter()
        self.reranker.load()
        self._log("Reranker model ready | model=%s load_duration_ms=%.1f", self.settings.reranker_model, (time.perf_counter() - rerank_load_started) * 1000)
        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(question, retrieved, top_k=self.settings.top_k)
        self._log("Reranking completed | final_context_chunks=%s duration_ms=%.1f", len(reranked), (time.perf_counter() - rerank_started) * 1000)
        
        messages = build_messages(question, reranked, self.prompts)
        generation_started = time.perf_counter()
        self._log("Generating answer | context_chunks=%s", len(reranked))
        raw_answer = self.llm.invoke(messages).content
        self._log("Answer generated | duration_ms=%.1f", (time.perf_counter() - generation_started) * 1000)
        citations = self.build_citations(reranked)
        answer = self.add_citations(raw_answer, citations)

        payload = {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "retrieved": [item.__dict__ for item in reranked],
            "citations": citations,
            "prompt_version": self.prompt_version,
        }
        output_file = self.save_run(payload)
        self._log("Hybrid query completed | retrieved_count=%s output=%s", len(reranked), output_file)
        return payload
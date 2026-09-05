from __future__ import annotations

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
        index_dir = Path(self.settings.hybrid_index_dir)
        self.logger.info("[QUERY 3/9] Loading hybrid manifest from %s", index_dir / MANIFEST_FILENAME)
        manifest = load_manifest(index_dir / MANIFEST_FILENAME)
        validate_manifest(manifest, chunks, self.settings, self.store.collection_name)
        self.logger.info("[QUERY 4/9] Manifest validated: corpus, chunking, embedding and BM25 settings match")
        bm25_path = index_dir / manifest["artifacts"].get("bm25", BM25_FILENAME)
        self.logger.info("[QUERY 5/9] Loading lexical inverted BM25 index from %s", bm25_path)
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
        self.logger.info("[QUERY 6/9] Loading dense Chroma vector index from %s", self.store.persist_dir)
        self.store.load(self.embeddings)
        self.store.validate(
            expected_count=len(chunks),
            expected_chunk_ids={str(chunk.metadata.get("chunk_id")) for chunk in chunks},
        )
        self.logger.info("[QUERY 7/9] Dense and lexical indexes validated successfully (%s chunks)", len(chunks))

    def run(self, question: str) -> dict:
        self.logger.info("[QUERY START] Received question: %s", question)
        chunks = self.load_processed_chunks()
        if not chunks:
            return {"question": question, "answer": "I do not know.", "retrieved": []}

        self.logger.info("[QUERY 2/9] Initializing embedding model for dense retrieval: %s", self.settings.openai_embedding_model)
        self.embeddings = create_embeddings(self.settings)
        self.llm = create_chat_llm(self.settings)
        self.prepare_indexes(chunks)
        dense = DenseRetriever(self.store, method_name="chroma_dense")
        if self.bm25 is None:
            raise RuntimeError("BM25 index was not loaded")
        hybrid = HybridRetriever(dense, self.bm25)

        # Retrieve candidate pool (initial_k instead of top_k)
        candidate_k = self.settings.initial_k if self.settings.reranker_enabled else self.settings.top_k

        if self.settings.hybrid_fusion_strategy == "weighted":
            self.logger.info("[QUERY 8/9] Retrieving candidates with dense similarity + BM25, then weighted fusion (dense=%s, bm25=%s)", self.settings.hybrid_dense_weight, self.settings.hybrid_bm25_weight)
            retrieved = hybrid.retrieve_weighted(
                question,
                top_k=candidate_k,
                dense_weight=self.settings.hybrid_dense_weight,
                bm25_weight=self.settings.hybrid_bm25_weight,
            )
        else:
            self.logger.info("[QUERY 8/9] Retrieving candidates with dense similarity + BM25, then reciprocal-rank fusion (top_k=%s)", candidate_k)
            retrieved = hybrid.retrieve_rrf(question, top_k=candidate_k)

        # Rerank and truncate down to top_k
        self.logger.info("[QUERY 8/9] Fusion returned %s candidates; reranking to top_k=%s", len(retrieved), self.settings.top_k)
        reranked = self.reranker.rerank(question, retrieved, top_k=self.settings.top_k)
        self.logger.info("[QUERY 9/9] Reranking completed: %s final context chunks; generating answer", len(reranked))
        
        messages = build_messages(question, reranked, self.prompts)
        answer = self.llm.invoke(messages).content

        payload = {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "retrieved": [item.__dict__ for item in reranked],
        }
        output_file = self.save_run(payload)
        self.logger.info("Run saved to %s", output_file)
        self.logger.info("[QUERY COMPLETE] Answer generated and run saved")
        return payload
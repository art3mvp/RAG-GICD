from __future__ import annotations

from src.embeddings.embedding_factory import create_embeddings
from src.llm.llm_factory import create_chat_llm
from src.llm.prompt_builder import build_messages
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.reranker import get_reranker
from src.stores.chroma_store import ChromaStore


class HybridRAGPipeline(BaseRAGPipeline):
    pipeline_name = "hybrid"

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = None
        self.llm = None
        self.store = ChromaStore(self.settings.chroma_persist_dir)
        self.bm25 = BM25Retriever(k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        self.reranker = get_reranker(self.settings)

    def prepare_indexes(self, chunks) -> None:
        if self.store.exists():
            self.store.load(self.embeddings)
            self.logger.info("Loaded existing Chroma index")
        else:
            self.store.build(chunks, self.embeddings)
            self.store.save()
            self.logger.info("Built and saved Chroma index")
        self.bm25.build(chunks)
        self.logger.info("Built BM25 index")

    def run(self, question: str) -> dict:
        chunks = self.load_chunks()
        if not chunks:
            return {"question": question, "answer": "I do not know.", "retrieved": []}

        self.embeddings = create_embeddings(self.settings)
        self.llm = create_chat_llm(self.settings)
        self.prepare_indexes(chunks)
        dense = DenseRetriever(self.store, method_name="chroma_dense")
        hybrid = HybridRetriever(dense, self.bm25)

        # Retrieve candidate pool (initial_k instead of top_k)
        candidate_k = self.settings.initial_k if self.settings.reranker_enabled else self.settings.top_k

        if self.settings.hybrid_fusion_strategy == "weighted":
            retrieved = hybrid.retrieve_weighted(
                question,
                top_k=candidate_k,
                dense_weight=self.settings.hybrid_dense_weight,
                bm25_weight=self.settings.hybrid_bm25_weight,
            )
        else:
            retrieved = hybrid.retrieve_rrf(question, top_k=candidate_k)

        # Rerank and truncate down to top_k
        reranked = self.reranker.rerank(question, retrieved, top_k=self.settings.top_k)
        
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
        return payload
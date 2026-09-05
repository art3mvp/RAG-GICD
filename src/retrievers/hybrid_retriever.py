from __future__ import annotations

from collections import defaultdict
import logging

from src.retrievers.dense_retriever import RetrievalResult


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.logger = logging.getLogger("hybrid")

    @staticmethod
    def _key(item: RetrievalResult) -> str:
        return str(item.metadata.get("chunk_id") or f"{item.source}:{item.metadata.get('chunk_index', 0)}")

    def retrieve_weighted(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        bm25_weight: float,
    ) -> list[RetrievalResult]:
        dense = self.dense_retriever.retrieve(query, top_k=top_k)
        self.logger.info("[RETRIEVAL] Dense retrieval completed: %s candidates from Chroma", len(dense))
        lexical = self.bm25_retriever.retrieve(query, top_k=top_k)
        self.logger.info("[RETRIEVAL] Lexical retrieval completed: %s candidates from inverted BM25 index", len(lexical))

        dense_scores = [item.score for item in dense] or [1.0]
        bm25_scores = [item.score for item in lexical] or [1.0]

        max_dense = max(dense_scores) if max(dense_scores) != 0 else 1.0
        max_bm25 = max(bm25_scores) if max(bm25_scores) != 0 else 1.0

        fused: dict[str, RetrievalResult] = {}
        merged_scores = defaultdict(float)

        for item in dense:
            key = self._key(item)
            merged_scores[key] += dense_weight * (item.score / max_dense)
            fused[key] = item

        for item in lexical:
            key = self._key(item)
            merged_scores[key] += bm25_weight * (item.score / max_bm25)
            if key not in fused:
                fused[key] = item

        ranked_keys = sorted(merged_scores, key=lambda key: merged_scores[key], reverse=True)[:top_k]
        return [
            RetrievalResult(
                text=fused[key].text,
                source=fused[key].source,
                score=float(merged_scores[key]),
                retrieval_method="hybrid_weighted",
                metadata=fused[key].metadata,
            )
            for key in ranked_keys
        ]

    def retrieve_rrf(self, query: str, top_k: int, k_constant: int = 60) -> list[RetrievalResult]:
        dense = self.dense_retriever.retrieve(query, top_k=top_k)
        self.logger.info("[RETRIEVAL] Dense retrieval completed: %s candidates from Chroma", len(dense))
        lexical = self.bm25_retriever.retrieve(query, top_k=top_k)
        self.logger.info("[RETRIEVAL] Lexical retrieval completed: %s candidates from inverted BM25 index", len(lexical))

        fused: dict[str, RetrievalResult] = {}
        scores = defaultdict(float)

        for rank, item in enumerate(dense, start=1):
            key = self._key(item)
            scores[key] += 1.0 / (k_constant + rank)
            fused[key] = item

        for rank, item in enumerate(lexical, start=1):
            key = self._key(item)
            scores[key] += 1.0 / (k_constant + rank)
            if key not in fused:
                fused[key] = item

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k]
        return [
            RetrievalResult(
                text=fused[key].text,
                source=fused[key].source,
                score=float(scores[key]),
                retrieval_method="hybrid_rrf",
                metadata=fused[key].metadata,
            )
            for key in ranked_keys
        ]

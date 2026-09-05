from dataclasses import dataclass

from src.retrievers.dense_retriever import RetrievalResult
from src.retrievers.hybrid_retriever import HybridRetriever


@dataclass
class StubRetriever:
    results: list[RetrievalResult]

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self.results[:top_k]


def test_rrf_fusion_deduplicates_by_chunk_id() -> None:
    dense = StubRetriever(
        [
            RetrievalResult("A", "s1", 0.9, "dense", {"chunk_id": "c1"}),
            RetrievalResult("B", "s2", 0.8, "dense", {"chunk_id": "c2"}),
        ]
    )
    bm25 = StubRetriever(
        [
            RetrievalResult("A", "s1", 2.0, "bm25", {"chunk_id": "c1"}),
            RetrievalResult("C", "s3", 1.5, "bm25", {"chunk_id": "c3"}),
        ]
    )

    hybrid = HybridRetriever(dense, bm25)
    results = hybrid.retrieve_rrf("query", top_k=3)

    assert len(results) == 3
    assert len({r.metadata["chunk_id"] for r in results}) == 3


def test_weighted_fusion_prefers_higher_is_better_dense_scores() -> None:
    dense = StubRetriever(
        [
            RetrievalResult("near", "s1", 0.9, "dense", {"chunk_id": "near"}),
            RetrievalResult("far", "s2", 0.2, "dense", {"chunk_id": "far"}),
        ]
    )
    bm25 = StubRetriever([])

    results = HybridRetriever(dense, bm25).retrieve_weighted(
        "query", top_k=2, dense_weight=1.0, bm25_weight=0.0
    )
    assert results[0].metadata["chunk_id"] == "near"

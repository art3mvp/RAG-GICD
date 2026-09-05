from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    text: str
    source: str
    score: float
    retrieval_method: str
    metadata: dict[str, Any]


class DenseRetriever:
    def __init__(self, store, method_name: str = "dense") -> None:
        self.store = store
        self.method_name = method_name

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []

        if hasattr(self.store, "similarity_search_with_score"):
            docs = self.store.similarity_search_with_score(query, k=top_k)
            for doc, score in docs:
                metadata = dict(doc.metadata)
                distance = max(float(score), 0.0)
                results.append(
                    RetrievalResult(
                        text=doc.page_content,
                        source=str(metadata.get("source", "unknown")),
                        score=1.0 / (1.0 + distance),
                        retrieval_method=self.method_name,
                        metadata=metadata,
                    )
                )
            return results

        docs = self.store.similarity_search_with_relevance_scores(query, k=top_k)
        for doc, score in docs:
            metadata = dict(doc.metadata)
            results.append(
                RetrievalResult(
                    text=doc.page_content,
                    source=str(metadata.get("source", "unknown")),
                    score=float(score),
                    retrieval_method=self.method_name,
                    metadata=metadata,
                )
            )
        return results

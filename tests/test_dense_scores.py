from dataclasses import dataclass

from src.retrievers.dense_retriever import DenseRetriever


@dataclass
class Document:
    page_content: str
    metadata: dict


class DistanceStore:
    def similarity_search_with_score(self, query: str, k: int):
        return [
            (Document("near", {"chunk_id": "near"}), 0.1),
            (Document("far", {"chunk_id": "far"}), 0.9),
        ][:k]


def test_distance_scores_become_higher_is_better() -> None:
    results = DenseRetriever(DistanceStore()).retrieve("q", top_k=2)
    assert results[0].score > results[1].score

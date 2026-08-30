from src.pipelines.hybrid_rag_pipeline import HybridRAGPipeline
from src.pipelines.naive_rag_pipeline import NaiveRAGPipeline


def test_naive_pipeline_handles_empty_corpus(monkeypatch) -> None:
    pipeline = NaiveRAGPipeline()
    monkeypatch.setattr(pipeline, "load_chunks", lambda: [])
    result = pipeline.run("What is this?")
    assert result["answer"] == "I do not know."


def test_hybrid_pipeline_handles_empty_corpus(monkeypatch) -> None:
    pipeline = HybridRAGPipeline()
    monkeypatch.setattr(pipeline, "load_chunks", lambda: [])
    result = pipeline.run("What is this?")
    assert result["answer"] == "I do not know."

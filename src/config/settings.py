from __future__ import annotations

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    project_name: str = "RAG-GICD"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    eval_data_dir: str = "data/eval"
    outputs_dir: str = "outputs"
    faiss_index_dir: str = "outputs/indexes/faiss"
    chroma_persist_dir: str = "outputs/indexes/chroma"
    runs_dir: str = "outputs/runs"
    reports_dir: str = "outputs/reports"
    logs_dir: str = "outputs/logs"
    top_k: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=600, ge=50)
    chunk_overlap: int = Field(default=120, ge=0)
    hybrid_fusion_strategy: str = "rrf"
    hybrid_dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    hybrid_bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_enabled: bool = False

    openai_api_key: str = ""  
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False

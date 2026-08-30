from __future__ import annotations

from src.config.settings import AppSettings


def create_embeddings(settings: AppSettings):
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings")

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError("langchain-openai is required for embedding creation") from exc

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

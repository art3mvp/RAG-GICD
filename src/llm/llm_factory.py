from __future__ import annotations

from src.config.settings import AppSettings


def create_chat_llm(settings: AppSettings):
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for LLM generation")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain-openai is required") from exc

    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)

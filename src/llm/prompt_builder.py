from __future__ import annotations

from src.retrievers.dense_retriever import RetrievalResult


def build_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for idx, item in enumerate(results, start=1):
        blocks.append(f"[{idx}] Source: {item.source}\n{item.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[RetrievalResult], prompts: dict[str, str]) -> list[tuple[str, str]]:
    context = build_context(results)
    system_prompt = prompts.get("system_prompt", "")
    user_prompt = prompts.get("qa_prompt", "{question}\n\n{context}").format(
        question=question,
        context=context,
    )
    return [("system", system_prompt), ("human", user_prompt)]

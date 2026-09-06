from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.loader import load_settings
from src.utils.io import dump_json


def _normalize_for_ragas_dataset(dataset: pd.DataFrame | list[dict[str, Any]]) -> Any:
    return _normalize_dataset(dataset, include_reference=True)


def _normalize_reference_free_dataset(dataset: pd.DataFrame | list[dict[str, Any]]) -> Any:
    return _normalize_dataset(dataset, include_reference=False)


def _normalize_dataset(
    dataset: pd.DataFrame | list[dict[str, Any]],
    include_reference: bool,
) -> Any:
    if hasattr(dataset, "to_dict"):
        rows = dataset.to_dict(orient="records")
    elif isinstance(dataset, list):
        rows = dataset
    else:
        raise TypeError("RAGAS evaluation expects a pandas DataFrame or list of dictionaries.")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        user_input = str(row.get("question") or "")
        response = str(row.get("answer") or "")
        retrieved_contexts = row.get("retrieved_contexts")
        if retrieved_contexts is None:
            retrieved_contexts = row.get("contexts") or []
        if isinstance(retrieved_contexts, str):
            retrieved_contexts = [retrieved_contexts]
        if not isinstance(retrieved_contexts, list):
            retrieved_contexts = list(retrieved_contexts or [])

        normalized_row = {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": [str(item) for item in retrieved_contexts],
        }
        if include_reference:
            normalized_row["reference"] = str(row.get("ground_truth") or row.get("reference") or "")
        normalized_rows.append(normalized_row)

    from ragas import EvaluationDataset

    return EvaluationDataset.from_list(normalized_rows)


def evaluate_with_ragas(dataset: pd.DataFrame, output_prefix: str | Path) -> dict:
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError("ragas, langchain-openai, and OpenAI dependencies are required for evaluation") from exc

    settings = load_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for RAGAS evaluation")

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

    ragas_dataset = _normalize_for_ragas_dataset(dataset)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        ragas_dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    result_df = result.to_pandas()
    summary = result_df.mean(numeric_only=True).to_dict()

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_prefix.with_suffix(".csv"), index=False)
    dump_json(output_prefix.with_suffix(".json"), {"summary": summary, "rows": result_df.to_dict(orient="records")})
    return {"summary": summary, "rows": result_df.to_dict(orient="records")}


def evaluate_reference_free_with_ragas(dataset: pd.DataFrame, output_prefix: str | Path) -> dict:
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        raise RuntimeError("ragas and langchain-openai are required for evaluation") from exc

    settings = load_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for RAGAS evaluation")

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    ragas_dataset = _normalize_reference_free_dataset(dataset)
    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    result_df = result.to_pandas()
    summary = result_df.mean(numeric_only=True).to_dict()

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_prefix.with_suffix(".csv"), index=False)
    dump_json(output_prefix.with_suffix(".json"), {"summary": summary, "rows": result_df.to_dict(orient="records")})
    return {"summary": summary, "rows": result_df.to_dict(orient="records")}

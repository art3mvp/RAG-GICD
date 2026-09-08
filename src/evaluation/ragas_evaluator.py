from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.loader import load_settings
from src.utils.io import dump_json
from src.utils.logger import get_logger


class _RagasGenerationNoticeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        match = re.match(
            r"LLM returned (\d+) generations instead of requested (\d+)\.",
            message,
        )
        if match:
            return False
        return True


def _configure_ragas_logging(logs_dir: str | Path) -> logging.Logger:
    logger = get_logger("evaluation", logs_dir)
    ragas_logger = logging.getLogger("ragas.prompt.pydantic_prompt")
    if not any(isinstance(item, _RagasGenerationNoticeFilter) for item in ragas_logger.filters):
        ragas_logger.addFilter(_RagasGenerationNoticeFilter())
    if not ragas_logger.handlers:
        for handler in logger.handlers:
            ragas_logger.addHandler(handler)
        ragas_logger.propagate = False
    return logger


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
            "retrieved_contexts": [
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in retrieved_contexts
            ],
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
    logger = _configure_ragas_logging(settings.logs_dir)
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
    logger.info("RAGAS evaluation started | rows=%s metrics=%s", len(dataset), len(metrics))
    result = evaluate(
        ragas_dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        show_progress=False,
    )
    result_df = result.to_pandas()
    summary = result_df.mean(numeric_only=True).to_dict()

    source_rows = dataset.to_dict(orient="records")
    provenance_by_question = {
        str(row.get("question") or ""): row.get("retrieved_contexts") or []
        for row in source_rows
    }
    report_rows = result_df.to_dict(orient="records")
    for row in report_rows:
        question = str(row.get("user_input") or "")
        if question in provenance_by_question:
            row["retrieved_contexts"] = provenance_by_question[question]

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_prefix.with_suffix(".json"), {"summary": summary, "rows": report_rows})
    logger.info("RAGAS evaluation completed | rows=%s output=%s", len(report_rows), output_prefix.with_suffix(".json"))
    return {"summary": summary, "rows": report_rows}


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
    logger = _configure_ragas_logging(settings.logs_dir)
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
    logger.info("Reference-free RAGAS evaluation started | rows=%s metrics=%s", len(dataset), 2)
    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        show_progress=False,
    )
    result_df = result.to_pandas()
    summary = result_df.mean(numeric_only=True).to_dict()

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_prefix.with_suffix(".json"), {"summary": summary, "rows": result_df.to_dict(orient="records")})
    logger.info("Reference-free RAGAS evaluation completed | rows=%s output=%s", len(result_df), output_prefix.with_suffix(".json"))
    return {"summary": summary, "rows": result_df.to_dict(orient="records")}

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Type

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.ingest import main as ingest_main
from src.evaluation.dataset_builder import build_eval_dataset
from src.evaluation.metrics_report import compare_reports
from src.evaluation.ragas_evaluator import evaluate_with_ragas
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.pipelines.hybrid_rag_pipeline import HybridRAGPipeline
from src.pipelines.naive_rag_pipeline import NaiveRAGPipeline
from src.utils.io import dump_json


async def _run_question(
    pipeline_class: Type[BaseRAGPipeline],
    question: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        return await asyncio.to_thread(pipeline_class().run, question)


async def _run_questions(
    questions: list[str],
    concurrency: int,
) -> tuple[list[dict], list[dict]]:
    semaphore = asyncio.Semaphore(concurrency)
    naive_tasks = [_run_question(NaiveRAGPipeline, question, semaphore) for question in questions]
    hybrid_tasks = [_run_question(HybridRAGPipeline, question, semaphore) for question in questions]
    naive_results, hybrid_results = await asyncio.gather(
        asyncio.gather(*naive_tasks),
        asyncio.gather(*hybrid_tasks),
    )
    for pipeline_name, results in (("naive", naive_results), ("hybrid", hybrid_results)):
        returned_questions = [str(result.get("question", "")) for result in results]
        if returned_questions != questions:
            raise ValueError(f"{pipeline_name} results do not match ground-truth question order")
    return naive_results, hybrid_results


def _read_questions(ground_truth: str | Path) -> list[str]:
    frame = pd.read_csv(ground_truth)
    if "question" not in frame or "ground_truth" not in frame:
        raise ValueError("Ground-truth CSV must contain question and ground_truth columns")
    questions = [str(question) for question in frame["question"].tolist()]
    if not questions or any(not question.strip() for question in questions):
        raise ValueError("Ground-truth CSV must contain non-empty questions")
    if len(set(questions)) != len(questions):
        raise ValueError("Ground-truth CSV contains duplicate questions")
    return questions


def _write_batch(path: str | Path, pipeline: str, results: list[dict]) -> None:
    dump_json(path, {"pipeline": pipeline, "results": results})


async def run_full_flow(
    ground_truth: str | Path,
    concurrency: int = 4,
    naive_run: str | Path = "outputs/runs/naive_batch.json",
    hybrid_run: str | Path = "outputs/runs/hybrid_batch.json",
    naive_report: str | Path = "outputs/reports/naive_eval",
    hybrid_report: str | Path = "outputs/reports/hybrid_eval",
    comparison: str | Path = "outputs/reports/comparison.json",
) -> dict:
    if concurrency < 1:
        raise ValueError("Concurrency must be at least 1")

    questions = _read_questions(ground_truth)
    ingest_main()
    naive_results, hybrid_results = await _run_questions(questions, concurrency)
    _write_batch(naive_run, "naive", naive_results)
    _write_batch(hybrid_run, "hybrid", hybrid_results)

    naive_dataset = build_eval_dataset(naive_run, ground_truth)
    hybrid_dataset = build_eval_dataset(hybrid_run, ground_truth)
    evaluate_with_ragas(naive_dataset, naive_report)
    evaluate_with_ragas(hybrid_dataset, hybrid_report)
    return compare_reports(
        Path(naive_report).with_suffix(".json"),
        Path(hybrid_report).with_suffix(".json"),
        comparison,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, run, evaluate and compare both RAG pipelines")
    parser.add_argument("ground_truth", nargs="?", default="data/eval/ground_truth.csv")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--naive-run", default="outputs/runs/naive_batch.json")
    parser.add_argument("--hybrid-run", default="outputs/runs/hybrid_batch.json")
    parser.add_argument("--naive-report", default="outputs/reports/naive_eval")
    parser.add_argument("--hybrid-report", default="outputs/reports/hybrid_eval")
    parser.add_argument("--comparison", default="outputs/reports/comparison.json")
    args = parser.parse_args()

    report = asyncio.run(
        run_full_flow(
            args.ground_truth,
            concurrency=args.concurrency,
            naive_run=args.naive_run,
            hybrid_run=args.hybrid_run,
            naive_report=args.naive_report,
            hybrid_report=args.hybrid_report,
            comparison=args.comparison,
        )
    )
    print(report["overall"])


if __name__ == "__main__":
    main()
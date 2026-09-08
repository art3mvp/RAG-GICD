from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_json


def build_eval_dataset(run_file: str | Path, ground_truth_file: str | Path) -> pd.DataFrame:
    run_data = load_json(run_file)
    gt_df = pd.read_csv(ground_truth_file)

    if "question" not in gt_df or "ground_truth" not in gt_df:
        raise ValueError("Ground-truth CSV must contain question and ground_truth columns")

    payloads = run_data.get("results") if isinstance(run_data, dict) else None
    if payloads is None:
        payloads = [run_data]
    if not isinstance(payloads, list) or not all(isinstance(payload, dict) for payload in payloads):
        raise ValueError("Run file must contain a payload or a results list")

    ground_truth_by_question = {
        str(row.question): str(row.ground_truth)
        for row in gt_df.itertuples(index=False)
    }
    questions = [str(payload.get("question", "")) for payload in payloads]
    if len(set(questions)) != len(questions):
        raise ValueError("Run file contains duplicate questions")

    missing = [question for question in questions if question not in ground_truth_by_question]
    if missing:
        raise ValueError(f"Questions missing from ground truth: {missing}")

    rows = []
    for payload, question in zip(payloads, questions):
        retrieved = payload.get("retrieved", [])
        context_records = [item for item in retrieved if isinstance(item, dict)]
        contexts = [item.get("text", "") for item in context_records]
        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth_by_question[question],
                "answer": payload.get("answer", ""),
                "contexts": contexts,
                "retrieved_contexts": context_records,
            }
        )

    return pd.DataFrame(rows)

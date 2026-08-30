from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_json


def build_eval_dataset(run_file: str | Path, ground_truth_file: str | Path) -> pd.DataFrame:
    run_data = load_json(run_file)
    gt_df = pd.read_csv(ground_truth_file)

    question = run_data.get("question", "")
    answer = run_data.get("answer", "")
    contexts = [item.get("text", "") for item in run_data.get("retrieved", [])]

    row = gt_df.loc[gt_df["question"] == question]
    if row.empty:
        ground_truth = ""
    else:
        ground_truth = str(row.iloc[0]["ground_truth"])

    return pd.DataFrame(
        [
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
                "retrieved_contexts": contexts,
            }
        ]
    )

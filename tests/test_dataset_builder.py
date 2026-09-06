import json

import pandas as pd

from src.evaluation.dataset_builder import build_eval_dataset


def test_build_eval_dataset_preserves_batch_question_alignment(tmp_path) -> None:
    run_file = tmp_path / "batch.json"
    ground_truth_file = tmp_path / "ground_truth.csv"
    run_file.write_text(
        json.dumps(
            {
                "pipeline": "naive",
                "results": [
                    {
                        "question": "first",
                        "answer": "answer 1",
                        "retrieved": [{"text": "context 1"}],
                    },
                    {
                        "question": "second",
                        "answer": "answer 2",
                        "retrieved": [{"text": "context 2"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"question": "first", "ground_truth": "reference 1"},
            {"question": "second", "ground_truth": "reference 2"},
        ]
    ).to_csv(ground_truth_file, index=False)

    dataset = build_eval_dataset(run_file, ground_truth_file)

    assert dataset["question"].tolist() == ["first", "second"]
    assert dataset["ground_truth"].tolist() == ["reference 1", "reference 2"]
    assert dataset["contexts"].tolist() == [["context 1"], ["context 2"]]
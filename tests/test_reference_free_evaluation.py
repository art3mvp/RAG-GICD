import pandas as pd

from src.evaluation.ragas_evaluator import _normalize_reference_free_dataset


def test_reference_free_dataset_has_no_reference_field() -> None:
    dataset = pd.DataFrame(
        [
            {
                "question": "What is AI?",
                "answer": "AI is a field of computer science.",
                "contexts": ["AI helps automate tasks."],
                "ground_truth": "This must not be used.",
            }
        ]
    )

    sample = _normalize_reference_free_dataset(dataset).to_list()[0]

    assert sample == {
        "user_input": "What is AI?",
        "response": "AI is a field of computer science.",
        "retrieved_contexts": ["AI helps automate tasks."],
    }
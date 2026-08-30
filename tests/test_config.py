from pathlib import Path

import pandas as pd

from src.config.loader import load_settings
from src.evaluation.ragas_evaluator import _normalize_for_ragas_dataset
from src.utils.io import load_json


def test_load_settings_uses_yaml_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("top_k: 7\nchunk_size: 300\n", encoding="utf-8")

    settings = load_settings(cfg)
    assert settings.top_k == 7
    assert settings.chunk_size == 300


def test_normalize_for_ragas_dataset_uses_ragas_schema() -> None:
    dataset = pd.DataFrame(
        [{
            "question": "What is AI?",
            "ground_truth": "A field of computer science.",
            "answer": "AI is a field of computer science.",
            "contexts": ["AI helps automate tasks."],
        }]
    )

    ragas_dataset = _normalize_for_ragas_dataset(dataset)
    sample = ragas_dataset.to_list()[0]

    assert sample["user_input"] == "What is AI?"
    assert sample["response"] == "AI is a field of computer science."
    assert sample["retrieved_contexts"] == ["AI helps automate tasks."]
    assert sample["reference"] == "A field of computer science."


def test_load_json_accepts_windows_style_backslash_paths(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "sample.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"answer": "ok"}', encoding="utf-8")

    path_with_backslashes = str(target).replace("/", "\\")
    assert load_json(path_with_backslashes) == {"answer": "ok"}

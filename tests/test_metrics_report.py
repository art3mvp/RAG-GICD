import json

from src.evaluation.metrics_report import compare_reports


def test_compare_reports_reads_json_and_writes_means(tmp_path) -> None:
    naive_path = tmp_path / "naive.json"
    hybrid_path = tmp_path / "hybrid.json"
    output_path = tmp_path / "comparison.json"

    naive_path.write_text(
        json.dumps({"summary": {"faithfulness": 0.8, "answer_relevancy": 0.6}}),
        encoding="utf-8",
    )
    hybrid_path.write_text(
        json.dumps({"summary": {"faithfulness": 0.7, "answer_relevancy": 0.9}}),
        encoding="utf-8",
    )

    report = compare_reports(naive_path, hybrid_path, output_path)

    assert report["means"] == {
        "answer_relevancy": {"naive": 0.6, "hybrid": 0.9},
        "faithfulness": {"naive": 0.8, "hybrid": 0.7},
    }
    assert report["winner_by_metric"] == {
        "answer_relevancy": "hybrid",
        "faithfulness": "naive",
    }
    assert report["overall"] == {
        "naive_mean": 0.7,
        "hybrid_mean": 0.8,
        "winner": "hybrid",
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
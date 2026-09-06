from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import dump_json, load_json


def _summary_from_report(path: str | Path) -> dict[str, float]:
    report: dict[str, Any] = load_json(path)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Report {path} does not contain a valid summary")

    return {
        metric: float(value)
        for metric, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def compare_reports(
    naive_json: str | Path,
    hybrid_json: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    naive = _summary_from_report(naive_json)
    hybrid = _summary_from_report(hybrid_json)

    metrics = sorted(set(naive) | set(hybrid))
    means = {
        metric: {
            "naive": naive.get(metric),
            "hybrid": hybrid.get(metric),
        }
        for metric in metrics
    }
    winner = {
        metric: (
            "naive"
            if values["naive"] is not None
            and (values["hybrid"] is None or values["naive"] > values["hybrid"])
            else "hybrid"
            if values["hybrid"] is not None
            and (values["naive"] is None or values["hybrid"] > values["naive"])
            else "tie"
        )
        for metric, values in means.items()
    }

    report = {
        "means": means,
        "winner_by_metric": winner,
        "overall": {
            "naive_mean": _mean(naive.values()),
            "hybrid_mean": _mean(hybrid.values()),
        },
    }

    report["overall"]["winner"] = (
        "naive"
        if report["overall"]["naive_mean"] > report["overall"]["hybrid_mean"]
        else "hybrid"
        if report["overall"]["hybrid_mean"] > report["overall"]["naive_mean"]
        else "tie"
    )

    dump_json(output_json, report)
    return report


def _mean(values: Any) -> float:
    numeric_values = list(values)
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics_report import compare_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive vs hybrid RAGAS metrics")
    parser.add_argument("naive_json", nargs="?", default="outputs/reports/naive_eval.json")
    parser.add_argument("hybrid_json", nargs="?", default="outputs/reports/hybrid_eval.json")
    parser.add_argument("--output", default="outputs/reports/comparison.json")
    args = parser.parse_args()

    report = compare_reports(args.naive_json, args.hybrid_json, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

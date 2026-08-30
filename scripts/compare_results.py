from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics_report import compare_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive vs hybrid RAGAS metrics")
    parser.add_argument("naive_csv", nargs="?", default="outputs/reports/naive_eval.csv")
    parser.add_argument("hybrid_csv", nargs="?", default="outputs/reports/hybrid_eval.csv")
    parser.add_argument("--output", default="outputs/reports/comparison.csv")
    args = parser.parse_args()

    report = compare_reports(args.naive_csv, args.hybrid_csv, args.output)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

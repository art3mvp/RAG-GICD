from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.dataset_builder import build_eval_dataset
from src.evaluation.ragas_evaluator import evaluate_with_ragas


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate naive RAG run with RAGAS")
    parser.add_argument("run_file")
    parser.add_argument("ground_truth_csv")
    parser.add_argument("--output", default="outputs/reports/naive_eval")
    args = parser.parse_args()

    dataset = build_eval_dataset(args.run_file, args.ground_truth_csv)
    result = evaluate_with_ragas(dataset, args.output)
    print(result["summary"])


if __name__ == "__main__":
    main()

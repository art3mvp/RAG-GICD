from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipelines.naive_rag_pipeline import NaiveRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive RAG pipeline")
    parser.add_argument("question", help="Question to answer")
    args = parser.parse_args()

    payload = NaiveRAGPipeline().run(args.question)
    print(payload["answer"])


if __name__ == "__main__":
    main()

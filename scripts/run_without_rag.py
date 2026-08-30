from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipelines.no_rag_pipeline import NoRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline without RAG")
    parser.add_argument("question", help="Question to answer without retrieval")
    args = parser.parse_args()

    payload = NoRAGPipeline().run(args.question)
    print(payload["answer"])


if __name__ == "__main__":
    main()

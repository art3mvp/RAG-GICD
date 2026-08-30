from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipelines.hybrid_rag_pipeline import HybridRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid RAG pipeline")
    parser.add_argument("question", help="Question to answer")
    args = parser.parse_args()

    payload = HybridRAGPipeline().run(args.question)
    print(payload["answer"])


if __name__ == "__main__":
    main()

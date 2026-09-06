from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from dotenv import load_dotenv


MODEL_REPO = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "bge-reranker-v2-m3"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download the local reranker snapshot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory for the model snapshot (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=args.output,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Reranker downloaded to {snapshot_path}")


if __name__ == "__main__":
    main()

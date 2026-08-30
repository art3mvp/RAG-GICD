from __future__ import annotations

from pathlib import Path

import pandas as pd


def compare_reports(naive_csv: str | Path, hybrid_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    naive = pd.read_csv(naive_csv)
    hybrid = pd.read_csv(hybrid_csv)

    naive_mean = naive.mean(numeric_only=True).add_prefix("naive_")
    hybrid_mean = hybrid.mean(numeric_only=True).add_prefix("hybrid_")
    merged = pd.concat([naive_mean, hybrid_mean]).to_frame().T

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged

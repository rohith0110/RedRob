import csv
from pathlib import Path

import pandas as pd

from .atomic_writes import write_csv_atomic


def write_v2_submission(run_path: str | Path, out_path: str | Path, limit: int = 100) -> Path:
    run_path = Path(run_path)
    out_path = Path(out_path)
    scores = pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet")
    reasoning = pd.read_parquet(run_path / "reasoning" / "reasoning_v2.parquet")
    merged = scores.merge(reasoning[["candidate_id", "reasoning"]], on="candidate_id", how="left").sort_values(["rank", "candidate_id"]).head(limit)

    def writer(handle) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        csv_writer.writeheader()
        for _, row in merged.iterrows():
            csv_writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "score": f"{float(row['final_score']):.6f}",
                "reasoning": row["reasoning"],
            })

    return write_csv_atomic(out_path, writer, expected_rows=len(merged), required_header=["candidate_id", "rank", "score", "reasoning"])

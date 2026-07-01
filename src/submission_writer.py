import csv
from pathlib import Path

from .atomic_writes import write_csv_atomic
from .baseline_reasoning import reasoning_for


def rank_rows(rows: list[dict], limit: int = 100) -> list[dict]:
    ranked = sorted(rows, key=lambda r: (-float(r["final_score"]), r["candidate_id"]))[:limit]
    for idx, row in enumerate(ranked, 1):
        row["rank"] = idx
    return ranked


def write_submission(rows: list[dict], out_path: str | Path, limit: int = 100) -> Path:
    out_path = Path(out_path)
    ranked = rank_rows(rows, limit)

    def writer(f):
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": f"{float(row['final_score']):.6f}",
                "reasoning": reasoning_for(row),
            })

    return write_csv_atomic(out_path, writer, expected_rows=len(ranked), required_header=["candidate_id", "rank", "score", "reasoning"])

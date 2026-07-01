import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--baseline-csv")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    scores = pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet").sort_values("rank").head(args.limit)
    evidence = pd.read_parquet(run_path / "evidence" / "evidence_summary.parquet").set_index("candidate_id")
    credibility = pd.read_parquet(run_path / "credibility" / "credibility_breakdown.parquet").set_index("candidate_id")
    behavioral = pd.read_parquet(run_path / "behavioral" / "availability_breakdown.parquet").set_index("candidate_id")
    baseline = None
    if args.baseline_csv:
        baseline = pd.read_csv(args.baseline_csv).set_index("candidate_id")
    for _, row in scores.iterrows():
        candidate_id = row["candidate_id"]
        print(f"{row['rank']:>3} {candidate_id} score={row['final_score']:.4f}")
        if baseline is not None and candidate_id in baseline.index:
            print(f"    baseline_rank={int(baseline.loc[candidate_id, 'rank'])}")
        print(f"    positive={evidence.loc[candidate_id, 'top_positive_evidence_ids']}")
        print(f"    negative={evidence.loc[candidate_id, 'top_negative_evidence_ids']}")
        print(f"    credibility_multiplier={credibility.loc[candidate_id, 'credibility_multiplier']:.4f}")
        print(f"    availability_multiplier={behavioral.loc[candidate_id, 'availability_multiplier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", required=True)
    args = parser.parse_args()

    run_path = Path(args.run_path)
    scores = pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet").sort_values("rank")
    normalized = pd.read_parquet(run_path / "normalized" / "candidates_normalized.parquet").set_index("candidate_id")
    credibility = pd.read_parquet(run_path / "credibility" / "credibility_breakdown.parquet").set_index("candidate_id")
    behavioral = pd.read_parquet(run_path / "behavioral" / "availability_breakdown.parquet").set_index("candidate_id")
    top = scores.head(100)
    top_ids = top["candidate_id"]
    summary = {
        "top_100_countries": dict(normalized.loc[top_ids, "country"].value_counts()),
        "top_100_titles": dict(normalized.loc[top_ids, "current_title"].value_counts().head(20)),
        "top_100_credibility_bands": {
            "high": int((credibility.loc[top_ids, "credibility_multiplier"] >= 0.95).sum()),
            "medium": int(((credibility.loc[top_ids, "credibility_multiplier"] < 0.95) & (credibility.loc[top_ids, "credibility_multiplier"] >= 0.8)).sum()),
            "low": int((credibility.loc[top_ids, "credibility_multiplier"] < 0.8).sum()),
        },
        "top_100_availability_bands": {
            "high": int((behavioral.loc[top_ids, "availability_multiplier"] >= 1.0).sum()),
            "medium": int(((behavioral.loc[top_ids, "availability_multiplier"] < 1.0) & (behavioral.loc[top_ids, "availability_multiplier"] >= 0.85)).sum()),
            "low": int((behavioral.loc[top_ids, "availability_multiplier"] < 0.85).sum()),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

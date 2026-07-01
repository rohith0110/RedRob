import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    baseline = pd.read_csv(Path(args.baseline)).set_index("candidate_id")
    v2 = pd.read_csv(Path(args.v2)).set_index("candidate_id")
    baseline_top = set(baseline.head(args.top).index)
    v2_top = set(v2.head(args.top).index)
    print("Entrants:", sorted(v2_top - baseline_top))
    print("Exits:", sorted(baseline_top - v2_top))
    shared = sorted(v2_top & baseline_top)
    changes = []
    for candidate_id in shared:
        changes.append({
            "candidate_id": candidate_id,
            "baseline_rank": int(baseline.loc[candidate_id, "rank"]),
            "v2_rank": int(v2.loc[candidate_id, "rank"]),
            "delta": int(baseline.loc[candidate_id, "rank"]) - int(v2.loc[candidate_id, "rank"]),
        })
    print(pd.DataFrame(changes).sort_values("delta", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

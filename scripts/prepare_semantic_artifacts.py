import argparse
import json
from pathlib import Path

import pandas as pd

from src.hashing import sha256_file
from src.io import iter_jsonl, validate_candidate_shape
from src.phase2_config import load_phase2_config
from src.semantic_artifact import candidate_checksum


def score_text(text: str) -> float:
    terms = ("search", "ranking", "retrieval", "recommend", "relevance", "production", "python")
    text = (text or "").lower()
    hits = sum(1 for term in terms if term in text)
    return min(1.0, hits / len(terms))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", default="artifacts/semantic")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_num, candidate, parse_issue in iter_jsonl(args.candidates):
        if parse_issue or validate_candidate_shape(candidate, row_num):
            continue
        profile = candidate.get("profile") or {}
        history = candidate.get("career_history") or []
        skills = candidate.get("skills") or []
        text = " ".join(
            [profile.get("summary", ""), profile.get("headline", "")]
            + [role.get("description", "") for role in history]
            + [skill.get("name", "") for skill in skills]
        )
        rows.append({"candidate_id": candidate["candidate_id"], "semantic_score": score_text(text)})

    parquet_path = out_dir / "semantic_scores.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    config = load_phase2_config()
    manifest = {
        "source_dataset_hash": sha256_file(args.candidates),
        "candidate_id_checksum": candidate_checksum([row["candidate_id"] for row in rows]),
        "resolved_config_hash": config.bundle_hashes["scoring"],
        "model_identifier": "offline-lexical-semantic-v1",
        "model_version": "offline-lexical-semantic-v1",
        "output_artifact_hash": sha256_file(parquet_path),
        "candidate_count": len(rows),
        "schema_version": "semantic-score-v1",
    }
    (out_dir / "semantic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

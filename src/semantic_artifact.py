from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .hashing import sha256_file, sha256_json


def candidate_checksum(candidate_ids: list[str]) -> str:
    return sha256_json(sorted(candidate_ids))


def resolve_semantic_paths(run_path: str | Path, artifact_path: str, manifest_path: str) -> tuple[Path, Path]:
    run_path = Path(run_path)
    artifact = Path(artifact_path)
    manifest = Path(manifest_path)
    if not artifact.is_absolute():
        run_local_artifact = run_path / artifact
        run_local_manifest = run_path / manifest
        if run_local_artifact.exists() or run_local_manifest.exists():
            return run_local_artifact, run_local_manifest
    return artifact, manifest


def load_semantic_scores(
    run_path: str | Path,
    config,
    candidate_ids: list[str],
    dataset_hash: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not bool(config.semantic_config.get("enabled", False)):
        return {}, {"status": "disabled", "reason": "semantic scoring disabled by configuration"}
    artifact_path, manifest_path = resolve_semantic_paths(
        run_path,
        config.semantic_config.get("artifact_path", "artifacts/semantic/semantic_scores.parquet"),
        config.semantic_config.get("manifest_path", "artifacts/semantic/semantic_manifest.json"),
    )
    if not artifact_path.exists() or not manifest_path.exists():
        return {}, {"status": "absent", "reason": "semantic artifact or manifest missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "source_dataset_hash": dataset_hash,
        "candidate_id_checksum": candidate_checksum(candidate_ids),
        "resolved_config_hash": config.bundle_hashes["scoring"],
        "candidate_count": len(candidate_ids),
        "output_artifact_hash": sha256_file(artifact_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return {}, {"status": "mismatch", "reason": f"{key} mismatch", "expected": value, "actual": manifest.get(key)}
    frame = pd.read_parquet(artifact_path)
    if set(frame["candidate_id"]) != set(candidate_ids):
        return {}, {"status": "mismatch", "reason": "candidate IDs mismatch"}
    scores = {row["candidate_id"]: float(row["semantic_score"]) for _, row in frame.iterrows()}
    return scores, {"status": "loaded", "artifact_path": str(artifact_path), "manifest_path": str(manifest_path)}

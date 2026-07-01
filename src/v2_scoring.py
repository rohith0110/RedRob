from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .atomic_writes import write_csv_atomic, write_json_atomic, write_parquet_atomic, write_text_atomic
from .hashing import sha256_file
from .phase2_config import Phase2ConfigBundle, load_phase2_config
from .semantic_artifact import load_semantic_scores

SCORE_VERSION = "phase2-v2-scoring-v1"


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize(value: float, cap: float) -> float:
    return _clamp(0.0 if cap <= 0 else float(value) / float(cap))


def _experience_score(years: float, bands: list[dict[str, Any]]) -> float:
    for band in bands:
        if float(band["min_years"]) <= years <= float(band["max_years"]):
            return float(band["score"])
    return 0.0


def _semantic_dataset_hash(run_path: Path) -> str:
    """Portable identity for binding a semantic artifact to this dataset.

    Prefer the raw-input hash recorded in the run manifest (the same value
    scripts/prepare_semantic_artifacts.py writes). Parquet bytes are not stable
    across pyarrow versions, so they make a poor cross-environment key; fall back
    to them only for standalone/test runs that never wrote a manifest.
    """
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        try:
            raw_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("input_file_sha256")
            if raw_hash:
                return str(raw_hash)
        except (ValueError, OSError):
            pass
    return sha256_file(run_path / "normalized" / "candidates_normalized.parquet")


def run_v2_scoring(
    run_path: str | Path,
    logger=None,
    config: Phase2ConfigBundle | None = None,
) -> dict[str, Any]:
    config = config or load_phase2_config()
    run_path = Path(run_path)
    normalized = pd.read_parquet(run_path / "normalized" / "candidates_normalized.parquet").set_index("candidate_id")
    evidence = pd.read_parquet(run_path / "evidence" / "evidence_summary.parquet").set_index("candidate_id")
    credibility = pd.read_parquet(run_path / "credibility" / "credibility_breakdown.parquet").set_index("candidate_id")
    behavioral = pd.read_parquet(run_path / "behavioral" / "availability_breakdown.parquet").set_index("candidate_id")

    weights = config.scoring_weights.get("feature_weights", {})
    evidence_caps = {
        category_name: float(category.get("max_contribution_cap", 1.0))
        for category_name, category in config.evidence_patterns.get("categories", {}).items()
    }
    semantic_cap = float(config.scoring_weights.get("score_caps", {}).get("semantic_score_cap", 0.08))
    experience_bands = list(config.role_rubric.get("experience_bands", []))
    semantic_scores, semantic_status = load_semantic_scores(
        run_path,
        config,
        candidate_ids=list(normalized.index),
        dataset_hash=_semantic_dataset_hash(run_path),
    )
    rows = []
    for candidate_id, normalized_row in normalized.iterrows():
        evidence_row = evidence.loc[candidate_id]
        credibility_row = credibility.loc[candidate_id]
        behavioral_row = behavioral.loc[candidate_id]
        retrieval = _normalize(evidence_row["retrieval_ranking_relevance_score"], evidence_caps["retrieval_ranking_relevance"])
        vector = _normalize(evidence_row["vector_ir_infrastructure_score"], evidence_caps["vector_ir_infrastructure"])
        ranking_eval = _normalize(evidence_row["ranking_evaluation_experimentation_score"], evidence_caps["ranking_evaluation_experimentation"])
        production = _normalize(evidence_row["production_delivery_systems_score"], evidence_caps["production_delivery_systems"])
        python = _normalize(evidence_row["python_practical_engineering_score"], evidence_caps["python_practical_engineering"])
        product = _normalize(evidence_row["product_founding_behavior_score"], evidence_caps["product_founding_behavior"])
        preferred = _normalize(evidence_row["preferred_differentiators_score"], evidence_caps["preferred_differentiators"])
        risk_signals = _normalize(evidence_row["risk_signals_score"], evidence_caps["risk_signals"])
        unsupported_skill_risk = _clamp(float(evidence_row["unsupported_skill_risk_score"]))
        career_relevance_score = _clamp(0.7 * retrieval + 0.15 * python + 0.15 * product)
        retrieval_ranking_depth_score = _clamp(0.45 * retrieval + 0.35 * vector + 0.20 * float(evidence_row["skill_corroboration_score"]))
        production_evaluation_score = _clamp(0.5 * production + 0.5 * ranking_eval)
        product_shipper_score = _clamp(0.6 * production + 0.4 * product)
        experience_recent_coding_score = _clamp(0.6 * _experience_score(float(normalized_row["years_of_experience"]), experience_bands) + 0.4 * python)
        corroborated_skill_score = _clamp(float(evidence_row["skill_corroboration_score"]) * max(retrieval, vector, python))
        preferred_differentiator_score = preferred
        location_logistics_score = _clamp(float(behavioral_row["location_logistics_score"]))
        semantic_score = _clamp(float(semantic_scores.get(candidate_id, 0.0)))
        base_fit_score = _clamp(
            float(weights.get("career_relevance_score", 0.35)) * career_relevance_score
            + float(weights.get("retrieval_ranking_depth_score", 0.15)) * retrieval_ranking_depth_score
            + float(weights.get("production_evaluation_score", 0.20)) * production_evaluation_score
            + float(weights.get("product_shipper_score", 0.10)) * product_shipper_score
            + float(weights.get("experience_recent_coding_score", 0.08)) * experience_recent_coding_score
            + float(weights.get("corroborated_skill_score", 0.05)) * corroborated_skill_score
            + float(weights.get("preferred_differentiator_score", 0.03)) * preferred_differentiator_score
            + float(weights.get("location_logistics_score", 0.04)) * location_logistics_score
            + semantic_cap * semantic_score
            - 0.08 * unsupported_skill_risk
            - 0.05 * risk_signals
        )
        final_score = round(
            base_fit_score
            * float(credibility_row["credibility_multiplier"])
            * float(behavioral_row["availability_multiplier"])
            * 100,
            6,
        )
        rows.append({
            "candidate_id": candidate_id,
            "career_relevance_score": round(career_relevance_score, 6),
            "retrieval_ranking_depth_score": round(retrieval_ranking_depth_score, 6),
            "production_evaluation_score": round(production_evaluation_score, 6),
            "product_shipper_score": round(product_shipper_score, 6),
            "experience_recent_coding_score": round(experience_recent_coding_score, 6),
            "corroborated_skill_score": round(corroborated_skill_score, 6),
            "unsupported_skill_risk_score": round(unsupported_skill_risk, 6),
            "preferred_differentiator_score": round(preferred_differentiator_score, 6),
            "location_logistics_score": round(location_logistics_score, 6),
            "credibility_multiplier": round(float(credibility_row["credibility_multiplier"]), 6),
            "availability_multiplier": round(float(behavioral_row["availability_multiplier"]), 6),
            "anomaly_risk_score": round(float(credibility_row["anomaly_risk_score"]), 6),
            "base_fit_score": round(base_fit_score, 6),
            "final_score": final_score,
            "rank": 0,
            "top_positive_evidence_ids": evidence_row["top_positive_evidence_ids"],
            "top_negative_evidence_ids": evidence_row["top_negative_evidence_ids"],
            "score_version": SCORE_VERSION,
            "deterministic_tie_break_key": candidate_id,
            "semantic_score": round(semantic_score, 6),
        })
        if logger and len(rows) % 10000 == 0:
            logger("scores_v2", "progress", f"scored {len(rows)} candidates", processed_count=len(rows))

    rows.sort(key=lambda row: (-row["final_score"], row["candidate_id"]))
    for index, row in enumerate(rows, 1):
        row["rank"] = index

    scores_dir = run_path / "scores"
    reports_dir = run_path / "reports"
    submissions_dir = run_path / "submissions"
    scores_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    breakdown_df = pd.DataFrame(rows)
    parquet_path = scores_dir / "score_breakdown_v2.parquet"
    csv_path = scores_dir / "score_breakdown_v2.csv"
    top_1000_path = scores_dir / "top_1000_diagnostics.csv"
    cohort_summary_path = scores_dir / "cohort_summary.json"
    report_path = reports_dir / "phase2_ranking_report.md"

    write_parquet_atomic(breakdown_df, parquet_path, expected_rows=len(breakdown_df), required_columns=["candidate_id"])

    def write_breakdown_csv(handle) -> None:
        breakdown_df.to_csv(handle, index=False)

    write_csv_atomic(csv_path, write_breakdown_csv, expected_rows=len(breakdown_df), required_header=list(breakdown_df.columns))

    def write_top_csv(handle) -> None:
        breakdown_df.head(1000).to_csv(handle, index=False)

    write_csv_atomic(top_1000_path, write_top_csv, expected_rows=min(1000, len(breakdown_df)), required_header=list(breakdown_df.columns))
    cohort_summary = {
        "candidate_count": len(rows),
        "top_100_country_distribution": dict(normalized.loc[breakdown_df.head(100)["candidate_id"], "country"].value_counts()) if len(breakdown_df) else {},
        "top_100_title_distribution": dict(normalized.loc[breakdown_df.head(100)["candidate_id"], "current_title"].value_counts()) if len(breakdown_df) else {},
    }
    write_json_atomic(cohort_summary_path, cohort_summary)
    write_json_atomic(scores_dir / "semantic_fallback_status.json", semantic_status)
    write_text_atomic(
        report_path,
        "# Phase 2 Ranking Report\n\n"
        f"Candidates scored: {len(rows)}\n\n"
        f"Top score: {rows[0]['final_score'] if rows else 0}\n\n"
        f"Semantic status: {semantic_status['status']}\n",
    )
    return {
        "score_breakdown_v2_parquet": str(parquet_path),
        "score_breakdown_v2_csv": str(csv_path),
        "top_1000_diagnostics_csv": str(top_1000_path),
        "cohort_summary_json": str(cohort_summary_path),
        "phase2_ranking_report_md": str(report_path),
        "semantic_status_json": str(scores_dir / "semantic_fallback_status.json"),
    }

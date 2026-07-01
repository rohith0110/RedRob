from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .atomic_writes import write_csv_atomic, write_parquet_atomic, write_text_atomic
from .audit import parse_date
from .phase2_config import Phase2ConfigBundle, load_phase2_config


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _last_active_recency_score(last_active_date: str, reference_date: date, config: dict[str, Any]) -> float:
    parsed = parse_date(last_active_date)
    if not parsed:
        return float(config.get("stale_score", 0.35))
    age_days = max(0, (reference_date - parsed).days)
    recent_days = int(config.get("recent_days", 60))
    stale_days = int(config.get("stale_days", 180))
    recent_score = float(config.get("recent_score", 1.0))
    stale_score = float(config.get("stale_score", 0.35))
    if age_days <= recent_days:
        return recent_score
    if age_days >= stale_days:
        return stale_score
    span = max(1, stale_days - recent_days)
    ratio = (age_days - recent_days) / span
    return recent_score - ratio * (recent_score - stale_score)


def run_behavioral_analysis(
    run_path: str | Path,
    logger=None,
    config: Phase2ConfigBundle | None = None,
    dataset_reference_date: date | None = None,
) -> dict[str, Any]:
    config = config or load_phase2_config()
    dataset_reference_date = dataset_reference_date or date.today()
    run_path = Path(run_path)
    normalized = pd.read_parquet(run_path / "normalized" / "candidates_normalized.parquet").set_index("candidate_id")
    evidence_summary = pd.read_parquet(run_path / "evidence" / "evidence_summary.parquet").set_index("candidate_id")
    rules = config.behavioral_rules
    inputs = rules.get("inputs", {})
    location_rules = rules.get("location_logistics", {})
    last_active_rules = rules.get("last_active_recency", {})
    bounds = rules.get("availability_multiplier", {})
    minimum = float(bounds.get("min", 0.72))
    maximum = float(bounds.get("max", 1.08))

    contexts = {}
    with (run_path / "normalized" / "candidate_context.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                contexts[row["candidate_id"]] = row

    rows = []
    for candidate_id, normalized_row in normalized.iterrows():
        context = contexts.get(candidate_id, {})
        signals = context.get("redrob_signals") or {}
        assessments = context.get("skill_assessments") or []
        relevant_assessment = max((float(item.get("score", 0)) for item in assessments), default=0.0) / 100.0
        if candidate_id in evidence_summary.index and evidence_summary.loc[candidate_id, "skill_corroboration_score"] <= 0:
            relevant_assessment = 0.0
        work_mode = str(normalized_row.get("preferred_work_mode", "")).lower()
        country = str(normalized_row.get("country", "")).lower()
        last_active_recency_score = _last_active_recency_score(str(signals.get("last_active_date", "")), dataset_reference_date, last_active_rules)
        location_logistics_score = float(location_rules.get("preferred_score", 1.0)) if (
            work_mode in set(location_rules.get("preferred_modes", []))
            or bool(normalized_row.get("willing_to_relocate"))
            or country == str(location_rules.get("preferred_country", "")).lower()
        ) else float(location_rules.get("fallback_score", 0.55))
        components = {
            "open_to_work_component": float(inputs.get("open_to_work_weight", 0.2)) * float(bool(normalized_row.get("open_to_work_flag"))),
            "response_rate_component": float(inputs.get("response_rate_weight", 0.18)) * _clamp(float(normalized_row.get("recruiter_response_rate") or 0), 0.0, 1.0),
            "response_time_component": float(inputs.get("response_time_weight", 0.12)) * (1.0 - _clamp(float(normalized_row.get("avg_response_time_hours") or 240) / 240.0, 0.0, 1.0)),
            "interview_completion_component": float(inputs.get("interview_completion_weight", 0.12)) * _clamp(float(normalized_row.get("interview_completion_rate") or 0), 0.0, 1.0),
            "notice_period_component": float(inputs.get("notice_period_weight", 0.12)) * (1.0 - _clamp(float(normalized_row.get("notice_period_days") or 120) / 120.0, 0.0, 1.0)),
            "relocation_component": float(inputs.get("relocation_weight", 0.08)) * float(bool(normalized_row.get("willing_to_relocate"))),
            "work_mode_component": float(inputs.get("work_mode_weight", 0.1)) * (1.0 if work_mode in set(location_rules.get("preferred_modes", [])) else 0.0),
            "relevant_assessment_component": float(inputs.get("relevant_assessment_weight", 0.05)) * relevant_assessment,
            "github_activity_component": float(inputs.get("github_activity_weight", 0.03)) * _clamp(float(signals.get("github_activity_score", normalized_row.get("github_activity_score") or 0)) / 100.0, 0.0, 1.0),
            "last_active_recency_component": float(inputs.get("last_active_recency_weight", 0.0)) * _clamp(last_active_recency_score, 0.0, 1.0),
        }
        availability_signal_score = sum(components.values())
        availability_multiplier = _clamp(0.72 + availability_signal_score * (maximum - minimum), minimum, maximum)
        rows.append({
            "candidate_id": candidate_id,
            "location_logistics_score": round(location_logistics_score, 6),
            "last_active_recency_score": round(last_active_recency_score, 6),
            "availability_signal_score": round(availability_signal_score, 6),
            "availability_multiplier": round(availability_multiplier, 6),
            **{name: round(value, 6) for name, value in components.items()},
        })
        if logger and len(rows) % 10000 == 0:
            logger("behavioral", "progress", f"evaluated availability for {len(rows)} candidates", processed_count=len(rows))

    breakdown_df = pd.DataFrame(rows).sort_values("candidate_id")
    behavioral_dir = run_path / "behavioral"
    reports_dir = run_path / "reports"
    behavioral_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = behavioral_dir / "availability_breakdown.parquet"
    csv_path = behavioral_dir / "availability_breakdown.csv"
    report_path = reports_dir / "behavioral_modifier_report.md"

    write_parquet_atomic(breakdown_df, parquet_path, expected_rows=len(breakdown_df), required_columns=["candidate_id"])

    def write_breakdown_csv(handle) -> None:
        breakdown_df.to_csv(handle, index=False)

    write_csv_atomic(csv_path, write_breakdown_csv, expected_rows=len(breakdown_df), required_header=list(breakdown_df.columns))
    write_text_atomic(
        report_path,
        "# Behavioral Modifier Report\n\n"
        f"Candidates: {len(breakdown_df)}\n\n"
        f"Average multiplier: {breakdown_df['availability_multiplier'].mean() if len(breakdown_df) else 0:.4f}\n",
    )
    return {
        "availability_breakdown_parquet": str(parquet_path),
        "availability_breakdown_csv": str(csv_path),
        "behavioral_report_md": str(report_path),
    }

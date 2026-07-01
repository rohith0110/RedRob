import csv
import json
import re
from pathlib import Path

import pandas as pd

from .atomic_writes import write_csv_atomic, write_parquet_atomic, write_text_atomic
from .scoring_config import ScoringConfig, TextCategory, load_scoring_config


def term_score(text: str, terms: tuple[str, ...], divisor: int) -> float:
    text = (text or "").lower()
    hits = sum(1 for term in terms if re.search(r"\b" + re.escape(term).replace("\\ ", r"\s+") + r"\b", text))
    return min(1.0, hits / divisor)


def clamp(value):
    return max(0.0, min(1.0, float(value)))


def category_score(fields: dict[str, str], category: TextCategory) -> float:
    terms = category.positive_phrases + category.aliases
    return clamp(sum(
        category.field_weights[name] * term_score(fields[name], terms, category.hit_divisors[name])
        for name in category.field_weights
    ))


def experience_score(years: float, config: ScoringConfig) -> float:
    for band in config.experience_bands:
        if band["min_years"] <= years <= band["max_years"]:
            return float(band["score"])
    return 0.0


def score_candidate_rows(rows: list[dict], config: ScoringConfig | None = None) -> list[dict]:
    config = config or load_scoring_config()
    scored = []
    for row in rows:
        career = row.get("combined_career_text", "")
        profile = row.get("combined_profile_text", "")
        skills = row.get("combined_skill_text", "")
        career_rel = term_score(career, config.role_relevance.positive_phrases + config.role_relevance.aliases, config.role_relevance.hit_divisors["career"])
        base = category_score({"career": career, "profile_skills": profile + " " + skills}, config.role_relevance)
        prod = category_score({"career_profile": career + " " + profile}, config.production_evaluation)
        py = category_score({"career_skills": career + " " + skills}, config.python_engineering)
        years = float(row.get("years_of_experience") or 0)
        exp = experience_score(years, config)
        mode = (row.get("preferred_work_mode") or "").lower()
        country = (row.get("country") or "").lower()
        loc = config.location["preferred_score"] if mode in config.location["preferred_modes"] or row.get("willing_to_relocate") or country == config.location["preferred_country"] else config.location["fallback_score"]
        response = clamp(row.get("recruiter_response_rate") or 0)
        response_time = float(row.get("avg_response_time_hours") or 999)
        notice = float(row.get("notice_period_days") or 180)
        avail = clamp(
            (config.availability["open_to_work"] if row.get("open_to_work_flag") else 0)
            + response * config.availability["response_rate"]
            + (1 - min(response_time, config.availability["max_response_time_hours"]) / config.availability["max_response_time_hours"]) * config.availability["response_time"]
            + (1 - min(notice, config.availability["max_notice_days"]) / config.availability["max_notice_days"]) * config.availability["notice_period"]
        )
        penalty = config.anomaly_penalty.get(row.get("anomaly_severity"), config.anomaly_penalty["default"])
        final = max(0.0, (
            config.positive_weights["career_text_relevance"] * base
            + config.positive_weights["production_evaluation"] * prod
            + config.positive_weights["python_engineering"] * py
            + config.positive_weights["experience_plausibility"] * exp
            + config.positive_weights["location_logistics"] * loc
            + config.positive_weights["behavioral_availability"] * avail
            - penalty
        ) * 100)
        if base < config.low_relevance_threshold:
            final = min(final, config.low_relevance_max_score)
        positives = []
        if career_rel:
            positives.append(config.role_relevance.positive_evidence)
        if prod:
            positives.append(config.production_evaluation.positive_evidence)
        if py:
            positives.append(config.python_engineering.positive_evidence)
        negatives = []
        if penalty:
            negatives.append(f"{row.get('anomaly_severity')} audit anomalies reduce confidence")
        if base < config.low_relevance_threshold:
            negatives.append(config.role_relevance.weak_evidence)
        scored.append({
            "candidate_id": row["candidate_id"],
            "base_relevance_score": round(base, 6),
            "production_evaluation_score": round(prod, 6),
            "python_engineering_score": round(py, 6),
            "experience_score": round(exp, 6),
            "location_logistics_score": round(loc, 6),
            "availability_score": round(avail, 6),
            "anomaly_penalty": round(penalty, 6),
            "final_score": round(final, 6),
            "deterministic_tie_break_key": row["candidate_id"],
            "top_positive_evidence": "; ".join(positives) or "limited role-specific evidence found",
            "top_negative_evidence": "; ".join(negatives),
        })
    return scored


def run_scoring(run_path: str | Path, logger=None, config: ScoringConfig | None = None) -> dict:
    config = config or load_scoring_config()
    run_path = Path(run_path)
    out_dir = run_path / "scores"
    reports_dir = run_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(run_path / "normalized" / "candidates_normalized.parquet").to_dict("records")
    scored = score_candidate_rows(rows, config=config)
    scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    parquet = out_dir / "baseline_score_breakdown.parquet"
    csv_path = out_dir / "baseline_score_breakdown.csv"
    top_path = out_dir / "top_500_baseline.csv"
    write_parquet_atomic(pd.DataFrame(scored), parquet, expected_rows=len(scored), required_columns=["candidate_id"])

    def write_scores_csv(f):
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    write_csv_atomic(csv_path, write_scores_csv, expected_rows=len(scored), required_header=list(scored[0].keys()))

    def write_top_csv(f):
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored[:500])

    write_csv_atomic(top_path, write_top_csv, expected_rows=min(500, len(scored)), required_header=list(scored[0].keys()))
    write_text_atomic(reports_dir / "baseline_summary.md", f"# Baseline Summary\n\nScored rows: {len(scored)}\n\nTop score: {scored[0]['final_score'] if scored else 0}\n")
    if logger:
        logger("scoring", "completed", f"scored {len(scored)} candidates", processed_count=len(scored))
    return {"scoring_output_row_count": len(scored), "score_breakdown_csv": str(csv_path), "score_breakdown_parquet": str(parquet)}

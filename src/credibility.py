from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .atomic_writes import write_csv_atomic, write_jsonl_atomic, write_parquet_atomic, write_text_atomic
from .audit import audit_candidate, months_between, parse_date
from .phase2_config import Phase2ConfigBundle, load_phase2_config


def _rebuild_candidate(context: dict[str, Any]) -> dict[str, Any]:
    profile = context.get("profile") or {}
    signals = context.get("redrob_signals") or {}
    return {
        "candidate_id": context["candidate_id"],
        "profile": {
            "headline": profile.get("headline", ""),
            "summary": profile.get("summary", ""),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "current_industry": profile.get("current_industry", ""),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "years_of_experience": float(profile.get("years_of_experience") or 0),
        },
        "career_history": [
            {
                "company": role.get("company", ""),
                "title": role.get("title", ""),
                "start_date": role.get("start_date"),
                "end_date": role.get("end_date"),
                "duration_months": role.get("duration_months"),
                "is_current": role.get("is_current"),
                "description": role.get("description", ""),
            }
            for role in context.get("career_history") or []
        ],
        "skills": [
            {
                "name": skill.get("name", ""),
                "proficiency": skill.get("proficiency", ""),
                "duration_months": skill.get("duration_months"),
            }
            for skill in context.get("skills") or []
        ],
        "education": [],
        "certifications": context.get("certifications") or [],
        "redrob_signals": {
            "signup_date": signals.get("signup_date", ""),
            "last_active_date": signals.get("last_active_date", ""),
            "expected_salary_range_inr_lpa": signals.get("expected_salary_range_inr_lpa") or {},
        },
    }


def _overlap_months(roles: list[dict[str, Any]], reference_date: date) -> int:
    overlap = 0
    dated_roles = []
    for role in roles:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date")) or reference_date
        if start and end:
            dated_roles.append((start, end))
    dated_roles.sort()
    for idx in range(len(dated_roles) - 1):
        current_start, current_end = dated_roles[idx]
        next_start, next_end = dated_roles[idx + 1]
        if next_start < current_end:
            overlap += months_between(next_start, min(current_end, next_end))
    return overlap


def _trigger(rule_name: str, rule: dict[str, Any], source_path: str, detail: str, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "rule_name": rule_name,
        "severity": rule["severity"],
        "score_contribution": round(float(rule["score_contribution"]), 6),
        "source_path": source_path,
        "detail": detail,
        "explanation_label": rule["explanation_label"],
        "candidate_facing_reasoning_eligible": bool(rule.get("candidate_facing_reasoning", False)),
    }


def _audit_issue_handler(rule_id: str) -> Callable[..., list[dict[str, Any]]]:
    def handler(candidate_id: str, context: dict[str, Any], summary_row: pd.Series | None, issues_by_type: dict[str, list[dict[str, Any]]], rule: dict[str, Any], reference_date: date) -> list[dict[str, Any]]:
        issues = issues_by_type.get(rule_id, [])
        return [_trigger(rule_id, rule, "candidate", issue["reason"], candidate_id) for issue in issues]

    return handler


def _excessive_overlap_handler(candidate_id: str, context: dict[str, Any], summary_row: pd.Series | None, issues_by_type: dict[str, list[dict[str, Any]]], rule: dict[str, Any], reference_date: date) -> list[dict[str, Any]]:
    overlap = _overlap_months(context.get("career_history") or [], reference_date)
    max_overlap_months = int(rule.get("max_overlap_months", 12))
    if overlap <= max_overlap_months:
        return []
    return [
        _trigger(
            "excessive_overlap",
            rule,
            "career_history",
            f"career history contains {overlap} overlapping months, above the {max_overlap_months}-month threshold",
            candidate_id,
        )
    ]


def _unsupported_advanced_skill_handler(candidate_id: str, context: dict[str, Any], summary_row: pd.Series | None, issues_by_type: dict[str, list[dict[str, Any]]], rule: dict[str, Any], reference_date: date) -> list[dict[str, Any]]:
    risky_terms = tuple(str(term).lower() for term in rule.get("risky_skill_terms", []))
    corroboration_threshold = float(rule.get("corroboration_threshold", 0.4))
    minimum_skill_count = int(rule.get("minimum_skill_count", 1))
    advanced_skills = [skill for skill in context.get("skills") or [] if str(skill.get("proficiency", "")).lower() in {"advanced", "expert"}]
    risky_advanced_skills = [
        skill
        for skill in advanced_skills
        if any(term in str(skill.get("name", "")).lower() for term in risky_terms)
    ]
    unsupported_skill_risk = float(summary_row["unsupported_skill_risk_score"]) if summary_row is not None else 0.0
    if len(risky_advanced_skills) < minimum_skill_count or unsupported_skill_risk <= corroboration_threshold:
        return []
    return [
        _trigger(
            "unsupported_advanced_skill",
            rule,
            "skills",
            f"{len(risky_advanced_skills)} advanced skills lack career corroboration",
            candidate_id,
        )
    ]


def _title_description_contradiction_handler(candidate_id: str, context: dict[str, Any], summary_row: pd.Series | None, issues_by_type: dict[str, list[dict[str, Any]]], rule: dict[str, Any], reference_date: date) -> list[dict[str, Any]]:
    title_terms = tuple(str(term).lower() for term in rule.get("supported_title_families", []))
    description_terms = tuple(str(term).lower() for term in rule.get("supporting_description_terms", []))
    threshold = int(rule.get("minimum_evidence_threshold", 1))
    title_text = " ".join(role.get("title", "") for role in context.get("career_history") or []).lower()
    description_text = " ".join(role.get("description", "") for role in context.get("career_history") or []).lower()
    title_hits = sum(1 for term in title_terms if term in title_text)
    if title_hits < threshold or any(term in description_text for term in description_terms):
        return []
    return [
        _trigger(
            "title_description_contradiction",
            rule,
            "career_history.title",
            "titles imply relevance but descriptions do not support it",
            candidate_id,
        )
    ]


RULE_HANDLERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "signup_after_last_active": _audit_issue_handler("signup_after_last_active"),
    "current_role_has_past_end_date": _audit_issue_handler("current_role_has_past_end_date"),
    "role_end_before_start": _audit_issue_handler("role_end_before_start"),
    "duration_months_mismatch": _audit_issue_handler("duration_months_mismatch"),
    "multiple_current_roles": _audit_issue_handler("multiple_current_roles"),
    "experience_span_mismatch": _audit_issue_handler("experience_span_mismatch"),
    "skill_duration_implausible": _audit_issue_handler("skill_duration_implausible"),
    "salary_min_greater_than_max": _audit_issue_handler("salary_min_greater_than_max"),
    "skills_career_evidence_mismatch": _audit_issue_handler("skills_career_evidence_mismatch"),
    "excessive_overlap": _excessive_overlap_handler,
    "unsupported_advanced_skill": _unsupported_advanced_skill_handler,
    "title_description_contradiction": _title_description_contradiction_handler,
}


def validate_credibility_rule_registry(config: Phase2ConfigBundle) -> None:
    configured_rules = dict(config.credibility_rules.get("rules", {}))
    missing_config = sorted(set(RULE_HANDLERS) - set(configured_rules))
    if missing_config:
        raise ValueError(f"active Python credibility rule handlers are not declared in config: {missing_config}")
    missing_handlers = sorted(
        rule_id
        for rule_id, rule in configured_rules.items()
        if bool(rule.get("enabled")) and rule_id not in RULE_HANDLERS
    )
    if missing_handlers:
        raise ValueError(f"enabled credibility rules lack a Python handler: {missing_handlers}")


def run_credibility_analysis(
    run_path: str | Path,
    logger=None,
    config: Phase2ConfigBundle | None = None,
    dataset_reference_date: date | None = None,
) -> dict[str, Any]:
    config = config or load_phase2_config()
    validate_credibility_rule_registry(config)
    dataset_reference_date = dataset_reference_date or date.today()
    run_path = Path(run_path)
    normalized_dir = run_path / "normalized"
    evidence_dir = run_path / "evidence"
    credibility_dir = run_path / "credibility"
    reports_dir = run_path / "reports"
    credibility_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    multiplier_bounds = config.credibility_rules.get("multiplier_bounds", {})
    rule_configs = dict(config.credibility_rules.get("rules", {}))
    evidence_summary = pd.read_parquet(evidence_dir / "evidence_summary.parquet").set_index("candidate_id")

    contexts = {}
    with (normalized_dir / "candidate_context.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                contexts[row["candidate_id"]] = row

    trigger_rows = []
    breakdown_rows = []

    for candidate_id, context in contexts.items():
        summary_row = evidence_summary.loc[candidate_id] if candidate_id in evidence_summary.index else None
        candidate = _rebuild_candidate(context)
        audit_issues = audit_candidate(candidate, reference_date=dataset_reference_date)
        issues_by_type: dict[str, list[dict[str, Any]]] = {}
        for item in audit_issues:
            issues_by_type.setdefault(item["type"], []).append(item)

        triggers = []
        for rule_id, rule in rule_configs.items():
            if not bool(rule.get("enabled", False)):
                continue
            triggers.extend(RULE_HANDLERS[rule_id](candidate_id, context, summary_row, issues_by_type, rule, dataset_reference_date))

        severity_counter = Counter(trigger["severity"] for trigger in triggers)
        risk_score = min(1.0, sum(float(trigger["score_contribution"]) for trigger in triggers))
        severe_min = float(multiplier_bounds.get("severe_min", 0.25))
        credibility_multiplier = max(severe_min, round(1.0 - risk_score, 6))
        breakdown_rows.append({
            "candidate_id": candidate_id,
            "anomaly_risk_score": round(risk_score, 6),
            "credibility_multiplier": credibility_multiplier,
            "triggered_rule_count": len(triggers),
            "minor_rule_count": severity_counter.get("minor", 0),
            "moderate_rule_count": severity_counter.get("moderate", 0),
            "severe_rule_count": severity_counter.get("severe", 0),
            "unsupported_skill_risk_score": round(float(summary_row["unsupported_skill_risk_score"]) if summary_row is not None else 0.0, 6),
            "evidence_risk_score": round(float(summary_row["risk_signals_score"]) if summary_row is not None else 0.0, 6),
        })
        trigger_rows.extend(triggers)
        if logger and len(breakdown_rows) % 10000 == 0:
            logger("credibility", "progress", f"evaluated credibility for {len(breakdown_rows)} candidates", processed_count=len(breakdown_rows))

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values("candidate_id")
    parquet_path = credibility_dir / "credibility_breakdown.parquet"
    csv_path = credibility_dir / "credibility_breakdown.csv"
    triggered_path = credibility_dir / "credibility_rules_triggered.jsonl"
    report_path = reports_dir / "credibility_report.md"

    write_parquet_atomic(breakdown_df, parquet_path, expected_rows=len(breakdown_df), required_columns=["candidate_id"])

    def write_breakdown_csv(handle) -> None:
        breakdown_df.to_csv(handle, index=False)

    write_csv_atomic(csv_path, write_breakdown_csv, expected_rows=len(breakdown_df), required_header=list(breakdown_df.columns))
    write_jsonl_atomic(triggered_path, trigger_rows)
    write_text_atomic(
        report_path,
        "# Credibility Report\n\n"
        f"Candidates: {len(breakdown_df)}\n\n"
        f"Triggered rules: {len(trigger_rows)}\n\n"
        f"Average multiplier: {breakdown_df['credibility_multiplier'].mean() if len(breakdown_df) else 0:.4f}\n\n"
        f"Dataset reference date: {dataset_reference_date}\n",
    )
    return {
        "credibility_breakdown_parquet": str(parquet_path),
        "credibility_breakdown_csv": str(csv_path),
        "credibility_rules_triggered_jsonl": str(triggered_path),
        "credibility_report_md": str(report_path),
    }

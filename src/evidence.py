from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .atomic_writes import write_csv_atomic, write_json_atomic, write_jsonl_atomic, write_parquet_atomic, write_text_atomic
from .audit import parse_date
from .hashing import sha256_json
from .phase2_config import Phase2ConfigBundle, load_phase2_config, phase2_config_manifest

SOURCE_FAMILIES = {
    "career_description": "career",
    "career_title": "career",
    "profile_summary": "profile",
    "profile_headline": "profile",
    "skill": "skill",
    "assessment": "assessment",
    "behavioral": "behavioral",
}
SUMMARY_CATEGORY_COLUMNS = {
    "retrieval_ranking_relevance": "retrieval_ranking_relevance_score",
    "vector_ir_infrastructure": "vector_ir_infrastructure_score",
    "ranking_evaluation_experimentation": "ranking_evaluation_experimentation_score",
    "production_delivery_systems": "production_delivery_systems_score",
    "python_practical_engineering": "python_practical_engineering_score",
    "product_founding_behavior": "product_founding_behavior_score",
    "preferred_differentiators": "preferred_differentiators_score",
    "risk_signals": "risk_signals_score",
}

def _match_terms(text: str, terms: Iterable[str]) -> list[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    return sorted({term for term in terms if term in text})


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def build_candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    certifications = candidate.get("certifications") or []
    assessments = signals.get("skill_assessment_scores") or {}
    return {
        "candidate_id": candidate["candidate_id"],
        "profile": {
            "headline": profile.get("headline", ""),
            "summary": profile.get("summary", ""),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "years_of_experience": float(profile.get("years_of_experience") or 0),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "current_company_size": profile.get("current_company_size", ""),
            "current_industry": profile.get("current_industry", ""),
        },
        "career_history": [
            {
                "index": idx,
                "company": role.get("company", ""),
                "title": role.get("title", ""),
                "start_date": role.get("start_date"),
                "end_date": role.get("end_date"),
                "duration_months": role.get("duration_months"),
                "is_current": bool(role.get("is_current")),
                "industry": role.get("industry", ""),
                "company_size": role.get("company_size", ""),
                "description": role.get("description", ""),
            }
            for idx, role in enumerate(history)
        ],
        "skills": [
            {
                "index": idx,
                "name": skill.get("name", ""),
                "proficiency": skill.get("proficiency", ""),
                "endorsements": skill.get("endorsements"),
                "duration_months": skill.get("duration_months"),
            }
            for idx, skill in enumerate(skills)
        ],
        "certifications": [
            {
                "index": idx,
                "name": certification.get("name", ""),
                "issuer": certification.get("issuer", ""),
            }
            for idx, certification in enumerate(certifications)
        ],
        "skill_assessments": [
            {"index": idx, "name": name, "score": float(score)}
            for idx, (name, score) in enumerate(sorted(assessments.items()))
            if isinstance(score, (int, float))
        ],
        "redrob_signals": {
            "signup_date": signals.get("signup_date", ""),
            "notice_period_days": int(signals.get("notice_period_days") or 0),
            "last_active_date": signals.get("last_active_date", ""),
            "open_to_work_flag": bool(signals.get("open_to_work_flag")),
            "recruiter_response_rate": float(signals.get("recruiter_response_rate") or 0),
            "avg_response_time_hours": float(signals.get("avg_response_time_hours") or 0),
            "interview_completion_rate": float(signals.get("interview_completion_rate") or 0),
            "preferred_work_mode": signals.get("preferred_work_mode", ""),
            "willing_to_relocate": bool(signals.get("willing_to_relocate")),
            "github_activity_score": float(signals.get("github_activity_score") or 0),
            "expected_salary_range_inr_lpa": signals.get("expected_salary_range_inr_lpa") or {},
        },
        "candidate_record_hash": sha256_json(candidate),
    }


def _iter_fragments(context: dict[str, Any]) -> Iterable[dict[str, Any]]:
    profile = context.get("profile") or {}
    for role_index, role in enumerate(context.get("career_history") or []):
        index = role.get("index", role_index)
        if role.get("description"):
            yield {
                "source_path": f"career_history[{index}].description",
                "source_type": "career_description",
                "exact_source_excerpt": role["description"],
                "role": role,
            }
        if role.get("title"):
            yield {
                "source_path": f"career_history[{index}].title",
                "source_type": "career_title",
                "exact_source_excerpt": role["title"],
                "role": role,
            }
    if profile.get("summary"):
        yield {
            "source_path": "profile.summary",
            "source_type": "profile_summary",
            "exact_source_excerpt": profile["summary"],
            "role": None,
        }
    if profile.get("headline"):
        yield {
            "source_path": "profile.headline",
            "source_type": "profile_headline",
            "exact_source_excerpt": profile["headline"],
            "role": None,
        }
    for skill_index, skill in enumerate(context.get("skills") or []):
        if skill.get("name"):
            yield {
                "source_path": f"skills[{skill.get('index', skill_index)}].name",
                "source_type": "skill",
                "exact_source_excerpt": skill["name"],
                "role": None,
            }
    for assessment_index, assessment in enumerate(context.get("skill_assessments") or []):
        yield {
            "source_path": f"skill_assessments[{assessment.get('index', assessment_index)}].name",
            "source_type": "assessment",
            "exact_source_excerpt": assessment["name"],
            "role": None,
        }


def _recency_weight(role: dict[str, Any] | None, recency_config: dict[str, Any], reference_date: date) -> float:
    if not role:
        return 1.0
    if role.get("is_current"):
        return float(recency_config.get("current_role_multiplier", 1.0))
    recent_years = int(recency_config.get("recent_years", 5))
    older_multiplier = float(recency_config.get("older_role_multiplier", 0.85))
    end_date = parse_date(role.get("end_date")) or parse_date(role.get("start_date"))
    if not end_date:
        return 1.0
    years_old = max(0.0, (reference_date - end_date).days / 365.25)
    return 1.0 if years_old <= recent_years else older_multiplier


def _source_is_reconstructible(context: dict[str, Any], source_path: str, excerpt: str) -> bool:
    if source_path.startswith("career_history["):
        index = int(source_path.split("[", 1)[1].split("]", 1)[0])
        field = source_path.split(".", 1)[1]
        history = context.get("career_history") or []
        return index < len(history) and str(history[index].get(field, "")) == str(excerpt)
    if source_path.startswith("skills["):
        index = int(source_path.split("[", 1)[1].split("]", 1)[0])
        field = source_path.split(".", 1)[1]
        skills = context.get("skills") or []
        return index < len(skills) and str(skills[index].get(field, "")) == str(excerpt)
    if source_path.startswith("skill_assessments["):
        index = int(source_path.split("[", 1)[1].split("]", 1)[0])
        field = source_path.split(".", 1)[1]
        assessments = context.get("skill_assessments") or []
        return index < len(assessments) and str(assessments[index].get(field, "")) == str(excerpt)
    if source_path.startswith("profile."):
        field = source_path.split(".", 1)[1]
        return str((context.get("profile") or {}).get(field, "")) == str(excerpt)
    return False


def _candidate_summary(candidate_id: str, evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "candidate_id": candidate_id,
        "evidence_item_count": len(evidence_rows),
        "positive_evidence_count": sum(1 for row in evidence_rows if row["polarity"] == "positive"),
        "negative_evidence_count": sum(1 for row in evidence_rows if row["polarity"] == "negative"),
        "career_evidence_score": round(sum(row["contribution_after_caps"] for row in evidence_rows if SOURCE_FAMILIES.get(row["source_type"]) == "career" and row["polarity"] == "positive"), 6),
        "skill_only_evidence_score": 0.0,
        "skill_corroboration_score": 0.0,
        "unsupported_skill_risk_score": 0.0,
        "top_positive_evidence_ids": json.dumps([row["evidence_id"] for row in sorted([r for r in evidence_rows if r["polarity"] == "positive"], key=lambda item: item["contribution_after_caps"], reverse=True)[:3]]),
        "top_negative_evidence_ids": json.dumps([row["evidence_id"] for row in sorted([r for r in evidence_rows if r["polarity"] == "negative"], key=lambda item: item["contribution_after_caps"], reverse=True)[:3]]),
        "evidence_confidence": round(sum(row["confidence"] for row in evidence_rows) / len(evidence_rows), 6) if evidence_rows else 0.0,
    }
    for column in SUMMARY_CATEGORY_COLUMNS.values():
        summary[column] = 0.0
    positive_by_category = defaultdict(float)
    positive_families_by_category = defaultdict(set)
    skill_positive_categories = set()
    for row in evidence_rows:
        column = SUMMARY_CATEGORY_COLUMNS.get(row["normalized_category"])
        if row["polarity"] == "positive" and column:
            positive_by_category[column] += float(row["contribution_after_caps"])
            positive_families_by_category[row["normalized_category"]].add(SOURCE_FAMILIES.get(row["source_type"], row["source_type"]))
            if SOURCE_FAMILIES.get(row["source_type"]) == "skill":
                skill_positive_categories.add(row["normalized_category"])
        elif row["polarity"] == "negative" and column:
            summary[column] = round(summary[column] + float(row["contribution_after_caps"]), 6)
    for column, value in positive_by_category.items():
        summary[column] = round(summary[column] + value, 6)
    corroborated_skill_categories = {
        category
        for category in skill_positive_categories
        if "career" in positive_families_by_category.get(category, set())
    }
    if skill_positive_categories:
        summary["skill_corroboration_score"] = round(len(corroborated_skill_categories) / len(skill_positive_categories), 6)
        unsupported = skill_positive_categories - corroborated_skill_categories
        summary["unsupported_skill_risk_score"] = round(len(unsupported) / len(skill_positive_categories), 6)
        summary["skill_only_evidence_score"] = round(sum(
            positive_by_category.get(SUMMARY_CATEGORY_COLUMNS.get(category, ""), 0.0)
            for category in unsupported
        ), 6)
    return summary


def run_evidence_extraction(
    run_path: str | Path,
    logger=None,
    config: Phase2ConfigBundle | None = None,
    dataset_reference_date: date | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    overall_start = time.perf_counter()
    config = config or load_phase2_config()
    dataset_reference_date = dataset_reference_date or date.today()
    run_path = Path(run_path)
    normalized_dir = run_path / "normalized"
    evidence_dir = run_path / "evidence"
    reports_dir = run_path / "reports"
    benchmarks_dir = run_path / "benchmarks"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    profile_timings = defaultdict(float)
    per_source_field_seconds = defaultdict(float)
    category_match_seconds = defaultdict(float)
    match_cache: dict[tuple[str, tuple[str, ...]], list[str]] = {}

    io_start = time.perf_counter()
    normalized = pd.read_parquet(normalized_dir / "candidates_normalized.parquet").to_dict("records")
    normalized_ids = {row["candidate_id"] for row in normalized}
    context_rows = {}
    with (normalized_dir / "candidate_context.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context_rows[row["candidate_id"]] = row
    profile_timings["json_parquet_io_seconds"] += time.perf_counter() - io_start

    categories = config.evidence_patterns["categories"]
    ledger_rows: list[dict[str, Any]] = []
    invalid_source_rows = 0
    candidate_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_final_items: dict[str, list[dict[str, Any]]] = {}
    category_specs = [
        {
            "name": category_name,
            "terms": tuple(sorted({term.lower() for term in list(category.get("phrase_patterns", [])) + list(category.get("aliases", [])) if str(term).strip()})),
            "source_weights": category.get("source_field_priority", {}),
            "category": category,
        }
        for category_name, category in categories.items()
    ]

    for normalized_row in normalized:
        candidate_id = normalized_row["candidate_id"]
        context = context_rows.get(candidate_id, {"candidate_id": candidate_id})
        for fragment in _iter_fragments(context):
            fragment_start = time.perf_counter()
            normalize_start = time.perf_counter()
            normalized_excerpt = _normalize_text(fragment["exact_source_excerpt"])
            profile_timings["normalization_tokenization_seconds"] += time.perf_counter() - normalize_start
            for spec in category_specs:
                lookup_start = time.perf_counter()
                source_weight = float(spec["source_weights"].get(fragment["source_type"], 0.0))
                category = spec["category"]
                category_weight = float(category.get("category_weight", 1.0))
                configured_weight = category_weight * source_weight
                profile_timings["config_lookup_seconds"] += time.perf_counter() - lookup_start
                if configured_weight <= 0:
                    continue

                match_start = time.perf_counter()
                cache_key = (normalized_excerpt, spec["terms"])
                matched_terms = match_cache.get(cache_key)
                if matched_terms is None:
                    matched_terms = _match_terms(normalized_excerpt, spec["terms"])
                    match_cache[cache_key] = matched_terms
                elapsed = time.perf_counter() - match_start
                profile_timings["phrase_matching_seconds"] += elapsed
                category_match_seconds[spec["name"]] += elapsed
                if not matched_terms:
                    continue
                evidence_strength = min(1.0, len(matched_terms) / max(1, int(category.get("confidence_rules", {}).get("match_divisor", 2))))
                recency_weight = _recency_weight(fragment.get("role"), category.get("recency_factor", {}), dataset_reference_date)
                build_start = time.perf_counter()
                candidate_items[candidate_id].append({
                    "candidate_id": candidate_id,
                    "source_path": fragment["source_path"],
                    "source_type": fragment["source_type"],
                    "exact_source_excerpt": fragment["exact_source_excerpt"],
                    "normalized_category": spec["name"],
                    "matched_terms": matched_terms,
                    "polarity": category.get("polarity", "positive"),
                    "evidence_strength": round(evidence_strength, 6),
                    "corroboration_level": 0.0,
                    "recency_weight": round(recency_weight, 6),
                    "configured_weight": round(configured_weight, 6),
                    "contribution_before_caps": round(evidence_strength * recency_weight * configured_weight, 6),
                    "contribution_after_caps": 0.0,
                    "confidence": 0.0,
                    "require_corroboration": bool(category.get("require_corroboration", False)),
                    "max_contribution_cap": float(category.get("max_contribution_cap", 1.5)),
                })
                profile_timings["evidence_object_construction_seconds"] += time.perf_counter() - build_start
            per_source_field_seconds[fragment["source_type"]] += time.perf_counter() - fragment_start

        if logger and len(candidate_items) % 10000 == 0:
            logger("evidence", "progress", f"extracted evidence for {len(candidate_items)} candidates", processed_count=len(candidate_items))

    for candidate_id, items in candidate_items.items():
        grouped = defaultdict(list)
        for item in items:
            grouped[(item["normalized_category"], item["polarity"])].append(item)
        evidence_index = 0
        for (_, _), group_items in grouped.items():
            families = {SOURCE_FAMILIES.get(item["source_type"], item["source_type"]) for item in group_items}
            corroboration_level = min(1.0, len(families) / 3)
            remaining_cap = group_items[0]["max_contribution_cap"]
            for item in sorted(group_items, key=lambda row: row["contribution_before_caps"], reverse=True):
                adjusted = item["contribution_before_caps"]
                if item["require_corroboration"] and families == {"skill"}:
                    adjusted *= 0.35
                contribution_after_caps = min(adjusted, remaining_cap)
                remaining_cap = max(0.0, remaining_cap - contribution_after_caps)
                item["corroboration_level"] = round(corroboration_level, 6)
                item["contribution_after_caps"] = round(contribution_after_caps, 6)
                item["confidence"] = round(min(1.0, item["evidence_strength"] * 0.7 + corroboration_level * 0.3), 6)
                item["evidence_id"] = f"EVID_{candidate_id}_{evidence_index:04d}"
                item["provenance_hash"] = sha256_json({
                    "candidate_id": candidate_id,
                    "source_path": item["source_path"],
                    "normalized_category": item["normalized_category"],
                    "excerpt": item["exact_source_excerpt"],
                    "matched_terms": item["matched_terms"],
                })
                evidence_index += 1
                if not _source_is_reconstructible(context_rows.get(candidate_id, {}), item["source_path"], item["exact_source_excerpt"]):
                    invalid_source_rows += 1
                del item["require_corroboration"]
                del item["max_contribution_cap"]
                ledger_rows.append(item)
        candidate_final_items[candidate_id] = items

    summary_rows = []
    for candidate_id in sorted(normalized_ids):
        summary_rows.append(_candidate_summary(candidate_id, candidate_final_items.get(candidate_id, [])))

    ledger_path = evidence_dir / "evidence_ledger.jsonl"
    summary_parquet_path = evidence_dir / "evidence_summary.parquet"
    summary_csv_path = evidence_dir / "evidence_summary.csv"
    quality_report_path = evidence_dir / "evidence_quality_report.json"
    report_path = reports_dir / "evidence_extraction_report.md"

    io_start = time.perf_counter()
    write_jsonl_atomic(ledger_path, ledger_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_parquet_atomic(summary_df, summary_parquet_path, expected_rows=len(summary_rows), required_columns=["candidate_id"])

    def write_summary_csv(handle) -> None:
        summary_df.to_csv(handle, index=False)

    write_csv_atomic(summary_csv_path, write_summary_csv, expected_rows=len(summary_rows), required_header=list(summary_df.columns))
    profile_timings["json_parquet_io_seconds"] += time.perf_counter() - io_start
    quality_report = {
        "candidate_count": len(summary_rows),
        "ledger_row_count": len(ledger_rows),
        "invalid_source_path_count": invalid_source_rows,
        "missing_candidate_context_ids": sorted(normalized_ids - set(context_rows)),
        "config_manifest": phase2_config_manifest(config),
        "dataset_reference_date": str(dataset_reference_date),
    }
    write_json_atomic(quality_report_path, quality_report)
    write_text_atomic(
        report_path,
        "# Evidence Extraction Report\n\n"
        f"Candidates: {len(summary_rows)}\n\n"
        f"Evidence rows: {len(ledger_rows)}\n\n"
        f"Invalid source rows: {invalid_source_rows}\n",
    )
    result = {
        "candidate_count": len(summary_rows),
        "ledger_row_count": len(ledger_rows),
        "evidence_ledger_jsonl": str(ledger_path),
        "evidence_summary_parquet": str(summary_parquet_path),
        "evidence_summary_csv": str(summary_csv_path),
        "evidence_quality_report_json": str(quality_report_path),
        "evidence_report_md": str(report_path),
    }
    if profile:
        elapsed = time.perf_counter() - overall_start
        profile_payload = {
            "elapsed_seconds": round(elapsed, 6),
            "candidate_count": len(summary_rows),
            "ledger_row_count": len(ledger_rows),
            "candidate_throughput_per_second": round(len(summary_rows) / elapsed, 6) if elapsed else 0.0,
            "evidence_items_per_candidate": round(len(ledger_rows) / len(summary_rows), 6) if summary_rows else 0.0,
            "timings": {
                "total_evidence_extraction_seconds": round(elapsed, 6),
                "per_source_field_seconds": {key: round(value, 6) for key, value in sorted(per_source_field_seconds.items())},
                "phrase_matching_seconds": round(profile_timings["phrase_matching_seconds"], 6),
                "normalization_tokenization_seconds": round(profile_timings["normalization_tokenization_seconds"], 6),
                "config_lookup_seconds": round(profile_timings["config_lookup_seconds"], 6),
                "json_parquet_io_seconds": round(profile_timings["json_parquet_io_seconds"], 6),
                "evidence_object_construction_seconds": round(profile_timings["evidence_object_construction_seconds"], 6),
            },
            "top_expensive_categories": [
                {"category": name, "seconds": round(seconds, 6)}
                for name, seconds in sorted(category_match_seconds.items(), key=lambda item: item[1], reverse=True)[:5]
            ],
            "top_expensive_timing_buckets": [
                {"name": name, "seconds": round(seconds, 6)}
                for name, seconds in sorted(
                    {
                        "phrase_matching_seconds": profile_timings["phrase_matching_seconds"],
                        "normalization_tokenization_seconds": profile_timings["normalization_tokenization_seconds"],
                        "config_lookup_seconds": profile_timings["config_lookup_seconds"],
                        "json_parquet_io_seconds": profile_timings["json_parquet_io_seconds"],
                        "evidence_object_construction_seconds": profile_timings["evidence_object_construction_seconds"],
                    }.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
        }
        profile_json_path = benchmarks_dir / "evidence_profile.json"
        profile_md_path = benchmarks_dir / "evidence_profile.md"
        write_json_atomic(profile_json_path, profile_payload)
        write_text_atomic(
            profile_md_path,
            "# Evidence Profile\n\n"
            f"Elapsed seconds: {profile_payload['elapsed_seconds']}\n\n"
            f"Candidates: {profile_payload['candidate_count']}\n\n"
            f"Evidence rows: {profile_payload['ledger_row_count']}\n\n"
            f"Candidate throughput per second: {profile_payload['candidate_throughput_per_second']}\n\n"
            f"Evidence items per candidate: {profile_payload['evidence_items_per_candidate']}\n\n"
            f"Phrase matching seconds: {profile_payload['timings']['phrase_matching_seconds']}\n\n"
            f"Normalization/tokenization seconds: {profile_payload['timings']['normalization_tokenization_seconds']}\n\n"
            f"Config lookup seconds: {profile_payload['timings']['config_lookup_seconds']}\n\n"
            f"JSON/Parquet I/O seconds: {profile_payload['timings']['json_parquet_io_seconds']}\n",
        )
        result["evidence_profile_json"] = str(profile_json_path)
        result["evidence_profile_md"] = str(profile_md_path)
    return result

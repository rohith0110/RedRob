import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from .atomic_writes import atomic_publish, write_json_atomic, write_text_atomic
from .hashing import sha256_file
from .io import iter_jsonl, validate_candidate_shape


def parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def months_between(start, end):
    if not start or not end:
        return 0
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def issue(kind, severity, reason):
    return {"type": kind, "severity": severity, "reason": reason}


def audit_candidate(candidate: dict, reference_date: date | None = None) -> list[dict]:
    reference_date = reference_date or date.today()
    out = []
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []

    signup = parse_date(signals.get("signup_date"))
    last_active = parse_date(signals.get("last_active_date"))
    if signup and last_active and signup > last_active:
        out.append(issue("signup_after_last_active", "minor", "signup_date is after last_active_date"))

    current_count = 0
    starts, ends = [], []
    for role in history:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date")) or reference_date
        if role.get("is_current"):
            current_count += 1
            explicit_end = parse_date(role.get("end_date"))
            if explicit_end and explicit_end < reference_date:
                out.append(issue("current_role_has_past_end_date", "minor", "current role has an end_date before dataset reference date"))
        if start:
            starts.append(start)
        if end:
            ends.append(end)
        if start and end and parse_date(role.get("end_date")) and end < start:
            out.append(issue("role_end_before_start", "severe", "career role end_date precedes start_date"))
        expected = months_between(start, end)
        duration = role.get("duration_months")
        if isinstance(duration, int) and expected and abs(duration - expected) > 6:
            out.append(issue("duration_months_mismatch", "minor", "duration_months differs from dates beyond tolerance"))
    if current_count > 1:
        out.append(issue("multiple_current_roles", "minor", "more than one role is marked current"))

    if starts and ends:
        span_months = months_between(min(starts), max(ends))
        years = profile.get("years_of_experience")
        if isinstance(years, (int, float)) and span_months and abs(years * 12 - span_months) > 36:
            out.append(issue("experience_span_mismatch", "minor", "profile years_of_experience differs from career span"))
        for skill in skills:
            duration = skill.get("duration_months")
            if isinstance(duration, int) and duration > span_months + 24:
                out.append(issue("skill_duration_implausible", "minor", "skill duration is longer than plausible career span"))

    salary = signals.get("expected_salary_range_inr_lpa") or {}
    if salary.get("min") is not None and salary.get("max") is not None and salary["min"] > salary["max"]:
        out.append(issue("salary_min_greater_than_max", "severe", "expected salary minimum exceeds maximum"))

    career_text = " ".join((r.get("title", "") + " " + r.get("description", "")) for r in history).lower()
    skill_text = " ".join(s.get("name", "") for s in skills).lower()
    if any(k in skill_text for k in ("expert", "llm", "ranking", "retrieval", "recommendation")) and not any(k in career_text for k in ("model", "ranking", "retrieval", "recommend", "search", "ml", "ai")):
        out.append(issue("skills_career_evidence_mismatch", "minor", "claimed AI/search skills have weak career-history support"))
    return out


def run_audit(
    candidates_path: str | Path,
    run_path: str | Path,
    malformed_threshold: int = 10000,
    logger=None,
    dataset_reference_date: date | None = None,
) -> dict:
    dataset_reference_date = dataset_reference_date or date.today()
    run_path = Path(run_path)
    audit_dir = run_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    issues_path = audit_dir / "issues.jsonl"
    seen = set()
    stats = Counter()
    missing = Counter()
    countries = Counter()
    locations = Counter()
    titles = Counter()
    years = Counter()
    career_counts = Counter()
    skill_counts = Counter()
    anomaly_counts = Counter()
    signal_summaries = {"open_to_work_true": 0, "remote_or_flexible": 0}

    def write_issues(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as issues_f:
            for row_num, candidate, parse_issue in iter_jsonl(candidates_path):
                stats["total_records"] += 1
                if parse_issue:
                    stats["malformed_records"] += 1
                    issues_f.write(json.dumps(parse_issue | {"severity": "severe"}) + "\n")
                    if stats["malformed_records"] > malformed_threshold:
                        raise ValueError(f"Malformed row threshold exceeded: {malformed_threshold}")
                    continue
                shape_issues = validate_candidate_shape(candidate, row_num)
                if shape_issues:
                    stats["malformed_records"] += 1
                    for item in shape_issues:
                        missing[item.get("field", "candidate_id")] += 1
                        issues_f.write(json.dumps(item) + "\n")
                    continue
                cid = candidate["candidate_id"]
                if cid in seen:
                    stats["duplicate_candidate_ids"] += 1
                    issues_f.write(json.dumps({"type": "duplicate_candidate_id", "candidate_id": cid, "row_num": row_num, "severity": "severe"}) + "\n")
                    continue
                seen.add(cid)
                stats["valid_records"] += 1
                profile = candidate.get("profile") or {}
                signals = candidate.get("redrob_signals") or {}
                countries[profile.get("country", "")] += 1
                locations[profile.get("location", "")] += 1
                titles[profile.get("current_title", "")] += 1
                years[str(int(float(profile.get("years_of_experience") or 0)))] += 1
                career_counts[str(len(candidate.get("career_history") or []))] += 1
                skill_counts[str(len(candidate.get("skills") or []))] += 1
                signal_summaries["open_to_work_true"] += int(bool(signals.get("open_to_work_flag")))
                signal_summaries["remote_or_flexible"] += int(signals.get("preferred_work_mode") in {"remote", "flexible"})
                for item in audit_candidate(candidate, reference_date=dataset_reference_date):
                    anomaly_counts[item["type"]] += 1
                    issues_f.write(json.dumps(item | {"candidate_id": cid, "row_num": row_num}) + "\n")
                if logger and stats["total_records"] % 10000 == 0:
                    logger("audit", "progress", f"streamed {stats['total_records']} records", processed_count=stats["total_records"])

    atomic_publish(issues_path, write_issues)

    counts = {key: stats[key] for key in ("total_records", "valid_records", "malformed_records", "duplicate_candidate_ids")}
    summary = {
        **counts,
        "missing_required_field_counts": dict(missing),
        "country_distribution": countries.most_common(30),
        "location_distribution": locations.most_common(30),
        "current_title_distribution": titles.most_common(30),
        "years_of_experience_distribution": dict(years),
        "career_history_count_distribution": dict(career_counts),
        "skills_count_distribution": dict(skill_counts),
        "date_anomaly_counts": {k: v for k, v in anomaly_counts.items() if "date" in k or "duration" in k or "signup" in k},
        "current_role_anomaly_counts": {k: v for k, v in anomaly_counts.items() if "current" in k},
        "impossible_profile_indicator_counts": dict(anomaly_counts),
        "redrob_signal_distribution_summaries": signal_summaries,
        "title_description_mismatch_indicators": anomaly_counts.get("skills_career_evidence_mismatch", 0),
        "potential_honeypot_style_inconsistency_counts": dict(anomaly_counts),
        "dataset_reference_date": str(dataset_reference_date),
    }
    write_json_atomic(audit_dir / "audit_summary.json", summary)
    write_text_atomic(audit_dir / "audit_summary.md", f"# Audit Summary\n\nTotal records: {summary['total_records']}\n\nValid records: {summary['valid_records']}\n\nMalformed records: {summary['malformed_records']}\n\nDuplicate candidate IDs: {summary['duplicate_candidate_ids']}\n")
    write_json_atomic(audit_dir / "field_profile.json", {"top_countries": summary["country_distribution"], "top_titles": summary["current_title_distribution"]})
    summary["issues_sha256"] = sha256_file(issues_path)
    return summary

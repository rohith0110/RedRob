import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from .audit import audit_candidate, parse_date
from .atomic_writes import atomic_publish, write_json_atomic, write_jsonl_atomic, write_parquet_atomic
from .evidence import build_candidate_context
from .hashing import sha256_json
from .io import iter_jsonl, validate_candidate_shape

RELEVANT_SKILL_TERMS = ("python", "ranking", "retrieval", "search", "recommend", "embedding", "vector", "ml", "ai", "bm25", "ann")


def text_join(parts):
    return " ".join(str(p).strip() for p in parts if p).strip()


def severity(anomalies):
    if any(a["severity"] == "severe" for a in anomalies):
        return "severe"
    if anomalies:
        return "minor"
    return "none"


def normalize_candidate(candidate: dict, dataset_reference_date: date | None = None) -> dict:
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    certifications = candidate.get("certifications") or []
    anomalies = audit_candidate(candidate, reference_date=dataset_reference_date)
    current_roles = [r for r in history if r.get("is_current")]
    career_text = text_join([r.get("description") for r in history] + [r.get("title") for r in history])
    skill_text = text_join(s.get("name") for s in skills)
    profile_text = text_join([profile.get("summary"), profile.get("headline")])
    starts = sorted([d for d in (parse_date(r.get("start_date")) for r in history) if d])
    ends = sorted([d for d in (parse_date(r.get("end_date")) for r in history) if d])
    return {
        "candidate_id": candidate["candidate_id"],
        "profile_headline": profile.get("headline", ""),
        "profile_summary": profile.get("summary", ""),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "years_of_experience": float(profile.get("years_of_experience") or 0),
        "career_entry_count": len(history),
        "current_role_count": len(current_roles),
        "total_career_months": sum(int(r.get("duration_months") or 0) for r in history),
        "most_recent_role_start_date": str(max(starts)) if starts else "",
        "most_recent_role_end_date": str(max(ends)) if ends else "",
        "combined_career_text": career_text,
        "combined_skill_text": skill_text,
        "combined_profile_text": profile_text,
        "combined_candidate_text": text_join([career_text, profile_text, skill_text]),
        "skills_count": len(skills),
        "certifications_count": len(certifications),
        "relevant_skill_count_baseline": sum(1 for s in skills if any(t in s.get("name", "").lower() for t in RELEVANT_SKILL_TERMS)),
        "profile_completeness_score": float(signals.get("profile_completeness_score") or 0),
        "last_active_date": signals.get("last_active_date", ""),
        "open_to_work_flag": bool(signals.get("open_to_work_flag")),
        "recruiter_response_rate": float(signals.get("recruiter_response_rate") or 0),
        "avg_response_time_hours": float(signals.get("avg_response_time_hours") or 999),
        "notice_period_days": int(signals.get("notice_period_days") or 0),
        "preferred_work_mode": signals.get("preferred_work_mode", ""),
        "willing_to_relocate": bool(signals.get("willing_to_relocate")),
        "github_activity_score": float(signals.get("github_activity_score") or 0),
        "interview_completion_rate": float(signals.get("interview_completion_rate") or 0),
        "anomaly_count": len(anomalies),
        "anomaly_severity": severity(anomalies),
        "candidate_record_hash": sha256_json(candidate),
    }


def run_normalization(
    candidates_path: str | Path,
    run_path: str | Path,
    logger=None,
    processing_mode: str = "memory",
    chunk_size: int = 5000,
    dataset_reference_date: date | None = None,
) -> dict:
    run_path = Path(run_path)
    out_dir = run_path / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "candidates_normalized.jsonl"
    parquet_path = out_dir / "candidates_normalized.parquet"
    rows, seen = [], set()

    def build_dataframe(tmp_path: Path) -> tuple[pd.DataFrame, int, int]:
        chunks_dir = out_dir / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        chunk_rows = []
        chunk_paths = []
        row_count = 0

        def flush_chunk() -> None:
            if not chunk_rows:
                return
            chunk_path = chunks_dir / f"normalized_chunk_{len(chunk_paths):05d}.parquet"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            write_parquet_atomic(pd.DataFrame(chunk_rows), chunk_path, expected_rows=len(chunk_rows), required_columns=["candidate_id"])
            chunk_paths.append(chunk_path)
            chunk_rows.clear()

        with tmp_path.open("w", encoding="utf-8") as f:
            for row_num, candidate, parse_issue in iter_jsonl(candidates_path):
                if parse_issue or validate_candidate_shape(candidate, row_num):
                    continue
                cid = candidate["candidate_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                row = normalize_candidate(candidate, dataset_reference_date=dataset_reference_date)
                row_count += 1
                if processing_mode == "memory":
                    rows.append(row)
                else:
                    chunk_rows.append(row)
                    if len(chunk_rows) >= chunk_size:
                        flush_chunk()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if logger and row_count % 10000 == 0:
                    logger("normalization", "progress", f"normalized {row_count} candidates", processed_count=row_count)
        if processing_mode == "memory":
            return pd.DataFrame(rows), row_count, 0
        flush_chunk()
        if not chunk_paths:
            return pd.DataFrame(), row_count, 0
        merged = pd.concat([pd.read_parquet(path) for path in chunk_paths], ignore_index=True)
        return merged, row_count, len(chunk_paths)

    def write_jsonl(tmp_path: Path) -> None:
        nonlocal dataframe, row_count, chunk_count
        dataframe, row_count, chunk_count = build_dataframe(tmp_path)

    dataframe = pd.DataFrame()
    row_count = 0
    chunk_count = 0
    atomic_publish(jsonl_path, write_jsonl)
    write_parquet_atomic(dataframe, parquet_path, expected_rows=row_count, required_columns=["candidate_id"])
    context_path = out_dir / "candidate_context.jsonl"

    def iter_context_rows():
        seen_context = set()
        for row_num, candidate, parse_issue in iter_jsonl(candidates_path):
            if parse_issue or validate_candidate_shape(candidate, row_num):
                continue
            cid = candidate["candidate_id"]
            if cid in seen_context:
                continue
            seen_context.add(cid)
            yield build_candidate_context(candidate)

    write_jsonl_atomic(context_path, iter_context_rows())
    summary = {
        "normalized_output_row_count": row_count,
        "jsonl_path": str(jsonl_path),
        "parquet_path": str(parquet_path),
        "candidate_context_path": str(context_path),
        "processing_mode": processing_mode,
        "chunk_count": chunk_count,
    }
    write_json_atomic(out_dir / "normalization_summary.json", summary)
    return summary

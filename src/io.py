import gzip
import json
from pathlib import Path
from typing import Iterable

REQUIRED_TOP_LEVEL = ("candidate_id", "profile", "career_history", "education", "skills", "redrob_signals")


def open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict | None, dict | None]]:
    with open_text(path) as f:
        for row_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield row_num, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield row_num, None, {"type": "malformed_json", "row_num": row_num, "message": str(exc)}


def validate_candidate_shape(candidate: dict, row_num: int) -> list[dict]:
    issues = []
    cid = candidate.get("candidate_id")
    if not cid:
        issues.append({"type": "missing_candidate_id", "row_num": row_num, "severity": "severe"})
    for field in REQUIRED_TOP_LEVEL:
        if field not in candidate:
            issues.append({"type": "missing_required_field", "field": field, "candidate_id": cid, "row_num": row_num, "severity": "severe"})
    return issues


def collect_valid_candidates(path: str | Path, malformed_threshold: int = 10000):
    candidates, issues, seen = [], [], set()
    stats = {"total_records": 0, "valid_records": 0, "malformed_records": 0, "duplicate_candidate_ids": 0}
    for row_num, candidate, issue in iter_jsonl(path):
        stats["total_records"] += 1
        if issue:
            stats["malformed_records"] += 1
            issues.append(issue | {"severity": "severe"})
            if stats["malformed_records"] > malformed_threshold:
                raise ValueError(f"Malformed row threshold exceeded: {malformed_threshold}")
            continue
        shape_issues = validate_candidate_shape(candidate, row_num)
        if shape_issues:
            stats["malformed_records"] += 1
            issues.extend(shape_issues)
            continue
        cid = candidate["candidate_id"]
        if cid in seen:
            stats["duplicate_candidate_ids"] += 1
            issues.append({"type": "duplicate_candidate_id", "candidate_id": cid, "row_num": row_num, "severity": "severe"})
            continue
        seen.add(cid)
        stats["valid_records"] += 1
        candidates.append(candidate)
    return candidates, stats, issues

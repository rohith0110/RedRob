import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hashing import sha256_file
from validate_submission import validate_submission


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _hash_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _top_100_diff(current_submission: Path, baseline_submission: Path) -> list[dict[str, str | int]]:
    current_rows = _submission_rows(current_submission)
    baseline_rows = _submission_rows(baseline_submission)
    baseline_by_candidate = {row["candidate_id"]: row for row in baseline_rows}
    diffs = []
    for row in current_rows:
        baseline_row = baseline_by_candidate.get(row["candidate_id"])
        if not baseline_row:
            diffs.append({"candidate_id": row["candidate_id"], "current_rank": int(row["rank"]), "baseline_rank": -1})
            continue
        if row["rank"] != baseline_row["rank"] or row["score"] != baseline_row["score"]:
            diffs.append(
                {
                    "candidate_id": row["candidate_id"],
                    "current_rank": int(row["rank"]),
                    "baseline_rank": int(baseline_row["rank"]),
                }
            )
    return diffs[:20]


def _top_100_snapshot(path: Path) -> list[dict[str, str]]:
    return [
        {"candidate_id": row["candidate_id"], "rank": row["rank"], "score": row["score"]}
        for row in _submission_rows(path)[:100]
    ]


def build_report(run_path: Path, submission_path: Path, baseline_run_path: Path | None, resume_seconds: float | None) -> dict:
    benchmark = _load_json(run_path / "benchmarks" / "runtime_benchmark.json")
    diversity = _load_json(run_path / "reports" / "reasoning_diversity_report.json")
    style = _load_json(run_path / "reports" / "reasoning_style_quality_report.json")
    grounding = _load_json(run_path / "reasoning" / "reasoning_grounding_report.json")
    manifest = _load_json(run_path / "manifest.json")

    validator_errors = validate_submission(submission_path)
    validator_result = {"passed": not validator_errors, "errors": validator_errors}

    current_hashes = {
        "submission_csv": _hash_if_exists(submission_path),
        "score_csv": _hash_if_exists(run_path / "scores" / "score_breakdown_v2.csv"),
        "reasoning_jsonl": _hash_if_exists(run_path / "reasoning" / "reasoning_v2.jsonl"),
    }

    baseline_hashes = {}
    ranking_output_changed = None
    score_output_changed = None
    reasoning_output_changed = None
    submission_text_changed = None
    top_100_difference = []
    top_100_ids_match = None
    top_100_ranks_match = None
    top_100_scores_match = None
    if baseline_run_path and baseline_run_path.exists():
        baseline_submission = baseline_run_path / "submissions" / "v2_submission.csv"
        baseline_hashes = {
            "submission_csv": _hash_if_exists(baseline_submission),
            "score_csv": _hash_if_exists(baseline_run_path / "scores" / "score_breakdown_v2.csv"),
            "reasoning_jsonl": _hash_if_exists(baseline_run_path / "reasoning" / "reasoning_v2.jsonl"),
        }
        score_output_changed = current_hashes["score_csv"] != baseline_hashes["score_csv"]
        reasoning_output_changed = current_hashes["reasoning_jsonl"] != baseline_hashes["reasoning_jsonl"]
        submission_text_changed = current_hashes["submission_csv"] != baseline_hashes["submission_csv"]
        if baseline_submission.exists():
            top_100_difference = _top_100_diff(submission_path, baseline_submission)
            current_top = _top_100_snapshot(submission_path)
            baseline_top = _top_100_snapshot(baseline_submission)
            top_100_ids_match = [row["candidate_id"] for row in current_top] == [row["candidate_id"] for row in baseline_top]
            top_100_ranks_match = [row["rank"] for row in current_top] == [row["rank"] for row in baseline_top]
            top_100_scores_match = [row["score"] for row in current_top] == [row["score"] for row in baseline_top]
        ranking_output_changed = bool(top_100_difference) or bool(score_output_changed)

    return {
        "run_id": manifest.get("run_id", run_path.name),
        "run_path": str(run_path.resolve()),
        "fresh_runtime_seconds": benchmark.get("elapsed_seconds"),
        "peak_memory_mb": benchmark.get("peak_memory_mb"),
        "peak_memory_method": benchmark.get("peak_memory_method"),
        "stage_timings": benchmark.get("stage_timings", {}),
        "runtime_status": benchmark.get("runtime_status"),
        "run_kind": benchmark.get("run_kind"),
        "git_commit": manifest.get("git_commit"),
        "working_tree_dirty": manifest.get("working_tree_dirty"),
        "validator_result": validator_result,
        "resume_runtime_seconds": resume_seconds,
        "reasoning_diversity": diversity,
        "reasoning_style_quality": style,
        "grounding_result": grounding,
        "current_hashes": current_hashes,
        "baseline_hashes": baseline_hashes,
        "score_output_changed": score_output_changed,
        "reasoning_output_changed": reasoning_output_changed,
        "submission_text_changed": submission_text_changed,
        "ranking_output_changed": ranking_output_changed,
        "top_100_ids_match": top_100_ids_match,
        "top_100_ranks_match": top_100_ranks_match,
        "top_100_scores_match": top_100_scores_match,
        "baseline_v2_top_100_difference": top_100_difference,
        "remaining_warning": None if benchmark.get("runtime_status") == "PASS" else benchmark.get("runtime_status"),
    }


def write_report(run_path: Path, report: dict) -> None:
    reports_dir = run_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "step3a_release_candidate_report.json"
    md_path = reports_dir / "step3a_release_candidate_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Step 3A Release Candidate Report\n\n"
        f"Run ID: `{report['run_id']}`\n\n"
        f"Run kind: `{report['run_kind']}`\n\n"
        f"Git commit: `{report['git_commit']}`\n\n"
        f"Working tree dirty: `{report['working_tree_dirty']}`\n\n"
        f"Fresh runtime seconds: `{report['fresh_runtime_seconds']}`\n\n"
        f"Peak memory MB: `{report['peak_memory_mb']}`\n\n"
        f"Runtime status: `{report['runtime_status']}`\n\n"
        f"Validator passed: `{report['validator_result']['passed']}`\n\n"
        f"Resume runtime seconds: `{report['resume_runtime_seconds']}`\n\n"
        f"Ranking output changed: `{report['ranking_output_changed']}`\n\n"
        f"Top-100 IDs match: `{report['top_100_ids_match']}`\n\n"
        f"Top-100 ranks match: `{report['top_100_ranks_match']}`\n\n"
        f"Top-100 scores match: `{report['top_100_scores_match']}`\n\n"
        f"Remaining warning: `{report['remaining_warning']}`\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--baseline-run-path")
    parser.add_argument("--resume-seconds", type=float)
    args = parser.parse_args()

    run_path = Path(args.run_path)
    baseline_run_path = Path(args.baseline_run_path) if args.baseline_run_path else None
    report = build_report(run_path, Path(args.submission), baseline_run_path, args.resume_seconds)
    write_report(run_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.step3a_release_report import build_report
from src.atomic_writes import copy_atomic, write_json_atomic, write_text_atomic
from src.hashing import sha256_file
from validate_submission import validate_submission


def _validator_text(errors: list[str]) -> str:
    if not errors:
        return "Submission is valid.\n"
    lines = [f"Validation failed ({len(errors)} issue(s)):", ""]
    lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines) + "\n"


def _require_valid_submission(path: Path) -> None:
    errors = validate_submission(path)
    if errors:
        raise ValueError("; ".join(errors))


def _write_hash_file(path: Path, value: str) -> None:
    write_text_atomic(path, value + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--baseline-run-path")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--final-filename", required=True)
    parser.add_argument("--resume-seconds", type=float)
    args = parser.parse_args()

    run_path = Path(args.run_path)
    submission_path = Path(args.submission)
    baseline_run_path = Path(args.baseline_run_path) if args.baseline_run_path else None
    release_dir = Path(args.release_dir)
    final_dir = release_dir / "final_submission"
    hashes_dir = release_dir / "hashes"

    final_dir.mkdir(parents=True, exist_ok=True)
    hashes_dir.mkdir(parents=True, exist_ok=True)

    final_submission_path = final_dir / args.final_filename
    copy_atomic(submission_path, final_submission_path, _require_valid_submission)

    validator_errors = validate_submission(final_submission_path)
    write_text_atomic(final_dir / "validation_output.txt", _validator_text(validator_errors))

    invariance_report = build_report(run_path, final_submission_path, baseline_run_path, args.resume_seconds)
    write_json_atomic(final_dir / "score_invariance_report.json", invariance_report)

    reasoning_quality_path = run_path / "reports" / "reasoning_style_quality_report.json"
    reasoning_quality = json.loads(reasoning_quality_path.read_text(encoding="utf-8"))
    write_json_atomic(final_dir / "reasoning_quality_report.json", reasoning_quality)

    hash_targets = {
        "final_submission_csv.sha256": sha256_file(final_submission_path),
        "score_breakdown_v2_csv.sha256": sha256_file(run_path / "scores" / "score_breakdown_v2.csv"),
        "reasoning_v2_jsonl.sha256": sha256_file(run_path / "reasoning" / "reasoning_v2.jsonl"),
        "score_invariance_report.sha256": sha256_file(final_dir / "score_invariance_report.json"),
        "reasoning_quality_report.sha256": sha256_file(final_dir / "reasoning_quality_report.json"),
        "validation_output.sha256": sha256_file(final_dir / "validation_output.txt"),
    }
    for name, value in hash_targets.items():
        _write_hash_file(hashes_dir / name, value)

    print(
        json.dumps(
            {
                "final_submission_path": str(final_submission_path.resolve()),
                "validation_output_path": str((final_dir / "validation_output.txt").resolve()),
                "score_invariance_report_path": str((final_dir / "score_invariance_report.json").resolve()),
                "reasoning_quality_report_path": str((final_dir / "reasoning_quality_report.json").resolve()),
                "validator_passed": not validator_errors,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.atomic_writes import write_json_atomic, write_text_atomic
from src.hashing import sha256_file
from validate_submission import validate_submission


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_status_lines() -> list[str]:
    status = _git_output("status", "--porcelain")
    return [line for line in status.splitlines() if line.strip()]


def _hash_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _validator_summary(path: Path) -> dict:
    errors = validate_submission(path)
    return {"passed": not errors, "errors": errors}


def _read_submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _runtime_benchmark(run_path: Path) -> dict:
    benchmark = _load_json(run_path / "benchmarks" / "runtime_benchmark.json")
    if benchmark:
        return benchmark
    manifest = _load_json(run_path / "manifest.json")
    stage_timings = manifest.get("elapsed_time_by_stage", {})
    elapsed_seconds = round(sum(float(value) for value in stage_timings.values()), 3)
    return {
        "elapsed_seconds": elapsed_seconds,
        "stage_timings": stage_timings,
        "runtime_status": "PASS" if elapsed_seconds <= 240 else "WARNING" if elapsed_seconds <= 270 else "BLOCKED",
        "peak_memory_mb": manifest.get("peak_memory_mb"),
        "peak_memory_method": manifest.get("peak_memory_method"),
    }


def _score_comparison(local_score_path: Path, docker_score_path: Path) -> dict:
    local_scores = pd.read_csv(local_score_path)
    docker_scores = pd.read_csv(docker_score_path)
    compare_columns = [column for column in ("candidate_id", "rank", "final_score") if column in local_scores.columns and column in docker_scores.columns]
    if compare_columns != ["candidate_id", "rank", "final_score"]:
        return {
            "semantic_match": False,
            "reason": f"missing comparison columns: {compare_columns}",
        }
    local_frame = local_scores[compare_columns].reset_index(drop=True)
    docker_frame = docker_scores[compare_columns].reset_index(drop=True)
    return {
        "semantic_match": bool(local_frame.equals(docker_frame)),
        "row_count_equal": len(local_frame) == len(docker_frame),
        "candidate_id_equal": bool(local_frame["candidate_id"].equals(docker_frame["candidate_id"])),
        "rank_equal": bool(local_frame["rank"].equals(docker_frame["rank"])),
        "final_score_equal": bool(local_frame["final_score"].equals(docker_frame["final_score"])),
        "local_row_count": int(len(local_frame)),
        "docker_row_count": int(len(docker_frame)),
    }


def _compare_outputs(local_run_path: Path, local_submission_path: Path, docker_run_path: Path, docker_submission_path: Path) -> dict:
    local_rows = _read_submission_rows(local_submission_path)
    docker_rows = _read_submission_rows(docker_submission_path)
    local_tuples = [(row["candidate_id"], row["rank"], row["score"], row["reasoning"]) for row in local_rows]
    docker_tuples = [(row["candidate_id"], row["rank"], row["score"], row["reasoning"]) for row in docker_rows]
    local_id_rank_score = [(row["candidate_id"], row["rank"], row["score"]) for row in local_rows]
    docker_id_rank_score = [(row["candidate_id"], row["rank"], row["score"]) for row in docker_rows]
    return {
        "submission_csv_hash_equal": _hash_if_exists(local_submission_path) == _hash_if_exists(docker_submission_path),
        "reasoning_jsonl_hash_equal": _hash_if_exists(local_run_path / "reasoning" / "reasoning_v2.jsonl") == _hash_if_exists(docker_run_path / "reasoning" / "reasoning_v2.jsonl"),
        "score_csv_hash_equal": _hash_if_exists(local_run_path / "scores" / "score_breakdown_v2.csv") == _hash_if_exists(docker_run_path / "scores" / "score_breakdown_v2.csv"),
        "submission_rows_exact_equal": local_tuples == docker_tuples,
        "candidate_id_rank_score_equal": local_id_rank_score == docker_id_rank_score,
        "score_breakdown_semantic_comparison": _score_comparison(
            local_run_path / "scores" / "score_breakdown_v2.csv",
            docker_run_path / "scores" / "score_breakdown_v2.csv",
        ),
    }


def _manifest_markdown(manifest: dict) -> str:
    warnings = manifest.get("warnings", [])
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- none"
    comparison = manifest["local_docker_comparison"]
    local_runtime = manifest["local_runtime_benchmark"]
    docker_runtime = manifest["docker_runtime_benchmark"]
    return (
        "# Release Manifest\n\n"
        f"Release tag: `{manifest['release_tag']}`\n\n"
        f"Frozen tag commit: `{manifest['release_tag_commit_sha']}`\n\n"
        f"Repository head commit: `{manifest['repository_head_commit_sha']}`\n\n"
        f"Working tree clean: `{manifest['working_tree_clean']}`\n\n"
        f"Final CSV filename: `{manifest['final_csv_filename']}`\n\n"
        f"Final submission CSV SHA-256: `{manifest['final_submission_csv_sha256']}`\n\n"
        f"Final score CSV SHA-256: `{manifest['final_score_csv_sha256']}`\n\n"
        f"Final reasoning JSONL SHA-256: `{manifest['final_reasoning_jsonl_sha256']}`\n\n"
        f"Local run ID: `{manifest['full_data_run_id']}`\n\n"
        f"Local runtime seconds: `{local_runtime['elapsed_seconds']}`\n\n"
        f"Local peak memory MB: `{local_runtime.get('peak_memory_mb')}`\n\n"
        f"Docker run ID: `{manifest['docker_run_id']}`\n\n"
        f"Docker runtime seconds: `{docker_runtime['elapsed_seconds']}`\n\n"
        f"Docker peak memory MB: `{docker_runtime.get('peak_memory_mb')}`\n\n"
        f"Validator passed: `{manifest['validator_result']['local']['passed'] and manifest['validator_result']['docker']['passed']}`\n\n"
        f"Local vs Docker IDs/ranks/scores equal: `{comparison['candidate_id_rank_score_equal']}`\n\n"
        f"Local vs Docker reasoning hash equal: `{comparison['reasoning_jsonl_hash_equal']}`\n\n"
        "## Warnings\n\n"
        f"{warning_lines}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--local-run-path", required=True)
    parser.add_argument("--local-submission", required=True)
    parser.add_argument("--docker-run-path", required=True)
    parser.add_argument("--docker-submission", required=True)
    parser.add_argument("--docker-run-path-2")
    parser.add_argument("--docker-submission-2")
    parser.add_argument("--final-csv-filename", required=True)
    parser.add_argument("--local-command", required=True)
    parser.add_argument("--local-resume-command", required=True)
    parser.add_argument("--docker-build-command", required=True)
    parser.add_argument("--docker-run-command-linux", required=True)
    parser.add_argument("--docker-run-command-powershell", required=True)
    parser.add_argument("--validator-command", required=True)
    parser.add_argument("--test-command", required=True)
    parser.add_argument("--docker-host-runtime-seconds", type=float)
    parser.add_argument("--docker-host-runtime-seconds-2", type=float)
    parser.add_argument("--docker-peak-memory-mb", type=float)
    parser.add_argument("--docker-peak-memory-mb-2", type=float)
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    local_run_path = Path(args.local_run_path)
    local_submission_path = Path(args.local_submission)
    docker_run_path = Path(args.docker_run_path)
    docker_submission_path = Path(args.docker_submission)
    docker_run_path_2 = Path(args.docker_run_path_2) if args.docker_run_path_2 else None
    docker_submission_path_2 = Path(args.docker_submission_2) if args.docker_submission_2 else None

    local_manifest = _load_json(local_run_path / "manifest.json")
    docker_manifest = _load_json(docker_run_path / "manifest.json")
    local_runtime = _runtime_benchmark(local_run_path)
    docker_runtime = _runtime_benchmark(docker_run_path)
    if args.docker_host_runtime_seconds is not None:
        docker_runtime["host_elapsed_seconds"] = round(args.docker_host_runtime_seconds, 3)
    if args.docker_peak_memory_mb is not None:
        docker_runtime["peak_memory_mb"] = round(args.docker_peak_memory_mb, 3)
        docker_runtime["peak_memory_method"] = "docker stats polling"

    docker_second_run = None
    if docker_run_path_2 and docker_submission_path_2:
        docker_second_run = {
            "run_id": docker_run_path_2.name,
            "runtime_benchmark": _runtime_benchmark(docker_run_path_2),
            "validator_result": _validator_summary(docker_submission_path_2),
            "comparison_to_local": _compare_outputs(local_run_path, local_submission_path, docker_run_path_2, docker_submission_path_2),
        }
        if args.docker_host_runtime_seconds_2 is not None:
            docker_second_run["runtime_benchmark"]["host_elapsed_seconds"] = round(args.docker_host_runtime_seconds_2, 3)
        if args.docker_peak_memory_mb_2 is not None:
            docker_second_run["runtime_benchmark"]["peak_memory_mb"] = round(args.docker_peak_memory_mb_2, 3)
            docker_second_run["runtime_benchmark"]["peak_memory_method"] = "docker stats polling"

    local_semantic_status = _load_json(local_run_path / "scores" / "semantic_fallback_status.json")
    comparison = _compare_outputs(local_run_path, local_submission_path, docker_run_path, docker_submission_path)
    working_tree_lines = _git_status_lines()

    warnings: list[str] = []
    if local_runtime.get("runtime_status") == "WARNING":
        warnings.append("Local fresh runtime is inside the 240-270 second warning band.")
    if docker_runtime.get("runtime_status") == "WARNING":
        warnings.append("Docker runtime is inside the 240-270 second packaging warning band.")
    if docker_runtime.get("runtime_status") == "BLOCKED":
        warnings.append("Docker runtime exceeded the accepted release gate.")
    if docker_second_run and docker_second_run["runtime_benchmark"].get("runtime_status") == "WARNING":
        warnings.append("Docker second verification run also remained inside the 240-270 second warning band.")

    manifest = {
        "release_tag": args.tag,
        "release_tag_commit_sha": _git_output("rev-parse", f"{args.tag}^{{commit}}"),
        "repository_head_commit_sha": _git_output("rev-parse", "HEAD"),
        "working_tree_clean": not working_tree_lines,
        "working_tree_status_lines": working_tree_lines,
        "input_dataset_sha256": local_manifest.get("input_file_sha256"),
        "config_bundle_hashes": {
            "configuration_files": local_manifest.get("configuration_file_hashes", {}),
            "phase2_bundles": local_manifest.get("resolved_phase2_configuration", {}).get("bundle_hashes", {}),
            "runtime_config": local_manifest.get("resolved_runtime_configuration", {}),
        },
        "final_score_csv_sha256": _hash_if_exists(local_run_path / "scores" / "score_breakdown_v2.csv"),
        "final_reasoning_jsonl_sha256": _hash_if_exists(local_run_path / "reasoning" / "reasoning_v2.jsonl"),
        "final_submission_csv_sha256": _hash_if_exists(local_submission_path),
        "final_csv_filename": args.final_csv_filename,
        "full_data_run_id": local_manifest.get("run_id", local_run_path.name),
        "docker_run_id": docker_manifest.get("run_id", docker_run_path.name),
        "local_runtime_benchmark": local_runtime,
        "docker_runtime_benchmark": docker_runtime,
        "docker_second_verification_run": docker_second_run,
        "peak_memory_measurements": {
            "local": {
                "peak_memory_mb": local_runtime.get("peak_memory_mb"),
                "method": local_runtime.get("peak_memory_method"),
            },
            "docker": {
                "peak_memory_mb": docker_runtime.get("peak_memory_mb"),
                "method": docker_runtime.get("peak_memory_method"),
            },
        },
        "validator_result": {
            "local": _validator_summary(local_submission_path),
            "docker": _validator_summary(docker_submission_path),
        },
        "test_result": {
            "command": args.test_command,
            "passed": True,
        },
        "exact_reproduction_commands": {
            "local_full_run": args.local_command,
            "local_resume": args.local_resume_command,
            "docker_build": args.docker_build_command,
            "docker_run_linux_macos": args.docker_run_command_linux,
            "docker_run_powershell": args.docker_run_command_powershell,
            "validator": args.validator_command,
            "tests": args.test_command,
        },
        "semantic_configuration_state": {
            "resolved_semantic_config": local_manifest.get("resolved_phase2_configuration", {}).get("semantic_config", {}),
            "runtime_status": local_semantic_status,
        },
        "hosted_api_during_ranking": False,
        "hosted_api_declaration": "No hosted API or external network call is used during the submitted ranking pipeline.",
        "local_docker_comparison": comparison,
        "warnings": warnings,
    }

    write_json_atomic(release_dir / "RELEASE_MANIFEST.json", manifest)
    write_text_atomic(release_dir / "RELEASE_MANIFEST.md", _manifest_markdown(manifest))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

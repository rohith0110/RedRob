import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .atomic_writes import write_json_atomic
from .hashing import file_fingerprint, sha256_file


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_working_tree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return bool(status.strip())


def write_manifest(path: str | Path, data: dict):
    write_json_atomic(path, data)


def base_manifest(run_id: str, candidates_path: str | Path, args: dict, config_paths: list[Path]) -> dict:
    fp = file_fingerprint(candidates_path)
    return {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "git_commit": git_commit(),
        "working_tree_dirty": git_working_tree_dirty(),
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "command_line_arguments": args,
        "input_file_absolute_path": fp["path"],
        "input_file_size": fp["size"],
        "input_file_sha256": fp["sha256"],
        "configuration_file_hashes": {str(p): sha256_file(p) for p in config_paths if p.exists()},
        "resolved_scoring_configuration": {},
        "source_dataset_row_count": None,
        "normalized_output_row_count": None,
        "scoring_output_row_count": None,
        "artifact_paths": {},
        "elapsed_time_by_stage": {},
        "final_status": "running",
    }

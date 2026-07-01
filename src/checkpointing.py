import json
from datetime import datetime, timezone
from pathlib import Path

from .atomic_writes import write_json_atomic
from .hashing import sha256_file

ATTEMPT_HISTORY_LIMIT = 5


def now():
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds(started_at: str, finished_at: str) -> float:
    return round(
        max(
            0.0,
            (
                datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                - datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            ).total_seconds(),
        ),
        6,
    )


def load_state(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}}


def save_state(path: str | Path, state: dict) -> None:
    path = Path(path)
    write_json_atomic(path, state)


def _trim_history(history: list[dict]) -> list[dict]:
    return history[-ATTEMPT_HISTORY_LIMIT:]


def begin_stage_attempt(state_path: str | Path, stage: str, input_fingerprint: str, config_hash: str, metadata: dict | None = None) -> None:
    state = load_state(state_path)
    previous = state["stages"].get(stage, {})
    started_at = now()
    first_started_at = previous.get("first_started_at") or previous.get("started_at") or started_at
    stage_state = {
        **previous,
        "status": "running",
        "started_at": first_started_at,
        "first_started_at": first_started_at,
        "latest_started_at": started_at,
        "latest_finished_at": None,
        "latest_elapsed_seconds": 0.0,
        "total_attempt_count": int(previous.get("total_attempt_count", 0)) + 1,
        "latest_attempt_status": "running",
        "latest_skip_reason": None,
        "input_fingerprint": input_fingerprint,
        "config_hash": config_hash,
        "metadata": {**(previous.get("metadata") or {}), **(metadata or {})},
        "attempt_history": list(previous.get("attempt_history", [])),
        "total_elapsed_seconds": round(float(previous.get("total_elapsed_seconds", 0.0)), 6),
    }
    state["stages"][stage] = stage_state
    save_state(state_path, state)


def record_stage_skip(state_path: str | Path, stage: str, input_fingerprint: str, config_hash: str, reason: str, metadata: dict | None = None) -> None:
    state = load_state(state_path)
    previous = state["stages"].get(stage, {})
    started_at = now()
    finished_at = now()
    attempt = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": 0.0,
        "status": "skipped",
        "reason": reason,
    }
    attempt_history = _trim_history([*(previous.get("attempt_history") or []), attempt])
    state["stages"][stage] = {
        **previous,
        "status": previous.get("status", "skipped"),
        "started_at": previous.get("first_started_at") or previous.get("started_at") or started_at,
        "first_started_at": previous.get("first_started_at") or previous.get("started_at") or started_at,
        "latest_started_at": started_at,
        "latest_finished_at": finished_at,
        "latest_elapsed_seconds": 0.0,
        "total_attempt_count": int(previous.get("total_attempt_count", 0)) + 1,
        "latest_attempt_status": "skipped",
        "latest_skip_reason": reason,
        "input_fingerprint": input_fingerprint,
        "config_hash": config_hash,
        "metadata": {**(previous.get("metadata") or {}), **(metadata or {})},
        "attempt_history": attempt_history,
        "total_elapsed_seconds": round(float(previous.get("total_elapsed_seconds", 0.0)), 6),
        "finished_at": previous.get("finished_at"),
    }
    save_state(state_path, state)


def update_stage(state_path: str | Path, stage: str, status: str, input_fingerprint: str, config_hash: str, output_paths: list[Path], output_row_count: int, metadata: dict | None = None):
    state = load_state(state_path)
    previous = state["stages"].get(stage, {})
    latest_started_at = previous.get("latest_started_at") or now()
    finished_at = now() if status in {"completed", "failed"} else None
    latest_elapsed_seconds = _elapsed_seconds(latest_started_at, finished_at) if finished_at else 0.0
    total_elapsed_seconds = round(float(previous.get("total_elapsed_seconds", 0.0)) + latest_elapsed_seconds, 6)
    attempt = {
        "started_at": latest_started_at,
        "finished_at": finished_at,
        "elapsed_seconds": latest_elapsed_seconds,
        "status": status,
    }
    if metadata:
        attempt["metadata"] = metadata
    state["stages"][stage] = {
        "status": status,
        "started_at": previous.get("first_started_at") or previous.get("started_at") or latest_started_at,
        "finished_at": finished_at if status in {"completed", "failed"} else previous.get("finished_at"),
        "first_started_at": previous.get("first_started_at") or previous.get("started_at") or latest_started_at,
        "latest_started_at": latest_started_at,
        "latest_finished_at": finished_at,
        "latest_elapsed_seconds": latest_elapsed_seconds,
        "total_attempt_count": max(1, int(previous.get("total_attempt_count", 0))),
        "latest_attempt_status": status,
        "latest_skip_reason": previous.get("latest_skip_reason"),
        "total_elapsed_seconds": total_elapsed_seconds,
        "input_fingerprint": input_fingerprint,
        "config_hash": config_hash,
        "output_paths": [str(Path(p)) for p in output_paths],
        "output_row_count": output_row_count,
        "output_file_sha256": {str(Path(p)): sha256_file(p) for p in output_paths if Path(p).exists()},
        "validation_result": "ok" if status == "completed" else status,
        "metadata": {**(previous.get("metadata") or {}), **(metadata or {})},
        "attempt_history": _trim_history([*(previous.get("attempt_history") or []), attempt]),
    }
    save_state(state_path, state)


def stage_validation_reason(state_path: str | Path, stage: str, input_fingerprint: str, config_hash: str) -> tuple[bool, str]:
    item = load_state(state_path).get("stages", {}).get(stage)
    if not item or item.get("status") != "completed":
        return False, "stage is not completed in state"
    if item.get("input_fingerprint") != input_fingerprint or item.get("config_hash") != config_hash:
        return False, "stage fingerprint or config hash changed"
    if not item.get("output_file_sha256"):
        return False, "stage has no declared output hashes"
    for path, expected in item.get("output_file_sha256", {}).items():
        p = Path(path)
        if not p.exists() or sha256_file(p) != expected:
            return False, f"output missing or hash mismatch: {path}"
    return True, "completed stage outputs and hashes are valid"


def stage_is_valid(state_path: str | Path, stage: str, input_fingerprint: str, config_hash: str) -> bool:
    valid, _ = stage_validation_reason(state_path, stage, input_fingerprint, config_hash)
    return valid

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

MANIFEST_STAGE_ALIASES = {
    "normalization": "normalized",
    "scoring": "scores",
}


def _runtime_status(elapsed: float) -> str:
    if elapsed > 270:
        return "BLOCKED"
    if elapsed > 240:
        return "WARNING"
    return "PASS"


def _measure_peak_memory(process: subprocess.Popen, poll_ms: int) -> tuple[float | None, str | None]:
    if psutil is None:
        return None, None
    peak_bytes = 0
    proc = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            family = [proc, *proc.children(recursive=True)]
            peak_bytes = max(
                peak_bytes,
                sum(getattr(member.memory_info(), "wset", member.memory_info().rss) for member in family),
            )
        except psutil.Error:
            pass
        time.sleep(max(0.01, poll_ms / 1000.0))
    try:
        family = [proc, *proc.children(recursive=True)]
        peak_bytes = max(
            peak_bytes,
            sum(getattr(member.memory_info(), "wset", member.memory_info().rss) for member in family),
        )
    except psutil.Error:
        pass
    return round(peak_bytes / (1024 * 1024), 3) if peak_bytes else None, "psutil working set"


def _run_command(command: list[str], poll_ms: int) -> tuple[int, dict]:
    start = time.perf_counter()
    completed = subprocess.Popen(command)
    peak_memory_mb, peak_method = _measure_peak_memory(completed, poll_ms)
    returncode = completed.wait()
    elapsed = time.perf_counter() - start
    return returncode, {
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "runtime_status": _runtime_status(elapsed),
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_method": peak_method,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_report(run_path: Path) -> dict:
    manifest = _load_json(run_path / "manifest.json")
    state = _load_json(run_path / "state.json")
    stages = state.get("stages", {})
    manifest_timings = {
        MANIFEST_STAGE_ALIASES.get(stage, stage): value
        for stage, value in manifest.get("elapsed_time_by_stage", {}).items()
    }
    stage_names = sorted(set(stages) | set(manifest_timings))
    latest_stage_timings = {
        stage: round(float(stages.get(stage, {}).get("latest_elapsed_seconds", manifest_timings.get(stage, 0.0))), 3)
        for stage in stage_names
    }
    total_stage_timings = {
        stage: round(float(stages.get(stage, {}).get("total_elapsed_seconds", latest_stage_timings.get(stage, 0.0))), 3)
        for stage in stage_names
    }
    stage_attempts = {
        stage: {
            "attempt_count": int(stages.get(stage, {}).get("total_attempt_count", 0)),
            "latest_status": stages.get(stage, {}).get("latest_attempt_status"),
            "latest_started_at": stages.get(stage, {}).get("latest_started_at"),
            "latest_finished_at": stages.get(stage, {}).get("latest_finished_at"),
        }
        for stage in stage_names
    }
    output_sizes = {
        str(path): Path(path).stat().st_size
        for path in manifest.get("artifact_paths", {}).values()
        if Path(path).exists()
    }
    semantic_status = _load_json(run_path / "scores" / "semantic_fallback_status.json")
    runtime_cfg = manifest.get("resolved_runtime_configuration", {})
    elapsed = sum(latest_stage_timings.values())
    return {
        "elapsed_seconds": round(elapsed, 3),
        "stage_timings": latest_stage_timings,
        "total_stage_timings": total_stage_timings,
        "stage_attempts": stage_attempts,
        "git_commit": manifest.get("git_commit"),
        "working_tree_dirty": manifest.get("working_tree_dirty"),
        "peak_memory_mb": manifest.get("peak_memory_mb"),
        "peak_memory_method": manifest.get("peak_memory_method"),
        "output_sizes": output_sizes,
        "warning": _runtime_status(elapsed),
        "cpu_info": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "operating_system": manifest.get("operating_system"),
        },
        "processing_mode": runtime_cfg.get("processing_mode"),
        "dataset_reference_date": runtime_cfg.get("dataset_reference_date"),
        "semantic_enabled": manifest.get("resolved_phase2_configuration", {}).get("semantic_config", {}).get("enabled"),
        "semantic_status": semantic_status,
        "run_kind": "resumed" if manifest.get("command_line_arguments", {}).get("resume") else "fresh",
        "runtime_status": _runtime_status(elapsed),
    }


def _infer_run_path_from_command(command: list[str]) -> Path | None:
    try:
        run_id = command[command.index("--run-id") + 1]
        run_dir = command[command.index("--run-dir") + 1]
    except (ValueError, IndexError):
        return None
    return Path(run_dir) / run_id


def _write_benchmark_artifacts(run_path: Path, report: dict) -> None:
    benchmarks_dir = run_path / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    json_path = benchmarks_dir / "runtime_benchmark.json"
    md_path = benchmarks_dir / "runtime_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    stage_lines = "\n".join(
        f"- `{stage}`: `{seconds}s`"
        for stage, seconds in report.get("stage_timings", {}).items()
    )
    md_path.write_text(
        "# Runtime Benchmark\n\n"
        f"Elapsed seconds: `{report.get('elapsed_seconds')}`\n\n"
        f"Runtime status: `{report.get('runtime_status')}`\n\n"
        f"Peak memory MB: `{report.get('peak_memory_mb')}`\n\n"
        f"Peak memory method: `{report.get('peak_memory_method')}`\n\n"
        "## Stage Timings\n\n"
        f"{stage_lines or '- none'}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    parser.add_argument("--poll-ms", type=int, default=100)
    args = parser.parse_args()

    if args.command:
        returncode, report = _run_command(args.command, args.poll_ms)
        inferred_run_path = _infer_run_path_from_command(args.command)
        if inferred_run_path and inferred_run_path.exists():
            report = _run_report(inferred_run_path) | report | {"run_path": str(inferred_run_path.resolve())}
            _write_benchmark_artifacts(inferred_run_path, report)
        print(json.dumps(report, indent=2))
        return returncode

    run_path = Path(args.run_path)
    report = _run_report(run_path)
    persisted_path = run_path / "benchmarks" / "runtime_benchmark.json"
    if persisted_path.exists():
        persisted = _load_json(persisted_path)
        for key in ("command", "returncode", "peak_memory_mb", "peak_memory_method", "run_path"):
            if key in persisted:
                report[key] = persisted[key]
    _write_benchmark_artifacts(run_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

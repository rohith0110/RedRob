from pathlib import Path


def ensure_run_dirs(run_path: str | Path):
    run_path = Path(run_path)
    for name in ("logs", "checkpoints", "audit", "normalized", "evidence", "credibility", "behavioral", "scores", "reasoning", "submissions", "reports", "benchmarks"):
        (run_path / name).mkdir(parents=True, exist_ok=True)
    return run_path

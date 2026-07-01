import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import run_audit
from src.logging_utils import RunLogger
from src.paths import ensure_run_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-dir", default="./runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    run_path = ensure_run_dirs(Path(args.run_dir) / args.run_id)
    logger = RunLogger(args.run_id, run_path, args.log_level)
    summary = run_audit(args.candidates, run_path, logger=logger)
    print(f"audited {summary['total_records']} rows; valid={summary['valid_records']}; malformed={summary['malformed_records']}")


if __name__ == "__main__":
    main()

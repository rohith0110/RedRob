import subprocess
import sys
import time


if __name__ == "__main__":
    t0 = time.perf_counter()
    code = subprocess.call([sys.executable, "rank.py", "--candidates", "./data/candidates.jsonl", "--run-id", "benchmark_baseline", "--run-dir", "./runs", "--force"])
    print(f"exit={code} elapsed_seconds={time.perf_counter() - t0:.3f}")
    raise SystemExit(code)

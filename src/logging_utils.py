import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, run_id: str, run_path: str | Path, level: str = "INFO"):
        self.run_id = run_id
        self.started = time.perf_counter()
        log_dir = Path(run_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = log_dir / "events.jsonl"
        self.logger = logging.getLogger(run_id)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()
        handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)

    def __call__(self, stage, event, message, level="INFO", processed_count=None, total_count=None, output_path=None):
        elapsed = time.perf_counter() - self.started
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
            "level": level,
            "elapsed_seconds": round(elapsed, 3),
            "processed_count": processed_count,
            "total_count": total_count,
            "output_path": str(output_path) if output_path else None,
            "message": message,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.logger.log(getattr(logging, level.upper(), logging.INFO), "%s:%s %s", stage, event, message)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

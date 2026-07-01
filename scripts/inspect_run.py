import json
import sys
from pathlib import Path


if __name__ == "__main__":
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/phase1_baseline")
    print((run / "manifest.json").read_text(encoding="utf-8"))
    state = run / "state.json"
    if state.exists():
        print(json.dumps(json.loads(state.read_text(encoding="utf-8")), indent=2))

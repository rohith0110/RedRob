# Reproduction

## Environment

- Local benchmark machine: `Windows-11-10.0.26200-SP0`
- Python: `3.14.6`
- CPU cores: `20`
- RAM: `31.7 GB`
- Ranking mode: CPU-only, offline

## Final frozen-tag local run

Frozen tag:

- `ranking-freeze-step3b`
- commit `dc2c92deae0e8cc95672f1b682c2e094b32df1eb`

Command:

```powershell
python scripts/benchmark_runtime.py --command python rank.py --mode v2 --candidates D:/Coding/Randoms/RedRob/data/candidates.jsonl --out D:/Coding/Randoms/RedRob/submissions/Team_loading.csv --run-id release_final_local --run-dir D:/Coding/Randoms/RedRob/runs
```

Observed result:

- runtime `226.949s`
- peak memory `3982.164 MB`
- memory method `psutil working set`
- status `PASS`

Validator:

```powershell
python validate_submission.py submissions/Team_loading.csv
```

Resume:

```powershell
python rank.py --mode v2 --resume --candidates D:/Coding/Randoms/RedRob/data/candidates.jsonl --out D:/Coding/Randoms/RedRob/submissions/Team_loading.csv --run-id release_final_local --run-dir D:/Coding/Randoms/RedRob/runs
```

Observed resume runtime: `6.295s`

## Docker reproduction

Build:

```powershell
docker build -t redrob-ranker:release .
```

Linux/macOS:

```bash
docker run --rm --network none \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/submissions:/out" \
  redrob-ranker:release \
  python rank.py \
  --mode v2 \
  --candidates /data/candidates.jsonl \
  --out /out/Team_loading.csv \
  --run-id docker_release_full \
  --run-dir /tmp/runs
```

PowerShell:

```powershell
docker run --rm --network none `
  -v "${PWD}/data:/data:ro" `
  -v "${PWD}/submissions:/out" `
  redrob-ranker:release `
  python rank.py `
  --mode v2 `
  --candidates /data/candidates.jsonl `
  --out /out/Team_loading.csv `
  --run-id docker_release_full `
  --run-dir /tmp/runs
```

Observed Docker results:

- `docker_release_full`: stage sum `247.042s`, container wall time `260.723s`
- `docker_release_full_2`: stage sum `251.110s`, container wall time `264.855s`
- verification peak memory: `7132.160 MB`
- memory method: `docker stats` polling
- network mode: `none`
- validator: `passed`

The Docker run stayed inside the packaging warning band, so a second deterministic verification run was executed as required.

## Local vs Docker equivalence

- Submission CSV hash equal: `true`
- Reasoning JSONL hash equal: `true`
- Score CSV hash equal: `false`
- Score CSV semantic equality on `candidate_id`, `rank`, `final_score`: `true`

The score CSV hash differs across environments because of platform-level formatting differences, not ranking changes.

## Release manifest

Final release details, hashes, commands, and warnings are recorded in:

- `release/RELEASE_MANIFEST.json`
- `release/RELEASE_MANIFEST.md`

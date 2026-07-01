# Docker Package

This folder records the offline Docker reproduction summary for the release package.

Primary runtime observations:

- `docker_release_full`: `247.042s` summed stage time, `260.723s` container wall time
- `docker_release_full_2`: `251.110s` summed stage time, `264.855s` container wall time
- validator: passed on Docker output
- network mode: `none`
- local and Docker submission CSV hashes: match
- local and Docker reasoning JSONL hashes: match

See `../RELEASE_MANIFEST.json` for the exact commands and full comparison details.

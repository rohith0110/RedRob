# Release Manifest

## Provenance

- Ranking freeze tag: `ranking-freeze-step3b`
- Ranking freeze commit: `dc2c92deae0e8cc95672f1b682c2e094b32df1eb`
- Ranking run commit: `dc2c92deae0e8cc95672f1b682c2e094b32df1eb`
- Package commit before this release-closure commit: `0141dbfcc270d38c0a60c6113ad54c73a783e91b`
- Final audit commit: recorded in git history by the packaging-only release-closure commit
- Ranking working tree clean: `true`
- Packaging working tree clean at package commit: `true`

Packaging-only commits after the freeze do not alter candidate IDs, ranks, the score CSV,
the score formula, ranker logic, or Docker ranking behavior.

## Frozen Outputs

- Final CSV filename: `Team_loading.csv`
- Final submission CSV SHA-256: `4ecfbb82fd7d591ad6c822df4546fe67453dd9ac703ddb6f1358431f302614b9`
- Final score CSV SHA-256: `039318ea80944900f28ad91a407025f0950552097b12006a3927bf9792091e49`
- Final reasoning JSONL SHA-256: `24734b7c3278cc9e4f96eaada425e344c0d3f101e794e76b9a0bdcc79320202a`

## Recorded Validation

- Local run ID: `release_final_local`
- Local runtime seconds: `226.949`
- Local peak memory MB: `3982.164`
- Docker run ID: `docker_release_full`
- Docker runtime seconds: `247.042`
- Docker second verification runtime seconds: `251.11`
- Validator passed: `true`
- Local vs Docker candidate IDs, ranks, and scores equal: `true`
- Local vs Docker reasoning hash equal: `true`

## Known Reproducibility Notes

- Docker full-run timing is host and storage dependent; all validated runs remained below
  the official 300-second limit.
- Final submission CSV is byte-identical across validated Docker runs.
- Diagnostic score-breakdown float formatting may differ by runtime environment, while
  official output fields remain invariant.

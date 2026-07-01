# Sandbox Deployment

## Scope

The sandbox is a local Streamlit app for small recruiter-facing samples. It is not an official submission path and is not claimed as publicly hosted in this repository.

## Install

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-sandbox.txt
```

## Launch

```powershell
streamlit run app.py
```

## Inputs

- Accepts `.json` and `.jsonl`
- Accepts at most `100` candidates
- Validates candidate schema before ranking
- Includes sample files:
  - [sample_data/demo_candidates.json](../sample_data/demo_candidates.json)
  - [sample_data/demo_candidates.jsonl](../sample_data/demo_candidates.jsonl)

## Behavior

The sandbox imports the same modules as production:

- normalization
- evidence extraction
- credibility analysis
- behavioral analysis
- v2 scoring
- recruiter-facing reasoning
- CSV submission writing

Displayed output:

- rank
- overall score
- score components
- positive and negative evidence excerpts
- credibility modifier
- availability modifier
- recruiter-facing reasoning
- ranked CSV download

## Deliberate sandbox constraints

- no API key
- no hosted API
- CPU-only
- no alternate demo scoring logic
- not valid as an official competition submission when fewer than `100` candidates are ranked

## Smoke test

```powershell
pytest tests/test_sandbox_smoke.py -q
```

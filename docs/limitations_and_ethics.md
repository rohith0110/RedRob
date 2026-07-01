# Limitations And Ethics

## Operational limitations

- The system ranks from profile text only. Missing or weak profile detail lowers what can be inferred.
- It is deterministic and conservative, so it prefers grounded evidence over aggressive narrative extrapolation.
- Availability and logistics are bounded modifiers, not decisive substitutes for role fit.
- Semantic scoring is disabled in the submitted release configuration.
- Docker reproduction remains under the official `300s` limit but inside the project warning band.

## Responsible-use limits

- This project is a recruiter-triage tool, not a hiring decision-maker.
- Human review is required before any shortlist is acted on.
- Explanations are constrained to grounded profile evidence and should not be interpreted as a full candidate assessment.
- No raw private candidate data should be exposed outside the allowed submission and demo surfaces.

## What is intentionally not claimed

- No accuracy, NDCG, MAP, or leaderboard claim
- No claim of fairness certification
- No claim that the system predicts future job performance
- No claim that the sandbox is publicly deployed

## Recommended human review

- spot-check top-ranked reasoning against source evidence
- review unexpectedly high or low candidates for sparse profiles
- treat availability signals as secondary to technical fit
- revisit profile quality before drawing strong conclusions from low-information records

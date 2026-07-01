# Methodology

## Challenge framing

The target role is retrieval and ranking heavy. Pure keyword counts overvalue copied skill lists, short-term experimentation, and generic AI buzzwords. This system favors evidence that the candidate actually built, shipped, measured, or operated ranking-relevant systems.

## Evidence hierarchy

Ordered from strongest to weakest:

1. `career_history[*].description`
2. `career_history[*].title`
3. `profile.summary`
4. `profile.headline`
5. `skills[*].name`
6. `skill_assessments[*].name`

Skills can support a profile, but unsupported skills do not outrank grounded career evidence.

## Core scoring components

- `career_relevance_score`
- `retrieval_ranking_depth_score`
- `production_evaluation_score`
- `product_shipper_score`
- `experience_recent_coding_score`
- `corroborated_skill_score`
- `preferred_differentiator_score`
- `location_logistics_score`

Multipliers:

- `credibility_multiplier`
- `availability_multiplier`

Concise formula:

`final_score = base_fit_score * credibility_multiplier * availability_multiplier * 100`

## Plain-language matching

The evidence taxonomy recognizes recruiter-meaningful language such as:

- search relevance
- recommendation systems
- personalization
- marketplace ranking
- product discovery
- semantic retrieval
- evaluation and experimentation
- production ownership

The method does not require exact tool buzzwords to score relevant experience.

## Anti-keyword defenses

- Skills without matching work-history evidence contribute less than corroborated skills.
- Unsupported advanced claims contribute to risk rather than positive fit.
- Contradictory or inflated profile wording is handled through credibility penalties.
- Availability modifiers are bounded and cannot overpower technical irrelevance.

## Reasoning realization

Top-100 explanations are generated from already-scored evidence rather than from a hosted model. The release path is:

1. Select grounded evidence IDs
2. Normalize facts into recruiter-facing fragments
3. Assemble clauses by rank band
4. Build concise sentences
5. Run cleanup and style lint
6. Validate grounding back to source evidence
7. Fall back to safer deterministic phrasing when a sentence is not release-safe

## Final release quality gates

From the final local release run `release_final_local`:

- Score output changed versus approved Step 3A.1 baseline: `false`
- Top-100 IDs match baseline: `true`
- Top-100 ranks match baseline: `true`
- Top-100 scores match baseline: `true`
- Top-100 lint failures: `0`
- Top-100 grounding failures: `0`

## What this method does not claim

- No NDCG, MAP, leaderboard, or accuracy claim is asserted here.
- No hiring-outcome or quality-of-hire claim is asserted here.
- No hosted LLM is used in the submitted ranking runtime.

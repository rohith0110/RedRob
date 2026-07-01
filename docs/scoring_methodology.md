# Scoring Methodology

## Evidence hierarchy

Highest value:

1. `career_history[*].description`
2. `career_history[*].title`
3. `profile.summary`
4. `profile.headline`
5. `skills[*].name`
6. `skill_assessments[*].name`

Skills alone can create evidence rows, but they do not become strongly corroborated unless career evidence supports the same category.

## V2 features

- `career_relevance_score`
- `retrieval_ranking_depth_score`
- `production_evaluation_score`
- `product_shipper_score`
- `experience_recent_coding_score`
- `corroborated_skill_score`
- `preferred_differentiator_score`
- `location_logistics_score`

## Multipliers

- `credibility_multiplier`
- `availability_multiplier`

Final score:

`base_fit_score * credibility_multiplier * availability_multiplier * 100`

## Plain-language recall

The retrieval/ranking taxonomy explicitly scores phrases such as:

- improved product discovery
- ranked marketplace listings
- built a personalized feed
- improved search results
- matched users to jobs

Current buzzwords are not required.

## Recruiter-facing reasoning

Top-100 reasoning is deterministic and grounded in the evidence ledger.

- realization pipeline:
  1. evidence selection
  2. fact normalization
  3. clause construction
  4. rank-band template selection
  5. sentence assembly
  6. text cleanup
  7. style linting
  8. grounding validation
  9. safe fallback when a fragment is not sentence-safe
- ranks `1-10`: strongest role-fit language
- ranks `11-30`: strong shortlist language with a clear gap to the very top
- ranks `31-60`: credible-but-less-complete language
- ranks `61-100`: near-cutoff language that must include a grounded positive plus a grounded limitation or explicit evidence-limited qualifier

Step 3A also adds:

- `runs/<run_id>/reasoning/reasoning_style_lint.jsonl` for candidate-level lint findings
- `runs/<run_id>/reports/reasoning_style_quality_report.json|md` for aggregate counts, rank-band breakdowns, sampled ranks, and pass/fail status

Unsafe literal snippets are rejected before final text when they are bare titles, bare technologies, dangling fragments, duplicated snippets, or lowercase sentence starts. In those cases the reasoner falls back to category-grounded facts without changing scores, candidate IDs, or rank order.

## Keyword-stuffer defense

- AI-heavy skills without matching career evidence lower `skill_corroboration_score`
- unsupported advanced AI/IR skills raise `unsupported_skill_risk_score`
- credibility handles compounded contradictions separately from role fit

## Availability bounds

Availability is clamped to `0.72..1.08`.

It can move similar candidates apart, but it cannot rescue an irrelevant technical profile into the top tier.

Phase 2.1 adds a date-aware `last_active_recency_component`, resolved against the fixed dataset reference date from `configs/runtime.yaml`.

## Limitations

- Semantic scoring is optional and capped.
- Chunked mode is correctness-tested, not the default 100k execution path.
- Reasoning is deterministic and conservative; it prefers grounded category statements over aggressive narrative detail.
- Evidence profiling shows the dominant Step 2 cost remains JSON/Parquet I/O, so Step 3A performance changes stayed in the extraction loop and preserved the Phase 2.1 score hash exactly.

## Semantic enablement

- `semantic_config.enabled: false`: skip artifact loading entirely and force semantic contribution to zero
- `semantic_config.enabled: true`: attempt local artifact loading, validate the manifest, and fall back safely to zero on absence or mismatch

The semantic cap remains config-driven and offline-only.

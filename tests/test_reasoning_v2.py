import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.behavioral import run_behavioral_analysis
from src.credibility import run_credibility_analysis
from src.evidence import run_evidence_extraction
from src.normalization import run_normalization
from src.paths import ensure_run_dirs
from src.reasoning_v2 import (
    FORBIDDEN_REASONING_TERMS,
    MAX_DUPLICATE_REASONING_COUNT,
    analyze_reasoning_quality,
    build_reasoning_plan,
    normalize_fact_phrase,
    render_reasoning_from_plan,
    run_reasoning_generation,
    validate_reasoning_style,
)
from src.v2_scoring import run_v2_scoring


def make_candidate(
    candidate_id: str,
    title: str,
    summary: str,
    description: str,
    skills: list[dict],
    *,
    work_mode: str = "remote",
    notice_period_days: int = 30,
    response_rate: float = 0.7,
    github_activity_score: int = 20,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": title,
            "summary": summary,
            "current_title": title,
            "current_company": "Acme",
            "current_industry": "Software",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 6,
        },
        "career_history": [
            {
                "title": title,
                "company": "Acme",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 48,
                "is_current": True,
                "description": description,
            }
        ],
        "education": [],
        "skills": skills,
        "certifications": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": response_rate,
            "avg_response_time_hours": 18,
            "notice_period_days": notice_period_days,
            "preferred_work_mode": work_mode,
            "willing_to_relocate": work_mode != "onsite",
            "github_activity_score": github_activity_score,
            "interview_completion_rate": 0.85,
            "skill_assessment_scores": {"Python": 88},
        },
    }


def reasoning_fixture_candidates() -> list[dict]:
    return [
        make_candidate(
            "CAND_0000001",
            "Search Engineer",
            "Built production search relevance systems and recommendation loops.",
            "Owned retrieval pipelines, tuned ranking metrics, and shipped search relevance improvements to live users.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
        ),
        make_candidate(
            "CAND_0000002",
            "Recommendation Engineer",
            "Improved recommendation quality and matching flows.",
            "Built recommendation services, ran A/B evaluation, and improved matching quality for marketplace users.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
        ),
        make_candidate(
            "CAND_0000003",
            "ML Platform Engineer",
            "Shipped backend systems for model-driven products.",
            "Built Python APIs, orchestration services, and production delivery tooling for ML teams.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 60}],
        ),
        make_candidate(
            "CAND_0000004",
            "Founding Engineer",
            "Led product ownership across search and onboarding.",
            "Owned product delivery, customer-facing prioritization, and experimentation for discovery workflows.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 42}],
        ),
        make_candidate(
            "CAND_0000005",
            "Research Engineer",
            "Worked on retrieval prototypes and evaluation studies.",
            "Ran retrieval experiments and offline evaluation, with less evidence of production ownership.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 36}],
        ),
        make_candidate(
            "CAND_0000006",
            "Backend Engineer",
            "Built backend services and relevance support tooling.",
            "Delivered Python services and practical engineering support for product teams.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
            notice_period_days=75,
        ),
        make_candidate(
            "CAND_0000007",
            "Data Engineer",
            "Supported analytics and data infrastructure.",
            "Built dashboards and data pipelines with limited direct ranking ownership.",
            [{"name": "SQL", "proficiency": "advanced", "duration_months": 48}],
            work_mode="onsite",
        ),
        make_candidate(
            "CAND_0000008",
            "Applied Scientist",
            "Worked on ranking and experimentation with mixed deployment depth.",
            "Improved ranking evaluation and experimentation, but the profile shows less production shipping detail.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 36}],
        ),
        make_candidate(
            "CAND_0000009",
            "Product Engineer",
            "Owned delivery for discovery-facing features.",
            "Shipped product features, Python services, and practical engineering work for user-facing flows.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
        ),
        make_candidate(
            "CAND_0000010",
            "Software Engineer",
            "Generalist backend profile.",
            "Built backend services with some applied ML support but limited retrieval depth in the available profile.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 30}],
        ),
        make_candidate(
            "CAND_0000011",
            "ML Engineer",
            "Mentions AI leadership in title only.",
            "Built dashboards for reporting and internal tooling with sparse direct relevance details.",
            [{"name": "LLM", "proficiency": "advanced", "duration_months": 24}],
            notice_period_days=90,
        ),
        make_candidate(
            "CAND_0000012",
            "Platform Engineer",
            "Adjacent infrastructure experience.",
            "Built APIs and batch systems with limited explicit search or ranking evidence.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 24}],
            work_mode="hybrid",
        ),
    ]


def sample_plan(rank: int) -> dict:
    return build_reasoning_plan(
        score_row=pd.Series(
            {
                "candidate_id": "CAND_PLAN",
                "rank": rank,
                "final_score": 91.4 if rank < 10 else 56.2,
                "credibility_multiplier": 1.0 if rank < 10 else 0.88,
                "availability_multiplier": 1.02 if rank < 10 else 0.82,
                "location_logistics_score": 1.0 if rank < 10 else 0.55,
            }
        ),
        context={
            "candidate_id": "CAND_PLAN",
            "profile": {"current_title": "Search Engineer", "country": "India"},
            "redrob_signals": {"preferred_work_mode": "remote", "notice_period_days": 30 if rank < 10 else 90},
        },
        positives=[
            {
                "evidence_id": "EVID_1",
                "normalized_category": "retrieval_ranking_relevance",
                "source_type": "career_description",
                "matched_terms": ["retrieval", "ranking"],
                "exact_source_excerpt": "Owned retrieval pipelines and tuned ranking metrics.",
            },
            {
                "evidence_id": "EVID_2",
                "normalized_category": "production_delivery_systems",
                "source_type": "career_description",
                "matched_terms": ["production", "delivery"],
                "exact_source_excerpt": "Shipped production systems for live users.",
            },
            {
                "evidence_id": "EVID_3",
                "normalized_category": "python_practical_engineering",
                "source_type": "skill",
                "matched_terms": ["python"],
                "exact_source_excerpt": "Python",
            },
        ],
        negatives=[],
    )


class ReasoningV2Test(unittest.TestCase):
    def test_reasoning_is_grounded_diverse_and_free_of_forbidden_terms(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in reasoning_fixture_candidates()) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            run_credibility_analysis(run_path)
            run_behavioral_analysis(run_path)
            run_v2_scoring(run_path)
            result = run_reasoning_generation(run_path)

            reasoning_df = pd.read_parquet(result["reasoning_parquet"])
            self.assertEqual(len(reasoning_df), 12)
            self.assertTrue(reasoning_df["reasoning"].str.len().gt(0).all())
            grounding = json.loads(Path(result["reasoning_grounding_report_json"]).read_text(encoding="utf-8"))
            diversity = json.loads((run_path / "reports" / "reasoning_diversity_report.json").read_text(encoding="utf-8"))

            self.assertEqual(grounding["failed_count"], 0)
            self.assertLessEqual(diversity["max_exact_duplicate_count"], MAX_DUPLICATE_REASONING_COUNT)
            self.assertGreaterEqual(diversity["unique_full_string_count"], 10)
            self.assertGreaterEqual(diversity["grounding_pass_count"], 12)
            self.assertGreaterEqual(diversity["concrete_evidence_count"], 10)
            for reasoning in reasoning_df["reasoning"]:
                lowered = reasoning.lower()
                self.assertFalse(any(word in lowered for word in FORBIDDEN_REASONING_TERMS))

    def test_different_evidence_structures_do_not_share_template_family(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in reasoning_fixture_candidates()[:4]) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            run_credibility_analysis(run_path)
            run_behavioral_analysis(run_path)
            run_v2_scoring(run_path)
            run_reasoning_generation(run_path)

            rows = [
                json.loads(line)
                for line in (run_path / "reasoning" / "reasoning_v2.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_id = {row["candidate_id"]: row for row in rows}
            self.assertNotEqual(by_id["CAND_0000001"]["template_family"], by_id["CAND_0000004"]["template_family"])
            self.assertNotEqual(by_id["CAND_0000001"]["reasoning"], by_id["CAND_0000004"]["reasoning"])

    def test_rank_band_changes_tone_and_lower_rank_is_more_qualified(self):
        top_plan = sample_plan(rank=4)
        cutoff_plan = sample_plan(rank=92)

        self.assertEqual(top_plan["rank_band"], "1-10")
        self.assertEqual(cutoff_plan["rank_band"], "61-100")

        top_reasoning = render_reasoning_from_plan(top_plan)
        cutoff_reasoning = render_reasoning_from_plan(cutoff_plan)

        self.assertNotEqual(top_reasoning, cutoff_reasoning)
        self.assertNotIn("below stronger candidates", top_reasoning.lower())
        self.assertIn("shortlist cutoff", cutoff_reasoning.lower())
        self.assertTrue(
            "limited" in cutoff_reasoning.lower() or "weaker" in cutoff_reasoning.lower()
        )

    def test_cutoff_band_reasoning_avoids_doubled_included_because_pattern(self):
        cutoff_plan = sample_plan(rank=92)
        cutoff_reasoning = render_reasoning_from_plan(cutoff_plan)
        lowered = cutoff_reasoning.lower()

        self.assertLessEqual(lowered.count("included"), 1)
        self.assertNotIn("because because", lowered)
        self.assertIsNone(re.search(r"included .* because .* included .* because", lowered))

    def test_reasoning_quality_analysis_tracks_repeated_openings_and_cutoff_caveats(self):
        report = analyze_reasoning_quality(
            [
                {
                    "candidate_id": "CAND_1",
                    "rank": 61,
                    "rank_band": "61-100",
                    "template_family": "lower_cutoff_candidate",
                    "reasoning": "Relevant ranking evaluation is present, although explicit evidence of production ownership is limited. This keeps the candidate near the shortlist cutoff rather than among stronger retrieval-and-ranking fits.",
                    "grounding_pass": True,
                    "material_caveat": False,
                    "grounded_positive": True,
                    "grounded_limitation_or_caveat": True,
                },
                {
                    "candidate_id": "CAND_2",
                    "rank": 62,
                    "rank_band": "61-100",
                    "template_family": "lower_cutoff_candidate",
                    "reasoning": "Relevant ranking evaluation is present, although explicit evidence of vector or search infrastructure is limited. This keeps the candidate near the shortlist cutoff rather than among stronger retrieval-and-ranking fits.",
                    "grounding_pass": True,
                    "material_caveat": False,
                    "grounded_positive": True,
                    "grounded_limitation_or_caveat": True,
                },
                {
                    "candidate_id": "CAND_3",
                    "rank": 7,
                    "rank_band": "1-10",
                    "template_family": "retrieval_ranking_heavy",
                    "reasoning": "Search Engineer experience shows retrieval pipelines and ranking metrics. The available profile remains strong across the core role-fit signals.",
                    "grounding_pass": True,
                    "material_caveat": False,
                    "grounded_positive": True,
                    "grounded_limitation_or_caveat": True,
                },
            ],
            repeated_opening_threshold=1,
        )

        self.assertEqual(report["repeated_opening_count"], 1)
        self.assertEqual(report["awkward_pattern_count"], 0)
        self.assertEqual(report["grounding_result"]["failed_count"], 0)
        self.assertEqual(report["cutoff_band_caveat_coverage"]["covered_count"], 2)
        self.assertEqual(report["cutoff_band_positive_coverage"]["covered_count"], 2)

    def test_normalize_fact_phrase_repairs_duplicates_and_rejects_bare_or_dangling_fragments(self):
        self.assertEqual(
            normalize_fact_phrase("Built ranking systems for for recruiters in production."),
            "Built ranking systems for recruiters in production.",
        )
        self.assertEqual(
            normalize_fact_phrase("Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store."),
            "Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and an internal feature store.",
        )
        self.assertIsNone(normalize_fact_phrase("Backend development with Python (FastAPI), PostgreSQL, and Redis at a"))
        self.assertIsNone(normalize_fact_phrase("Flask"))
        self.assertIsNone(normalize_fact_phrase("and service"))
        self.assertIsNone(normalize_fact_phrase("while ci/cd remains visible"))

    def test_validate_reasoning_style_flags_live_reviewer_failures(self):
        lint = validate_reasoning_style(
            "Senior Machine Learning Engineer experience shows Built a RAG-based ranking pipeline serving 50M+ queries per month, "
            "backed by Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for for production recommendation work. "
            "some profile detail remains less complete around langchain",
            rank=1,
            rank_band="1-10",
            grounding_phrases=[
                "Built a RAG-based ranking pipeline serving 50M+ queries per month",
                "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for",
            ],
            limitation="some profile detail remains less complete around langchain",
        )
        self.assertFalse(lint["passed"])
        self.assertIn("duplicate_adjacent_word", lint["failed_checks"])
        self.assertIn("lowercase_sentence_start", lint["failed_checks"])
        self.assertIn("missing_terminal_punctuation", lint["failed_checks"])

    def test_validate_reasoning_style_accepts_availability_constraint_sentence(self):
        lint = validate_reasoning_style(
            "Recommendation Systems Engineer background still shows retrieval and ranking work. A 90-day notice period slows availability.",
            rank=32,
            rank_band="31-60",
            grounding_phrases=["retrieval and ranking work"],
            limitation="availability signals are weaker than the strongest candidates",
        )
        self.assertTrue(lint["passed"])

        flexible_lint = validate_reasoning_style(
            "Search Engineer background still shows retrieval and ranking work. Location logistics are less flexible than the strongest candidates.",
            rank=34,
            rank_band="31-60",
            grounding_phrases=["retrieval and ranking work"],
            limitation="location logistics are less flexible than the strongest candidates",
        )
        self.assertTrue(flexible_lint["passed"])

        narrow_lint = validate_reasoning_style(
            "Recommendation Systems Engineer background still shows retrieval and ranking work. Onsite work-mode preference narrows logistics flexibility.",
            rank=39,
            rank_band="31-60",
            grounding_phrases=["retrieval and ranking work"],
            limitation="onsite work-mode preference narrows logistics flexibility",
        )
        self.assertTrue(narrow_lint["passed"])

        thinner_lint = validate_reasoning_style(
            "Staff Machine Learning Engineer background still shows retrieval and ranking work. Response history is thinner than stronger candidates.",
            rank=58,
            rank_band="31-60",
            grounding_phrases=["retrieval and ranking work"],
            limitation="response history is thinner than stronger candidates",
        )
        self.assertTrue(thinner_lint["passed"])

    def test_render_reasoning_from_plan_avoids_bare_repeated_and_dangling_fragments(self):
        plan = {
            "candidate_id": "CAND_PATCH",
            "rank": 88,
            "score": 40.0,
            "rank_band": "61-100",
            "template_family": "lower_cutoff_candidate",
            "title": "Senior Software Engineer",
            "primary_role_fit_category": "production_delivery_systems",
            "primary_phrase": "Backend development with Python (FastAPI), PostgreSQL, and Redis at a",
            "production_phrase": "Backend development with Python (FastAPI), PostgreSQL, and Redis at a",
            "practical_phrase": "and service",
            "availability_phrase": None,
            "availability_constraint_phrase": None,
            "limitation": "explicit evidence of ranking evaluation is limited in the available profile",
            "material_caveat": False,
            "grounding_phrases": [
                "Backend development with Python (FastAPI), PostgreSQL, and Redis at a",
                "and service",
            ],
            "grounding_evidence_ids": ["EVID_1"],
            "positive_evidence_ids": ["EVID_1"],
            "negative_evidence_ids": [],
            "concrete_evidence": True,
        }

        reasoning = render_reasoning_from_plan(plan)
        lowered = reasoning.lower()

        self.assertNotIn("and service", lowered)
        self.assertNotIn(" at a", lowered)
        self.assertEqual(lowered.count("backend development with python"), 0)
        self.assertTrue(reasoning.endswith("."))

    def test_render_reasoning_from_plan_neutralizes_first_person_product_phrases(self):
        plan = sample_plan(rank=13)
        plan["title"] = "Applied ML Engineer"
        plan["primary_phrase"] = "Machine learning engineer with 7.4 years of experience building ML-powered features in production."
        plan["production_phrase"] = "Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM."
        plan["practical_phrase"] = "Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store."
        plan["grounding_phrases"] = [
            plan["primary_phrase"],
            plan["production_phrase"],
            plan["practical_phrase"],
        ]

        reasoning = render_reasoning_from_plan(plan)
        lowered = reasoning.lower()

        self.assertNotIn("our product", lowered)
        self.assertIn("candidate's product", lowered)

    def test_render_reasoning_from_plan_neutralizes_we_deployed_at_sentence_start(self):
        plan = sample_plan(rank=4)
        plan["primary_phrase"] = "We deployed semantic search and retrieval improvements to live users."
        plan["grounding_phrases"] = [plan["primary_phrase"], plan["production_phrase"], plan["practical_phrase"]]

        reasoning = render_reasoning_from_plan(plan)

        self.assertIn("the candidate's team deployed semantic search and retrieval improvements to live users", reasoning.lower())
        self.assertNotIn("We deployed", reasoning)
        self.assertTrue(reasoning.endswith("."))

    def test_run_reasoning_generation_writes_style_lint_artifacts_and_zero_failures(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in reasoning_fixture_candidates()) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            run_credibility_analysis(run_path)
            run_behavioral_analysis(run_path)
            run_v2_scoring(run_path)
            run_reasoning_generation(run_path)

            lint_rows = [
                json.loads(line)
                for line in (run_path / "reasoning" / "reasoning_style_lint.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lint_rows), 12)
            self.assertTrue(all("failed_checks" in row for row in lint_rows))
            self.assertTrue(all(row["passed"] for row in lint_rows))

            style_report = json.loads((run_path / "reports" / "reasoning_style_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(style_report["total_explanations"], 12)
            self.assertEqual(style_report["lint_fail_count"], 0)
            self.assertEqual(style_report["explanation_quality_status"], "PASS")
            self.assertIn("sampled_explanations", style_report)

    def test_run_reasoning_generation_marks_neutralized_candidate_voice_as_grounded_positive(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            candidate = make_candidate(
                "CAND_NEUTRAL",
                "Applied ML Engineer",
                "Machine learning engineer with 7.4 years of experience building ML-powered features in production.",
                (
                    "Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. "
                    "Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store."
                ),
                [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
            )
            dataset.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            run_credibility_analysis(run_path)
            run_behavioral_analysis(run_path)
            run_v2_scoring(run_path)
            run_reasoning_generation(run_path)

            row = json.loads((run_path / "reasoning" / "reasoning_v2.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertTrue(row["grounded_positive"])
            self.assertNotIn("our product", row["reasoning"].lower())
            self.assertNotIn("our internal", row["reasoning"].lower())

    def test_reasoning_generation_is_deterministic_and_preserves_score_csv_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in reasoning_fixture_candidates()) + "\n", encoding="utf-8")

            def run_once(name: str) -> tuple[str, bytes]:
                run_path = ensure_run_dirs(td_path / name)
                run_normalization(dataset, run_path)
                run_evidence_extraction(run_path)
                run_credibility_analysis(run_path)
                run_behavioral_analysis(run_path)
                run_v2_scoring(run_path)
                score_bytes_before = (run_path / "scores" / "score_breakdown_v2.csv").read_bytes()
                run_reasoning_generation(run_path)
                score_bytes_after = (run_path / "scores" / "score_breakdown_v2.csv").read_bytes()
                self.assertEqual(score_bytes_before, score_bytes_after)
                return (
                    (run_path / "reasoning" / "reasoning_v2.jsonl").read_text(encoding="utf-8"),
                    score_bytes_after,
                )

            first_reasoning, first_score_bytes = run_once("run_one")
            second_reasoning, second_score_bytes = run_once("run_two")
            self.assertEqual(first_reasoning, second_reasoning)
            self.assertEqual(first_score_bytes, second_score_bytes)


if __name__ == "__main__":
    unittest.main()

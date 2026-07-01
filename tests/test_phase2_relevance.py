import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evidence import run_evidence_extraction
from src.paths import ensure_run_dirs


def normalized_row(candidate_id: str, career_text: str, profile_text: str, skill_text: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "combined_career_text": career_text,
        "combined_profile_text": profile_text,
        "combined_skill_text": skill_text,
        "country": "India",
        "preferred_work_mode": "remote",
        "willing_to_relocate": False,
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 12,
        "notice_period_days": 30,
        "github_activity_score": 20,
        "interview_completion_rate": 0.8,
        "years_of_experience": 6,
        "anomaly_severity": "none",
    }


def context_row(candidate_id: str, title: str, description: str, summary: str, skills: list[dict]) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": title,
            "summary": summary,
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 6,
            "current_title": title,
            "current_company": "Acme",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "index": 0,
                "company": "Acme",
                "title": title,
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 48,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": description,
            }
        ],
        "skills": skills,
        "certifications": [],
        "skill_assessments": [],
        "redrob_signals": {
            "notice_period_days": 30,
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "interview_completion_rate": 0.8,
            "preferred_work_mode": "remote",
            "willing_to_relocate": False,
            "github_activity_score": 20,
        },
        "candidate_record_hash": candidate_id,
    }


class Phase2RelevanceTest(unittest.TestCase):
    def write_inputs(self, run_path: Path, normalized_rows: list[dict], context_rows: list[dict]) -> None:
        normalized_dir = run_path / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(normalized_rows).to_parquet(normalized_dir / "candidates_normalized.parquet", index=False)
        with (normalized_dir / "candidates_normalized.jsonl").open("w", encoding="utf-8") as handle:
            for row in normalized_rows:
                handle.write(json.dumps(row) + "\n")
        with (normalized_dir / "candidate_context.jsonl").open("w", encoding="utf-8") as handle:
            for row in context_rows:
                handle.write(json.dumps(row) + "\n")

    def test_evidence_backed_plain_language_candidate_beats_skill_stuffer_signals(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = ensure_run_dirs(Path(td) / "run")
            normalized_rows = [
                normalized_row(
                    "CAND_0000001",
                    "Improved product discovery and ranked marketplace listings for live users.",
                    "Built recommendations and improved search results.",
                    "Python Elasticsearch",
                ),
                normalized_row(
                    "CAND_0000002",
                    "Built internal dashboards for finance reporting.",
                    "Interested in AI and LLM tools.",
                    "LLM Embeddings Pinecone LangChain",
                ),
            ]
            context_rows = [
                context_row(
                    "CAND_0000001",
                    "Software Engineer",
                    "Improved product discovery, tuned relevance, and ranked marketplace listings for live users.",
                    "Built recommendations and improved search results.",
                    [{"index": 0, "name": "Python", "proficiency": "advanced", "duration_months": 48}],
                ),
                context_row(
                    "CAND_0000002",
                    "Analyst",
                    "Built internal dashboards for reporting and weekly reviews.",
                    "Interested in AI and learning Pinecone tutorials.",
                    [
                        {"index": 0, "name": "LLM", "proficiency": "advanced", "duration_months": 12},
                        {"index": 1, "name": "Embeddings", "proficiency": "advanced", "duration_months": 12},
                        {"index": 2, "name": "Pinecone", "proficiency": "advanced", "duration_months": 12},
                    ],
                ),
            ]
            self.write_inputs(run_path, normalized_rows, context_rows)

            result = run_evidence_extraction(run_path)
            summary = pd.read_parquet(result["evidence_summary_parquet"]).set_index("candidate_id")

            self.assertGreater(
                summary.loc["CAND_0000001", "retrieval_ranking_relevance_score"],
                summary.loc["CAND_0000002", "retrieval_ranking_relevance_score"],
            )
            self.assertEqual(summary.loc["CAND_0000002", "skill_corroboration_score"], 0.0)
            self.assertGreater(summary.loc["CAND_0000002", "unsupported_skill_risk_score"], 0.5)

    def test_plain_language_relevance_cases_score_without_buzzwords(self):
        phrases = [
            "improved product discovery for marketplace users",
            "ranked marketplace listings using engagement signals",
            "built a personalized feed for repeat buyers",
            "improved search results for customer queries",
            "matched users to jobs with better relevance",
        ]
        with tempfile.TemporaryDirectory() as td:
            run_path = ensure_run_dirs(Path(td) / "run")
            normalized_rows = []
            context_rows = []
            for index, phrase in enumerate(phrases, 1):
                candidate_id = f"CAND_{index:07d}"
                normalized_rows.append(normalized_row(candidate_id, phrase, "", "Python"))
                context_rows.append(
                    context_row(
                        candidate_id,
                        "Engineer",
                        phrase,
                        "Shipped customer-facing search and recommendation features.",
                        [{"index": 0, "name": "Python", "proficiency": "intermediate", "duration_months": 24}],
                    )
                )
            self.write_inputs(run_path, normalized_rows, context_rows)

            result = run_evidence_extraction(run_path)
            summary = pd.read_parquet(result["evidence_summary_parquet"])
            self.assertTrue((summary["retrieval_ranking_relevance_score"] > 0.4).all())


if __name__ == "__main__":
    unittest.main()

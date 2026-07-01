import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evidence import run_evidence_extraction
from src.paths import ensure_run_dirs


def candidate(candidate_id: str, title: str, description: str, summary: str, skills: list[dict]) -> dict:
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
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "notice_period_days": 30,
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "interview_completion_rate": 0.9,
            "skill_assessment_scores": {"Python": 91.0},
        },
    }


class EvidenceLedgerTest(unittest.TestCase):
    def test_evidence_stage_emits_reconstructible_ledger_and_one_row_summary(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            run_path = ensure_run_dirs(td_path / "run")
            normalized_dir = run_path / "normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)

            normalized_rows = [
                {
                    "candidate_id": "CAND_0000001",
                    "combined_career_text": "Improved product discovery and ranked marketplace listings for live users.",
                    "combined_profile_text": "Built recommendations that improved search results.",
                    "combined_skill_text": "Python Elasticsearch",
                    "country": "India",
                    "preferred_work_mode": "remote",
                    "willing_to_relocate": True,
                    "open_to_work_flag": True,
                    "recruiter_response_rate": 0.8,
                    "avg_response_time_hours": 12,
                    "notice_period_days": 30,
                    "github_activity_score": 70,
                    "interview_completion_rate": 0.9,
                    "years_of_experience": 6,
                    "anomaly_severity": "none",
                },
                {
                    "candidate_id": "CAND_0000002",
                    "combined_career_text": "Built dashboards for internal reporting.",
                    "combined_profile_text": "Interested in AI.",
                    "combined_skill_text": "LLM Embeddings Pinecone",
                    "country": "India",
                    "preferred_work_mode": "remote",
                    "willing_to_relocate": False,
                    "open_to_work_flag": True,
                    "recruiter_response_rate": 0.6,
                    "avg_response_time_hours": 24,
                    "notice_period_days": 60,
                    "github_activity_score": 10,
                    "interview_completion_rate": 0.7,
                    "years_of_experience": 6,
                    "anomaly_severity": "none",
                },
            ]
            pd.DataFrame(normalized_rows).to_parquet(normalized_dir / "candidates_normalized.parquet", index=False)
            with (normalized_dir / "candidates_normalized.jsonl").open("w", encoding="utf-8") as handle:
                for row in normalized_rows:
                    handle.write(json.dumps(row) + "\n")

            context_rows = [
                candidate(
                    "CAND_0000001",
                    "Software Engineer",
                    "Improved product discovery, tuned relevance, and ranked marketplace listings for live users.",
                    "Built recommendations and improved search results.",
                    [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
                ),
                candidate(
                    "CAND_0000002",
                    "Analyst",
                    "Built dashboards for reporting.",
                    "Interested in AI and learning Pinecone tutorials.",
                    [{"name": "LLM", "proficiency": "advanced", "duration_months": 12}],
                ),
            ]
            with (normalized_dir / "candidate_context.jsonl").open("w", encoding="utf-8") as handle:
                for row in context_rows:
                    handle.write(json.dumps(row) + "\n")

            result = run_evidence_extraction(run_path)

            ledger_path = Path(result["evidence_ledger_jsonl"])
            summary_path = Path(result["evidence_summary_parquet"])
            self.assertTrue(ledger_path.exists())
            self.assertTrue(summary_path.exists())

            ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(any(row["source_path"] == "career_history[0].description" for row in ledger_rows))
            self.assertTrue(any(row["normalized_category"] == "retrieval_ranking_relevance" for row in ledger_rows))
            self.assertTrue(all("exact_source_excerpt" in row and row["exact_source_excerpt"] for row in ledger_rows))

            summary = pd.read_parquet(summary_path)
            self.assertEqual(summary["candidate_id"].nunique(), 2)
            self.assertEqual(len(summary), 2)

    def test_evidence_profile_artifacts_are_written_when_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            run_path = ensure_run_dirs(td_path / "run")
            normalized_dir = run_path / "normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)

            normalized_rows = [
                {
                    "candidate_id": "CAND_0000001",
                    "combined_career_text": "Improved product discovery and ranked marketplace listings for live users.",
                    "combined_profile_text": "Built recommendations that improved search results.",
                    "combined_skill_text": "Python Elasticsearch",
                    "country": "India",
                    "preferred_work_mode": "remote",
                    "willing_to_relocate": True,
                    "open_to_work_flag": True,
                    "recruiter_response_rate": 0.8,
                    "avg_response_time_hours": 12,
                    "notice_period_days": 30,
                    "github_activity_score": 70,
                    "interview_completion_rate": 0.9,
                    "years_of_experience": 6,
                    "anomaly_severity": "none",
                }
            ]
            pd.DataFrame(normalized_rows).to_parquet(normalized_dir / "candidates_normalized.parquet", index=False)
            with (normalized_dir / "candidate_context.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        candidate(
                            "CAND_0000001",
                            "Software Engineer",
                            "Improved product discovery, tuned relevance, and ranked marketplace listings for live users.",
                            "Built recommendations and improved search results.",
                            [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
                        )
                    )
                    + "\n"
                )

            result = run_evidence_extraction(run_path, profile=True)

            profile_json = Path(result["evidence_profile_json"])
            profile_md = Path(result["evidence_profile_md"])
            self.assertTrue(profile_json.exists())
            self.assertTrue(profile_md.exists())

            profile = json.loads(profile_json.read_text(encoding="utf-8"))
            self.assertIn("elapsed_seconds", profile)
            self.assertIn("candidate_throughput_per_second", profile)
            self.assertIn("evidence_items_per_candidate", profile)
            self.assertIn("timings", profile)
            self.assertIn("per_source_field_seconds", profile["timings"])
            self.assertIn("phrase_matching_seconds", profile["timings"])


if __name__ == "__main__":
    unittest.main()

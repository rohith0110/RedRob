import json
import tempfile
from pathlib import Path
import unittest

from src.normalization import normalize_candidate, run_normalization
from src.paths import ensure_run_dirs


class NormalizationTest(unittest.TestCase):
    def test_one_row_per_valid_candidate(self):
        candidate = {
            "candidate_id": "CAND_0000003",
            "profile": {
                "headline": "Search Engineer",
                "summary": "Built ranking systems.",
                "current_title": "Senior Engineer",
                "current_company": "Acme",
                "current_industry": "Software",
                "location": "Bengaluru",
                "country": "India",
                "years_of_experience": 7,
            },
            "career_history": [
                {
                    "title": "Search Engineer",
                    "company": "Acme",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "description": "Owned retrieval and ranking pipeline in Python.",
                }
            ],
            "education": [],
            "skills": [{"name": "Python", "duration_months": 72}],
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
            },
        }

        row = normalize_candidate(candidate)
        self.assertEqual(row["candidate_id"], "CAND_0000003")
        self.assertIn("retrieval", row["combined_candidate_text"])
        self.assertEqual(row["career_entry_count"], 1)
        self.assertEqual(row["skills_count"], 1)

    def test_phase2_candidate_context_preserves_structured_source_fields(self):
        candidate = {
            "candidate_id": "CAND_0000003",
            "profile": {
                "headline": "Search Engineer",
                "summary": "Built ranking systems.",
                "current_title": "Senior Engineer",
                "current_company": "Acme",
                "current_industry": "Software",
                "location": "Bengaluru",
                "country": "India",
                "years_of_experience": 7,
            },
            "career_history": [
                {
                    "title": "Search Engineer",
                    "company": "Acme",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "description": "Owned retrieval and ranking pipeline in Python.",
                }
            ],
            "education": [],
            "skills": [{"name": "Python", "duration_months": 72}],
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
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)

            context_path = run_path / "normalized" / "candidate_context.jsonl"
            self.assertTrue(context_path.exists())
            record = json.loads(context_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["candidate_id"], "CAND_0000003")
            self.assertEqual(record["career_history"][0]["index"], 0)
            self.assertEqual(record["career_history"][0]["description"], "Owned retrieval and ranking pipeline in Python.")
            self.assertEqual(record["skills"][0]["index"], 0)
            self.assertEqual(record["redrob_signals"]["notice_period_days"], 30)
            self.assertEqual(record["skill_assessments"][0]["name"], "Python")


if __name__ == "__main__":
    unittest.main()

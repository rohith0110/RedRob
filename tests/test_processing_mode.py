import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.baseline_scoring import run_scoring
from src.normalization import run_normalization
from src.paths import ensure_run_dirs


def candidate(candidate_id: str, description: str, skills: list[dict], signals: dict) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": "Engineer",
            "summary": description,
            "current_title": "Engineer",
            "current_company": "Acme",
            "current_industry": "Software",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 6,
        },
        "career_history": [
            {
                "title": "Engineer",
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
        "redrob_signals": signals,
    }


class ProcessingModeTest(unittest.TestCase):
    def write_fixture(self, path: Path) -> None:
        rows = [
            candidate(
                "CAND_0000001",
                "Built personalized feed ranking and improved search results for live users.",
                [{"name": "Python", "duration_months": 48}],
                {
                    "profile_completeness_score": 90,
                    "last_active_date": "2026-06-01",
                    "open_to_work_flag": True,
                    "recruiter_response_rate": 0.7,
                    "avg_response_time_hours": 12,
                    "notice_period_days": 30,
                    "preferred_work_mode": "remote",
                    "willing_to_relocate": True,
                    "github_activity_score": 70,
                    "interview_completion_rate": 0.9,
                },
            ),
            candidate(
                "CAND_0000002",
                "Built dashboards and ETL jobs for finance reporting.",
                [{"name": "AI", "duration_months": 12}, {"name": "Embeddings", "duration_months": 12}],
                {
                    "profile_completeness_score": 80,
                    "last_active_date": "2026-05-15",
                    "open_to_work_flag": True,
                    "recruiter_response_rate": 0.9,
                    "avg_response_time_hours": 2,
                    "notice_period_days": 0,
                    "preferred_work_mode": "remote",
                    "willing_to_relocate": False,
                    "github_activity_score": 90,
                    "interview_completion_rate": 0.95,
                },
            ),
        ]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_memory_and_chunked_modes_produce_identical_rankings(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            candidates_path = td_path / "candidates.jsonl"
            self.write_fixture(candidates_path)

            memory_run = ensure_run_dirs(td_path / "memory")
            chunked_run = ensure_run_dirs(td_path / "chunked")

            run_normalization(candidates_path, memory_run, processing_mode="memory")
            run_normalization(candidates_path, chunked_run, processing_mode="chunked", chunk_size=1)

            memory_normalized = pd.read_parquet(memory_run / "normalized" / "candidates_normalized.parquet").to_dict("records")
            chunked_normalized = pd.read_parquet(chunked_run / "normalized" / "candidates_normalized.parquet").to_dict("records")
            self.assertEqual(memory_normalized, chunked_normalized)

            memory_scores = run_scoring(memory_run)
            chunked_scores = run_scoring(chunked_run)
            memory_breakdown = pd.read_csv(memory_scores["score_breakdown_csv"]).to_dict("records")
            chunked_breakdown = pd.read_csv(chunked_scores["score_breakdown_csv"]).to_dict("records")
            self.assertEqual(memory_breakdown, chunked_breakdown)


if __name__ == "__main__":
    unittest.main()

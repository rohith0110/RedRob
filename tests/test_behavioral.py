import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.behavioral import run_behavioral_analysis
from src.evidence import run_evidence_extraction
from src.normalization import run_normalization
from src.paths import ensure_run_dirs


def candidate(
    candidate_id: str,
    description: str,
    response_rate: float,
    response_time: float,
    notice_period: int,
    work_mode: str,
    github_activity: float,
    assessment_scores: dict,
) -> dict:
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
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
        "certifications": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": response_rate,
            "avg_response_time_hours": response_time,
            "notice_period_days": notice_period,
            "preferred_work_mode": work_mode,
            "willing_to_relocate": False,
            "github_activity_score": github_activity,
            "interview_completion_rate": 0.8,
            "skill_assessment_scores": assessment_scores,
        },
    }


class BehavioralTest(unittest.TestCase):
    def test_availability_multiplier_is_bounded_and_respects_signal_strength(self):
        fast = candidate(
            "CAND_0000001",
            "Improved search results and shipped ranking systems in production.",
            response_rate=0.9,
            response_time=4,
            notice_period=15,
            work_mode="remote",
            github_activity=70,
            assessment_scores={"Python": 90},
        )
        slow = candidate(
            "CAND_0000002",
            "Improved search results and shipped ranking systems in production.",
            response_rate=0.1,
            response_time=240,
            notice_period=120,
            work_mode="onsite",
            github_activity=0,
            assessment_scores={},
        )
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in [fast, slow]) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            result = run_behavioral_analysis(run_path)

            breakdown = pd.read_parquet(result["availability_breakdown_parquet"]).set_index("candidate_id")
            self.assertGreater(
                breakdown.loc["CAND_0000001", "availability_multiplier"],
                breakdown.loc["CAND_0000002", "availability_multiplier"],
            )
            self.assertTrue((breakdown["availability_multiplier"] >= 0.72).all())
            self.assertTrue((breakdown["availability_multiplier"] <= 1.08).all())


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.behavioral import run_behavioral_analysis
from src.credibility import run_credibility_analysis
from src.evidence import run_evidence_extraction
from src.normalization import run_normalization
from src.paths import ensure_run_dirs
from src.v2_scoring import run_v2_scoring


def candidate(
    candidate_id: str,
    title: str,
    summary: str,
    description: str,
    skills: list[dict],
    response_rate: float,
    response_time: float,
    github_activity: float,
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
            "avg_response_time_hours": response_time,
            "notice_period_days": 30,
            "preferred_work_mode": "remote",
            "willing_to_relocate": False,
            "github_activity_score": github_activity,
            "interview_completion_rate": 0.8,
            "skill_assessment_scores": {"Python": 88},
        },
    }


class V2ScoringTest(unittest.TestCase):
    def test_relevant_candidate_beats_high_activity_skill_stuffer_and_research_profile(self):
        relevant = candidate(
            "CAND_0000001",
            "Software Engineer",
            "Built recommendations and improved search results.",
            "Improved product discovery, tuned relevance, and shipped ranking systems for live users.",
            [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
            response_rate=0.5,
            response_time=24,
            github_activity=15,
        )
        skill_stuffer = candidate(
            "CAND_0000002",
            "Analyst",
            "Interested in AI and agent tools.",
            "Built internal dashboards for weekly reporting.",
            [
                {"name": "LLM", "proficiency": "advanced", "duration_months": 12},
                {"name": "Embeddings", "proficiency": "advanced", "duration_months": 12},
                {"name": "Pinecone", "proficiency": "advanced", "duration_months": 12},
            ],
            response_rate=1.0,
            response_time=1,
            github_activity=95,
        )
        research_only = candidate(
            "CAND_0000003",
            "Research Engineer",
            "Published work on neural retrieval and LangChain demos.",
            "Ran research experiments and tutorials with no deployment or production ownership.",
            [{"name": "LangChain", "proficiency": "advanced", "duration_months": 18}],
            response_rate=0.9,
            response_time=2,
            github_activity=80,
        )
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in [relevant, skill_stuffer, research_only]) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")

            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            run_credibility_analysis(run_path)
            run_behavioral_analysis(run_path)
            result = run_v2_scoring(run_path)

            breakdown = pd.read_parquet(result["score_breakdown_v2_parquet"]).sort_values("rank")
            self.assertEqual(list(breakdown["candidate_id"])[0], "CAND_0000001")
            self.assertEqual(len(breakdown), 3)


if __name__ == "__main__":
    unittest.main()

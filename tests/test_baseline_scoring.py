import unittest

from src.baseline_scoring import score_candidate_rows
from src.submission_writer import rank_rows


class BaselineScoringTest(unittest.TestCase):
    def test_career_retrieval_evidence_beats_skill_only_ai(self):
        rows = [
            {
                "candidate_id": "CAND_0000001",
                "combined_career_text": "owned semantic search retrieval ranking evaluation pipeline",
                "combined_profile_text": "",
                "combined_skill_text": "",
                "years_of_experience": 6,
                "country": "India",
                "preferred_work_mode": "remote",
                "willing_to_relocate": False,
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.5,
                "avg_response_time_hours": 24,
                "notice_period_days": 30,
                "github_activity_score": 50,
                "interview_completion_rate": 0.8,
                "anomaly_severity": "none",
            },
            {
                "candidate_id": "CAND_0000002",
                "combined_career_text": "frontend dashboards",
                "combined_profile_text": "",
                "combined_skill_text": "AI ML embeddings ranking",
                "years_of_experience": 6,
                "country": "India",
                "preferred_work_mode": "remote",
                "willing_to_relocate": False,
                "open_to_work_flag": True,
                "recruiter_response_rate": 1,
                "avg_response_time_hours": 1,
                "notice_period_days": 0,
                "github_activity_score": 100,
                "interview_completion_rate": 1,
                "anomaly_severity": "none",
            },
        ]
        scored = score_candidate_rows(rows)
        self.assertGreater(scored[0]["final_score"], scored[1]["final_score"])

    def test_tie_break_uses_candidate_id_ascending(self):
        rows = [
            {"candidate_id": "CAND_0000002", "final_score": 1, "top_positive_evidence": "", "top_negative_evidence": ""},
            {"candidate_id": "CAND_0000001", "final_score": 1, "top_positive_evidence": "", "top_negative_evidence": ""},
        ]
        ranked = rank_rows(rows, limit=2)
        self.assertEqual([r["candidate_id"] for r in ranked], ["CAND_0000001", "CAND_0000002"])


if __name__ == "__main__":
    unittest.main()

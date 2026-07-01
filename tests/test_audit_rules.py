import unittest
import json
import tempfile
from pathlib import Path

from src.audit import audit_candidate, run_audit


class AuditRulesTest(unittest.TestCase):
    def test_detects_synthetic_bad_examples(self):
        candidate = {
            "candidate_id": "CAND_0000002",
            "profile": {"years_of_experience": 1},
            "career_history": [
                {
                    "title": "Engineer",
                    "start_date": "2025-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 99,
                    "is_current": True,
                    "description": "",
                },
                {
                    "title": "Architect",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "duration_months": 12,
                    "is_current": True,
                    "description": "",
                },
            ],
            "skills": [{"name": "Python", "duration_months": 120}],
            "redrob_signals": {
                "signup_date": "2026-01-02",
                "last_active_date": "2026-01-01",
                "expected_salary_range_inr_lpa": {"min": 40, "max": 20},
            },
        }

        issues = audit_candidate(candidate)
        kinds = {issue["type"] for issue in issues}
        self.assertIn("signup_after_last_active", kinds)
        self.assertIn("role_end_before_start", kinds)
        self.assertIn("multiple_current_roles", kinds)
        self.assertIn("salary_min_greater_than_max", kinds)
        self.assertIn("skill_duration_implausible", kinds)

    def test_audit_summary_keeps_zero_count_fields(self):
        candidate = {
            "candidate_id": "CAND_0000004",
            "profile": {"country": "India", "location": "Bengaluru", "current_title": "Engineer", "years_of_experience": 5},
            "career_history": [{"title": "Engineer", "start_date": "2020-01-01", "end_date": None, "duration_months": 60, "is_current": True, "description": "Python backend"}],
            "education": [],
            "skills": [],
            "redrob_signals": {"open_to_work_flag": True, "preferred_work_mode": "remote"},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidates.jsonl"
            path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            summary = run_audit(path, Path(td) / "run")
            self.assertEqual(summary["malformed_records"], 0)
            self.assertEqual(summary["duplicate_candidate_ids"], 0)


if __name__ == "__main__":
    unittest.main()

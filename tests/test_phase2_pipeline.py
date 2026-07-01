import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.behavioral import run_behavioral_analysis
from src.cli import main
from src.credibility import run_credibility_analysis
from src.evidence import run_evidence_extraction
from src.normalization import run_normalization
from src.paths import ensure_run_dirs
from src.submission_validation import require_valid_submission
from src.v2_scoring import run_v2_scoring


def make_candidate(
    candidate_id: str,
    title: str,
    summary: str,
    description: str,
    skills: list[dict],
    response_rate: float = 0.5,
    response_time: float = 24,
    github_activity: float = 10,
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


class Phase2PipelineTest(unittest.TestCase):
    def test_breakdowns_have_one_row_per_candidate_and_scoring_is_deterministic(self):
        candidates = [
            make_candidate("CAND_0000001", "Software Engineer", "Built recommendations.", "Improved product discovery and shipped ranking systems.", [{"name": "Python", "proficiency": "advanced", "duration_months": 48}]),
            make_candidate("CAND_0000002", "Analyst", "Interested in AI.", "Built dashboards for reporting.", [{"name": "LLM", "proficiency": "advanced", "duration_months": 12}], response_rate=1.0, response_time=1, github_activity=90),
            make_candidate("CAND_0000003", "Research Engineer", "Neural retrieval demos.", "Ran research experiments with no deployment.", [{"name": "LangChain", "proficiency": "advanced", "duration_months": 18}]),
            make_candidate("CAND_0000004", "Platform Engineer", "Shipped backend APIs.", "Built Python APIs and production services.", [{"name": "Python", "proficiency": "advanced", "duration_months": 48}]),
            make_candidate("CAND_0000005", "Search Engineer", "Improved search results.", "Tuned relevance and measured ranking outcomes.", [{"name": "Python", "proficiency": "advanced", "duration_months": 48}]),
        ]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in candidates) + "\n", encoding="utf-8")

            def run_pipeline(target: Path) -> pd.DataFrame:
                run_path = ensure_run_dirs(target)
                run_normalization(dataset, run_path)
                run_evidence_extraction(run_path)
                run_credibility_analysis(run_path)
                run_behavioral_analysis(run_path)
                result = run_v2_scoring(run_path)
                self.assertEqual(len(pd.read_parquet(run_path / "evidence" / "evidence_summary.parquet")), 5)
                self.assertEqual(len(pd.read_parquet(run_path / "credibility" / "credibility_breakdown.parquet")), 5)
                self.assertEqual(len(pd.read_parquet(run_path / "behavioral" / "availability_breakdown.parquet")), 5)
                self.assertEqual(len(pd.read_parquet(result["score_breakdown_v2_parquet"])), 5)
                return pd.read_parquet(result["score_breakdown_v2_parquet"]).sort_values("rank").reset_index(drop=True)

            first = run_pipeline(td_path / "run_one")
            second = run_pipeline(td_path / "run_two")
            pd.testing.assert_frame_equal(first[["candidate_id", "rank", "final_score"]], second[["candidate_id", "rank", "final_score"]])

    def test_v2_cli_writes_valid_submission_and_benchmark_report(self):
        candidates = [
            make_candidate(
                f"CAND_{index:07d}",
                "Software Engineer" if index % 3 else "Search Engineer",
                "Built recommendations and improved search results." if index % 2 else "Shipped Python APIs.",
                "Improved product discovery and shipped ranking systems for live users." if index % 4 else "Built Python APIs and backend services.",
                [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
                response_rate=0.5 + (index % 5) * 0.1,
                response_time=12 + (index % 6) * 6,
                github_activity=10 + index % 20,
            )
            for index in range(1, 101)
        ]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            out_path = td_path / "v2_submission.csv"
            dataset.write_text("\n".join(json.dumps(row) for row in candidates) + "\n", encoding="utf-8")
            exit_code = main(["--mode", "v2", "--candidates", str(dataset), "--out", str(out_path), "--run-id", "v2_test", "--run-dir", str(td_path / "runs")])
            self.assertEqual(exit_code, 0)
            require_valid_submission(out_path)
            run_path = td_path / "runs" / "v2_test"
            benchmark = subprocess.check_output(["python", "scripts/benchmark_runtime.py", "--run-path", str(run_path)], text=True, cwd=Path.cwd())
            report = json.loads(benchmark)
            self.assertIn("elapsed_seconds", report)
            self.assertIn("stage_timings", report)
            self.assertEqual(report["run_kind"], "fresh")
            self.assertTrue((run_path / "reports" / "reasoning_style_quality_report.json").exists())
            self.assertTrue((run_path / "reports" / "reasoning_style_quality_report.md").exists())
            self.assertTrue((run_path / "reasoning" / "reasoning_style_lint.jsonl").exists())
            self.assertTrue((run_path / "benchmarks" / "runtime_benchmark.json").exists())
            self.assertTrue((run_path / "benchmarks" / "runtime_benchmark.md").exists())
            manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("working_tree_dirty", manifest)
            self.assertIsInstance(manifest["working_tree_dirty"], bool)

    def test_relevant_candidate_beats_consulting_research_and_cv_variants(self):
        candidates = [
            make_candidate("CAND_0000001", "Software Engineer", "Built recommendations and improved search results.", "Improved product discovery, tuned relevance, and shipped ranking systems for live users.", [{"name": "Python", "proficiency": "advanced", "duration_months": 48}]),
            make_candidate("CAND_0000002", "Consultant", "Client delivery role.", "Provided generic consulting support across projects.", [{"name": "Python", "proficiency": "advanced", "duration_months": 24}]),
            make_candidate("CAND_0000003", "Consultant", "Client delivery role.", "Previously built search relevance systems before moving into consulting.", [{"name": "Python", "proficiency": "advanced", "duration_months": 48}]),
            make_candidate("CAND_0000004", "Research Engineer", "LangChain demos.", "Ran tutorials and research experiments with no production deployment.", [{"name": "LangChain", "proficiency": "advanced", "duration_months": 18}], response_rate=0.9, response_time=2, github_activity=80),
            make_candidate("CAND_0000005", "CV Engineer", "Speech and robotics.", "Built computer vision, speech recognition, and robotics systems.", [{"name": "Speech Recognition", "proficiency": "advanced", "duration_months": 24}], response_rate=0.9, response_time=2, github_activity=70),
        ]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset = td_path / "candidates.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in candidates) + "\n", encoding="utf-8")
            run_path = ensure_run_dirs(td_path / "run")
            run_normalization(dataset, run_path)
            run_evidence_extraction(run_path)
            run_credibility_analysis(run_path)
            run_behavioral_analysis(run_path)
            result = run_v2_scoring(run_path)
            ranked = pd.read_parquet(result["score_breakdown_v2_parquet"]).sort_values("rank").set_index("candidate_id")
            self.assertEqual(ranked.index[0], "CAND_0000001")
            self.assertLess(ranked.loc["CAND_0000004", "rank"], len(ranked) + 1)
            self.assertLess(ranked.loc["CAND_0000003", "rank"], ranked.loc["CAND_0000002", "rank"])


if __name__ == "__main__":
    unittest.main()

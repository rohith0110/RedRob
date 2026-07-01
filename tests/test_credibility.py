import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

import src.audit as audit_module
import src.behavioral as behavioral_module
import src.credibility as credibility_module
import src.evidence as evidence_module
from src.audit import audit_candidate, parse_date
from src.credibility import run_credibility_analysis, validate_credibility_rule_registry
from src.evidence import run_evidence_extraction
from src.normalization import run_normalization
from src.paths import ensure_run_dirs
from src.phase2_config import load_phase2_config


def candidate(
    candidate_id: str,
    profile_years: float,
    career_history: list[dict],
    skills: list[dict],
    redrob_signals: dict,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": "Engineer",
            "summary": "Profile summary",
            "current_title": "Engineer",
            "current_company": "Acme",
            "current_industry": "Software",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": profile_years,
        },
        "career_history": career_history,
        "education": [],
        "skills": skills,
        "certifications": [],
        "redrob_signals": redrob_signals,
    }


def contradictory_candidate(candidate_id: str = "CAND_0000001") -> dict:
    return candidate(
        candidate_id,
        14,
        [
            {
                "title": "Search Engineer",
                "company": "Acme",
                "start_date": "2025-01-01",
                "end_date": None,
                "duration_months": 18,
                "is_current": True,
                "description": "Built dashboards for reporting.",
            },
            {
                "title": "Lead Engineer",
                "company": "Beta",
                "start_date": "2026-03-01",
                "end_date": None,
                "duration_months": 16,
                "is_current": True,
                "description": "Internal tooling only.",
            },
        ],
        [
            {"name": "LLM", "proficiency": "advanced", "duration_months": 120},
            {"name": "Pinecone", "proficiency": "advanced", "duration_months": 120},
        ],
        {
            "profile_completeness_score": 90,
            "signup_date": "2026-06-10",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "notice_period_days": 30,
            "preferred_work_mode": "remote",
            "willing_to_relocate": False,
            "github_activity_score": 30,
            "interview_completion_rate": 0.8,
            "expected_salary_range_inr_lpa": {"min": 40, "max": 20},
        },
    )


def clean_candidate(candidate_id: str = "CAND_0000002") -> dict:
    return candidate(
        candidate_id,
        6,
        [
            {
                "title": "Search Engineer",
                "company": "Acme",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "description": "Improved search results and shipped ranking systems in production.",
            }
        ],
        [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
        {
            "profile_completeness_score": 90,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "notice_period_days": 30,
            "preferred_work_mode": "remote",
            "willing_to_relocate": False,
            "github_activity_score": 30,
            "interview_completion_rate": 0.8,
            "expected_salary_range_inr_lpa": {"min": 18, "max": 20},
        },
    )


class CredibilityTest(unittest.TestCase):
    def _prepare_run(self, td: str, rows: list[dict], *, dataset_reference_date: str = "2026-06-30") -> Path:
        td_path = Path(td)
        dataset = td_path / "candidates.jsonl"
        dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        run_path = ensure_run_dirs(td_path / "run")
        run_normalization(dataset, run_path, dataset_reference_date=parse_date(dataset_reference_date))
        run_evidence_extraction(run_path, dataset_reference_date=parse_date(dataset_reference_date))
        return run_path

    def _copy_phase2_config(self, td: str, mutate=None):
        root = Path(td)
        paths = {}
        for name in (
            "role_rubric.yaml",
            "evidence_patterns.yaml",
            "scoring_weights.yaml",
            "credibility_rules.yaml",
            "behavioral_rules.yaml",
            "semantic_config.yaml",
        ):
            target = root / name
            target.write_text((Path("configs") / name).read_text(encoding="utf-8"), encoding="utf-8")
            paths[name] = target
        if mutate:
            data = yaml.safe_load(paths["credibility_rules.yaml"].read_text(encoding="utf-8"))
            mutate(data)
            paths["credibility_rules.yaml"].write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return load_phase2_config(
            paths["role_rubric.yaml"],
            paths["evidence_patterns.yaml"],
            paths["scoring_weights.yaml"],
            paths["credibility_rules.yaml"],
            paths["behavioral_rules.yaml"],
            paths["semantic_config.yaml"],
        )

    def test_compounded_contradictions_reduce_multiplier_more_than_clean_profile(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = self._prepare_run(td, [contradictory_candidate(), clean_candidate()])
            result = run_credibility_analysis(run_path, dataset_reference_date=parse_date("2026-06-30"))

            breakdown = pd.read_parquet(result["credibility_breakdown_parquet"]).set_index("candidate_id")
            self.assertLess(
                breakdown.loc["CAND_0000001", "credibility_multiplier"],
                breakdown.loc["CAND_0000002", "credibility_multiplier"],
            )
            self.assertLess(breakdown.loc["CAND_0000001", "credibility_multiplier"], 0.78)
            self.assertGreater(breakdown.loc["CAND_0000002", "credibility_multiplier"], 0.85)

    def test_reference_date_changes_overlap_result_predictably(self):
        row = contradictory_candidate()
        with tempfile.TemporaryDirectory() as td_early, tempfile.TemporaryDirectory() as td_late:
            early_run = self._prepare_run(td_early, [row], dataset_reference_date="2026-06-30")
            late_run = self._prepare_run(td_late, [row], dataset_reference_date="2027-04-30")

            early = run_credibility_analysis(early_run, dataset_reference_date=parse_date("2026-06-30"))
            late = run_credibility_analysis(late_run, dataset_reference_date=parse_date("2027-04-30"))

            early_breakdown = pd.read_parquet(early["credibility_breakdown_parquet"]).set_index("candidate_id")
            late_breakdown = pd.read_parquet(late["credibility_breakdown_parquet"]).set_index("candidate_id")
            self.assertGreater(
                late_breakdown.loc["CAND_0000001", "triggered_rule_count"],
                early_breakdown.loc["CAND_0000001", "triggered_rule_count"],
            )
            early_rules = Path(early["credibility_rules_triggered_jsonl"]).read_text(encoding="utf-8")
            late_rules = Path(late["credibility_rules_triggered_jsonl"]).read_text(encoding="utf-8")
            self.assertNotIn("excessive_overlap", early_rules)
            self.assertIn("excessive_overlap", late_rules)

    def test_audit_and_credibility_share_the_same_reference_date_logic(self):
        row = candidate(
            "CAND_0000003",
            8,
            [
                {
                    "title": "Engineer",
                    "company": "Acme",
                    "start_date": "2024-01-01",
                    "end_date": "2026-07-15",
                    "duration_months": 30,
                    "is_current": True,
                    "description": "Backend engineering work.",
                }
            ],
            [{"name": "Python", "proficiency": "advanced", "duration_months": 24}],
            {
                "profile_completeness_score": 90,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-05-01",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 12,
                "notice_period_days": 30,
                "preferred_work_mode": "remote",
                "willing_to_relocate": False,
                "github_activity_score": 30,
                "interview_completion_rate": 0.8,
                "expected_salary_range_inr_lpa": {"min": 18, "max": 20},
            },
        )
        issues = audit_candidate(row, reference_date=parse_date("2026-08-01"))
        kinds = {issue["type"] for issue in issues}
        self.assertIn("current_role_has_past_end_date", kinds)

        with tempfile.TemporaryDirectory() as td:
            run_path = self._prepare_run(td, [row], dataset_reference_date="2026-08-01")
            result = run_credibility_analysis(run_path, dataset_reference_date=parse_date("2026-08-01"))
            triggered = [
                json.loads(line)
                for line in Path(result["credibility_rules_triggered_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("current_role_has_past_end_date", {row["rule_name"] for row in triggered})

    def test_title_description_contradiction_is_fully_config_governed(self):
        with tempfile.TemporaryDirectory() as td_enabled, tempfile.TemporaryDirectory() as td_disabled:
            enabled_run = self._prepare_run(td_enabled, [contradictory_candidate()])
            disabled_run = self._prepare_run(td_disabled, [contradictory_candidate()])

            enabled_config = self._copy_phase2_config(td_enabled)
            disabled_config = self._copy_phase2_config(
                td_disabled,
                mutate=lambda data: data["rules"]["title_description_contradiction"].update({"enabled": False}),
            )

            enabled = run_credibility_analysis(
                enabled_run,
                config=enabled_config,
                dataset_reference_date=parse_date("2026-06-30"),
            )
            disabled = run_credibility_analysis(
                disabled_run,
                config=disabled_config,
                dataset_reference_date=parse_date("2026-06-30"),
            )

            enabled_breakdown = pd.read_parquet(enabled["credibility_breakdown_parquet"]).set_index("candidate_id")
            disabled_breakdown = pd.read_parquet(disabled["credibility_breakdown_parquet"]).set_index("candidate_id")
            enabled_rules = Path(enabled["credibility_rules_triggered_jsonl"]).read_text(encoding="utf-8")
            disabled_rules = Path(disabled["credibility_rules_triggered_jsonl"]).read_text(encoding="utf-8")

            self.assertGreater(
                enabled_breakdown.loc["CAND_0000001", "triggered_rule_count"],
                disabled_breakdown.loc["CAND_0000001", "triggered_rule_count"],
            )
            self.assertIn("title_description_contradiction", enabled_rules)
            self.assertNotIn("title_description_contradiction", disabled_rules)

    def test_validation_fails_when_enabled_rule_lacks_handler(self):
        config = self._copy_phase2_config(
            tempfile.mkdtemp(),
            mutate=lambda data: data["rules"].update(
                {
                    "imaginary_rule": {
                        "enabled": True,
                        "severity": "minor",
                        "score_contribution": 0.01,
                        "explanation_label": "imaginary",
                        "candidate_facing_reasoning": False,
                    }
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "imaginary_rule"):
            validate_credibility_rule_registry(config)

    def test_business_logic_modules_do_not_embed_hardcoded_reference_date(self):
        for module in (audit_module, evidence_module, credibility_module, behavioral_module):
            source = inspect.getsource(module)
            self.assertNotIn("2026-06-27", source)
            self.assertNotIn("2026-06-30", source)


if __name__ == "__main__":
    unittest.main()

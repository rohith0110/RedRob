import tempfile
import unittest
from pathlib import Path

import yaml

from src.baseline_scoring import score_candidate_rows
from src.scoring_config import load_scoring_config


def row(text="owned semantic search retrieval ranking pipeline"):
    return {
        "candidate_id": "CAND_0000001",
        "combined_career_text": text,
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
        "anomaly_severity": "none",
    }


class ScoringConfigTest(unittest.TestCase):
    def write_configs(self, td):
        td = Path(td)
        weights = td / "baseline_weights.yaml"
        rubric = td / "role_rubric.yaml"
        weights.write_text(Path("configs/baseline_weights.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        rubric.write_text(Path("configs/role_rubric.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        return weights, rubric

    def test_weight_change_changes_score_predictably(self):
        with tempfile.TemporaryDirectory() as td:
            weights, rubric = self.write_configs(td)
            base = score_candidate_rows([row()], load_scoring_config(weights, rubric))[0]["final_score"]
            data = yaml.safe_load(weights.read_text(encoding="utf-8"))
            data["positive_score_weights"]["career_text_relevance"] = 0.50
            data["positive_score_weights"]["production_evaluation"] = 0.10
            weights.write_text(yaml.safe_dump(data), encoding="utf-8")

            changed = score_candidate_rows([row()], load_scoring_config(weights, rubric))[0]["final_score"]
            self.assertGreater(changed, base)

    def test_rubric_phrase_change_changes_evidence_detection(self):
        with tempfile.TemporaryDirectory() as td:
            weights, rubric = self.write_configs(td)
            candidate = row("built graph search systems")
            base = score_candidate_rows([candidate], load_scoring_config(weights, rubric))[0]
            data = yaml.safe_load(rubric.read_text(encoding="utf-8"))
            data["concept_categories"]["role_relevance"]["positive_phrases"].append("graph search")
            rubric.write_text(yaml.safe_dump(data), encoding="utf-8")

            changed = score_candidate_rows([candidate], load_scoring_config(weights, rubric))[0]
            self.assertLess(base["base_relevance_score"], changed["base_relevance_score"])
            self.assertIn("career history mentions", changed["top_positive_evidence"])

    def test_invalid_weight_sum_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            weights, rubric = self.write_configs(td)
            data = yaml.safe_load(weights.read_text(encoding="utf-8"))
            data["positive_score_weights"]["career_text_relevance"] = 0.99
            weights.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                load_scoring_config(weights, rubric)

    def test_unknown_scoring_config_key_fails(self):
        with tempfile.TemporaryDirectory() as td:
            weights, rubric = self.write_configs(td)
            data = yaml.safe_load(weights.read_text(encoding="utf-8"))
            data["unused_scoring_knob"] = 1
            weights.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_scoring_config(weights, rubric)

    def test_scoring_is_deterministic_when_configs_are_unchanged(self):
        config = load_scoring_config()
        first = score_candidate_rows([row()], config)
        second = score_candidate_rows([row()], config)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

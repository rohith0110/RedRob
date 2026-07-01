import tempfile
import unittest
from pathlib import Path

import yaml

from src.phase2_config import load_phase2_config


class Phase2ConfigTest(unittest.TestCase):
    def copy_configs(self, td: str) -> dict[str, Path]:
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
            path = root / name
            path.write_text(Path("configs") .joinpath(name).read_text(encoding="utf-8"), encoding="utf-8")
            paths[name] = path
        return paths

    def test_evidence_config_change_only_moves_evidence_bundle_hash(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self.copy_configs(td)
            base = load_phase2_config(
                paths["role_rubric.yaml"],
                paths["evidence_patterns.yaml"],
                paths["scoring_weights.yaml"],
                paths["credibility_rules.yaml"],
                paths["behavioral_rules.yaml"],
                paths["semantic_config.yaml"],
            )
            evidence_data = yaml.safe_load(paths["evidence_patterns.yaml"].read_text(encoding="utf-8"))
            evidence_data["categories"]["retrieval_ranking_relevance"]["phrase_patterns"].append("fresh phrase")
            paths["evidence_patterns.yaml"].write_text(yaml.safe_dump(evidence_data), encoding="utf-8")
            changed = load_phase2_config(
                paths["role_rubric.yaml"],
                paths["evidence_patterns.yaml"],
                paths["scoring_weights.yaml"],
                paths["credibility_rules.yaml"],
                paths["behavioral_rules.yaml"],
                paths["semantic_config.yaml"],
            )
            self.assertNotEqual(base.bundle_hashes["evidence"], changed.bundle_hashes["evidence"])
            self.assertEqual(base.bundle_hashes["scoring"], changed.bundle_hashes["scoring"])
            self.assertEqual(base.bundle_hashes["credibility"], changed.bundle_hashes["credibility"])
            self.assertEqual(base.bundle_hashes["behavioral"], changed.bundle_hashes["behavioral"])

    def test_semantic_enablement_changes_semantic_bundle_hash(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self.copy_configs(td)
            base = load_phase2_config(
                paths["role_rubric.yaml"],
                paths["evidence_patterns.yaml"],
                paths["scoring_weights.yaml"],
                paths["credibility_rules.yaml"],
                paths["behavioral_rules.yaml"],
                paths["semantic_config.yaml"],
            )
            semantic_data = yaml.safe_load(paths["semantic_config.yaml"].read_text(encoding="utf-8"))
            semantic_data["enabled"] = not semantic_data["enabled"]
            paths["semantic_config.yaml"].write_text(yaml.safe_dump(semantic_data, sort_keys=False), encoding="utf-8")
            changed = load_phase2_config(
                paths["role_rubric.yaml"],
                paths["evidence_patterns.yaml"],
                paths["scoring_weights.yaml"],
                paths["credibility_rules.yaml"],
                paths["behavioral_rules.yaml"],
                paths["semantic_config.yaml"],
            )
            self.assertNotEqual(base.bundle_hashes["semantic"], changed.bundle_hashes["semantic"])

    def test_credibility_rules_require_structured_rule_declarations(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self.copy_configs(td)
            credibility_data = yaml.safe_load(paths["credibility_rules.yaml"].read_text(encoding="utf-8"))
            credibility_data["rules"]["title_description_contradiction"].pop("supported_title_families", None)
            paths["credibility_rules.yaml"].write_text(yaml.safe_dump(credibility_data, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supported_title_families"):
                load_phase2_config(
                    paths["role_rubric.yaml"],
                    paths["evidence_patterns.yaml"],
                    paths["scoring_weights.yaml"],
                    paths["credibility_rules.yaml"],
                    paths["behavioral_rules.yaml"],
                    paths["semantic_config.yaml"],
                )


if __name__ == "__main__":
    unittest.main()

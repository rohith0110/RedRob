import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.behavioral import run_behavioral_analysis
from src.credibility import run_credibility_analysis
from src.evidence import run_evidence_extraction
from src.normalization import run_normalization
from src.paths import ensure_run_dirs
from src.phase2_config import load_phase2_config
from src.v2_scoring import run_v2_scoring


def candidate() -> dict:
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Software Engineer",
            "summary": "Built recommendations and improved search results.",
            "current_title": "Software Engineer",
            "current_company": "Acme",
            "current_industry": "Software",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 6,
        },
        "career_history": [
            {
                "title": "Software Engineer",
                "company": "Acme",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 48,
                "is_current": True,
                "description": "Improved product discovery, tuned relevance, and shipped ranking systems for live users.",
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
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24,
            "notice_period_days": 30,
            "preferred_work_mode": "remote",
            "willing_to_relocate": False,
            "github_activity_score": 15,
            "interview_completion_rate": 0.8,
            "skill_assessment_scores": {"Python": 88},
        },
    }


class SemanticFallbackTest(unittest.TestCase):
    def prepare_run(self, td: str) -> Path:
        td_path = Path(td)
        dataset = td_path / "candidates.jsonl"
        dataset.write_text(json.dumps(candidate()) + "\n", encoding="utf-8")
        run_path = ensure_run_dirs(td_path / "run")
        run_normalization(dataset, run_path)
        run_evidence_extraction(run_path)
        run_credibility_analysis(run_path)
        run_behavioral_analysis(run_path)
        return run_path

    def _copy_phase2_config(self, td: str, *, semantic_enabled: bool) -> object:
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
        semantic_data = yaml.safe_load(paths["semantic_config.yaml"].read_text(encoding="utf-8"))
        semantic_data["enabled"] = semantic_enabled
        paths["semantic_config.yaml"].write_text(yaml.safe_dump(semantic_data, sort_keys=False), encoding="utf-8")
        return load_phase2_config(
            paths["role_rubric.yaml"],
            paths["evidence_patterns.yaml"],
            paths["scoring_weights.yaml"],
            paths["credibility_rules.yaml"],
            paths["behavioral_rules.yaml"],
            paths["semantic_config.yaml"],
        )

    def test_absent_semantic_artifact_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = self.prepare_run(td)
            config = self._copy_phase2_config(td, semantic_enabled=True)
            result = run_v2_scoring(run_path, config=config)
            breakdown = pd.read_parquet(result["score_breakdown_v2_parquet"])
            self.assertEqual(len(breakdown), 1)
            status = json.loads((run_path / "scores" / "semantic_fallback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "absent")

    def _write_valid_artifact(self, run_path: Path, dataset_hash: str, score: float) -> None:
        from src.hashing import sha256_file
        from src.semantic_artifact import candidate_checksum

        norm = pd.read_parquet(run_path / "normalized" / "candidates_normalized.parquet")
        cids = list(norm["candidate_id"])
        semantic_dir = run_path / "artifacts" / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        artifact = semantic_dir / "semantic_scores.parquet"
        pd.DataFrame([{"candidate_id": c, "semantic_score": score} for c in cids]).to_parquet(artifact, index=False)
        (semantic_dir / "semantic_manifest.json").write_text(json.dumps({
            "source_dataset_hash": dataset_hash,
            "candidate_id_checksum": candidate_checksum(cids),
            "resolved_config_hash": self.phase2_config.bundle_hashes["scoring"],
            "output_artifact_hash": sha256_file(artifact),
            "candidate_count": len(cids),
        }), encoding="utf-8")

    def test_valid_semantic_artifact_loads_and_is_capped(self):
        # Mirrors the prepare-script workflow: the manifest is keyed on the raw
        # input hash recorded in the run manifest, and the contribution is bounded
        # by the configured semantic cap (0.08) even at the maximum score.
        from src.hashing import sha256_file

        with tempfile.TemporaryDirectory() as td:
            run_path = self.prepare_run(td)
            self.phase2_config = self._copy_phase2_config(td, semantic_enabled=True)
            raw_hash = sha256_file(Path(td) / "candidates.jsonl")
            (run_path / "manifest.json").write_text(json.dumps({"input_file_sha256": raw_hash}), encoding="utf-8")

            baseline = pd.read_parquet(run_v2_scoring(run_path, config=self.phase2_config)["score_breakdown_v2_parquet"]).set_index("candidate_id")

            self._write_valid_artifact(run_path, raw_hash, score=1.0)
            result = run_v2_scoring(run_path, config=self.phase2_config)
            breakdown = pd.read_parquet(result["score_breakdown_v2_parquet"]).set_index("candidate_id")
            status = json.loads((run_path / "scores" / "semantic_fallback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "loaded")
            self.assertEqual(breakdown.loc["CAND_0000001", "semantic_score"], 1.0)
            # Maximum semantic score lifts base_fit by no more than the cap.
            delta = breakdown.loc["CAND_0000001", "base_fit_score"] - baseline.loc["CAND_0000001", "base_fit_score"]
            self.assertLessEqual(round(delta, 6), 0.08 + 1e-9)
            self.assertGreater(delta, 0.0)

    def test_mismatched_semantic_artifact_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = self.prepare_run(td)
            config = self._copy_phase2_config(td, semantic_enabled=True)
            semantic_dir = run_path / "artifacts" / "semantic"
            semantic_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"candidate_id": "CAND_0000001", "semantic_score": 0.9}]).to_parquet(semantic_dir / "semantic_scores.parquet", index=False)
            (semantic_dir / "semantic_manifest.json").write_text(json.dumps({"candidate_count": 999, "output_artifact_hash": "bad"}), encoding="utf-8")
            result = run_v2_scoring(run_path, config=config)
            breakdown = pd.read_parquet(result["score_breakdown_v2_parquet"])
            self.assertEqual(len(breakdown), 1)
            status = json.loads((run_path / "scores" / "semantic_fallback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "mismatch")

    def test_disabled_semantic_config_does_not_attempt_artifact_loading(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = self.prepare_run(td)
            config = self._copy_phase2_config(td, semantic_enabled=False)
            result = run_v2_scoring(run_path, config=config)
            breakdown = pd.read_parquet(result["score_breakdown_v2_parquet"])
            self.assertEqual(float(breakdown.loc[0, "semantic_score"]), 0.0)
            status = json.loads((run_path / "scores" / "semantic_fallback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "disabled")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.checkpointing import begin_stage_attempt, record_stage_skip, stage_is_valid, update_stage
from src.cli import artifact_fingerprint, score_stage_payload, stage_fingerprint
from src.manifest import base_manifest


class ResumeBehaviorTest(unittest.TestCase):
    def test_completed_stage_reruns_when_output_hash_mismatches(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            state = Path(td) / "state.json"
            out.write_text("ok", encoding="utf-8")
            begin_stage_attempt(state, "audit", "fingerprint", "config")
            update_stage(state, "audit", "completed", "fingerprint", "config", [out], 1)
            self.assertTrue(stage_is_valid(state, "audit", "fingerprint", "config"))

            out.write_text("changed", encoding="utf-8")
            self.assertFalse(stage_is_valid(state, "audit", "fingerprint", "config"))
            self.assertEqual(json.loads(state.read_text())["stages"]["audit"]["status"], "completed")

    def test_stage_attempt_fields_track_first_run_and_resume_skip(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            state = Path(td) / "state.json"
            out.write_text("ok", encoding="utf-8")

            begin_stage_attempt(state, "evidence", "fingerprint", "config")
            update_stage(state, "evidence", "completed", "fingerprint", "config", [out], 1)
            first = json.loads(state.read_text(encoding="utf-8"))["stages"]["evidence"]
            self.assertEqual(first["total_attempt_count"], 1)
            self.assertEqual(first["latest_attempt_status"], "completed")
            self.assertEqual(first["status"], "completed")

            record_stage_skip(state, "evidence", "fingerprint", "config", "resume validated existing artifacts")
            second = json.loads(state.read_text(encoding="utf-8"))["stages"]["evidence"]
            self.assertEqual(second["first_started_at"], first["first_started_at"])
            self.assertEqual(second["total_attempt_count"], 2)
            self.assertEqual(second["latest_attempt_status"], "skipped")
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["latest_skip_reason"], "resume validated existing artifacts")

    def test_forced_rerun_updates_latest_attempt_fields_without_losing_first_start(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            state = Path(td) / "state.json"
            out.write_text("first", encoding="utf-8")

            begin_stage_attempt(state, "scores", "fingerprint", "config")
            update_stage(state, "scores", "completed", "fingerprint", "config", [out], 1)
            first = json.loads(state.read_text(encoding="utf-8"))["stages"]["scores"]

            out.write_text("second", encoding="utf-8")
            begin_stage_attempt(state, "scores", "fingerprint", "config")
            update_stage(state, "scores", "completed", "fingerprint", "config", [out], 1)
            second = json.loads(state.read_text(encoding="utf-8"))["stages"]["scores"]

            self.assertEqual(second["first_started_at"], first["first_started_at"])
            self.assertEqual(second["total_attempt_count"], 2)
            self.assertEqual(second["latest_attempt_status"], "completed")
            self.assertNotEqual(second["output_file_sha256"], first["output_file_sha256"])
            self.assertGreaterEqual(second["latest_elapsed_seconds"], 0.0)

    def test_corrupted_artifact_rerun_is_distinguishable_in_attempt_history(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            state = Path(td) / "state.json"
            out.write_text("ok", encoding="utf-8")

            begin_stage_attempt(state, "behavioral", "fingerprint", "config")
            update_stage(state, "behavioral", "completed", "fingerprint", "config", [out], 1)
            out.write_text("corrupted", encoding="utf-8")
            self.assertFalse(stage_is_valid(state, "behavioral", "fingerprint", "config"))

            begin_stage_attempt(state, "behavioral", "fingerprint", "config")
            update_stage(state, "behavioral", "completed", "fingerprint", "config", [out], 1)
            stage = json.loads(state.read_text(encoding="utf-8"))["stages"]["behavioral"]
            self.assertEqual(stage["total_attempt_count"], 2)
            self.assertEqual(stage["latest_attempt_status"], "completed")
            self.assertEqual(len(stage["attempt_history"]), 2)

    def test_downstream_fingerprint_changes_when_upstream_artifact_changes(self):
        with tempfile.TemporaryDirectory() as td:
            normalized = Path(td) / "candidates_normalized.parquet"
            normalized.write_text("one", encoding="utf-8")
            first = stage_fingerprint("scores", {"upstream_normalized": artifact_fingerprint([normalized], 1), "seed": 42})

            normalized.write_text("two", encoding="utf-8")
            second = stage_fingerprint("scores", {"upstream_normalized": artifact_fingerprint([normalized], 1), "seed": 42})
            self.assertNotEqual(first, second)

    def test_submission_fingerprint_changes_for_output_destination_only(self):
        with tempfile.TemporaryDirectory() as td:
            score = Path(td) / "baseline_score_breakdown.csv"
            score.write_text("candidate_id,final_score\nCAND_0000001,1\n", encoding="utf-8")
            upstream = artifact_fingerprint([score], 1)
            scores_fp = stage_fingerprint("scores", {"upstream_normalized": {"row_count": 1, "files": {"x": "y"}}, "seed": 42})
            sub_a = stage_fingerprint("submission", {"upstream_scoring": upstream, "official_output": str(Path(td) / "a.csv")})
            sub_b = stage_fingerprint("submission", {"upstream_scoring": upstream, "official_output": str(Path(td) / "b.csv")})
            self.assertEqual(scores_fp, stage_fingerprint("scores", {"upstream_normalized": {"row_count": 1, "files": {"x": "y"}}, "seed": 42}))
            self.assertNotEqual(sub_a, sub_b)

    def test_score_stage_payload_contains_resolved_config_hash_once(self):
        payload = score_stage_payload(
            upstream_normalized={"row_count": 10, "files": {"normalized.parquet": "file-hash"}},
            resolved_config_hash="resolved-config-hash",
            seed=42,
            source_hashes={"src/baseline_scoring.py": "code-hash"},
        )
        serialized = json.dumps(payload, sort_keys=True)
        self.assertEqual(serialized.count("resolved-config-hash"), 1)
        self.assertIn("resolved_config_hash", payload)

    def test_base_manifest_records_clean_head_fields(self):
        with tempfile.TemporaryDirectory() as td, patch("src.manifest.git_commit", return_value="abc123"), patch(
            "src.manifest.git_working_tree_dirty", return_value=False
        ):
            candidates = Path(td) / "candidates.jsonl"
            candidates.write_text('{"candidate_id":"CAND_1"}\n', encoding="utf-8")
            manifest = base_manifest("run_id", candidates, {"resume": False}, [])
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertIn("working_tree_dirty", manifest)
            self.assertFalse(manifest["working_tree_dirty"])


if __name__ == "__main__":
    unittest.main()

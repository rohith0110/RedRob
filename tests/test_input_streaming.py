import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.io import collect_valid_candidates


class InputStreamingTest(unittest.TestCase):
    def test_reads_jsonl_and_gz_and_reports_malformed(self):
        with tempfile.TemporaryDirectory() as td:
            plain = Path(td) / "candidates.jsonl"
            gz = Path(td) / "candidates.jsonl.gz"
            good = {
                "candidate_id": "CAND_0000001",
                "profile": {},
                "career_history": [],
                "education": [],
                "skills": [],
                "redrob_signals": {},
            }
            text = json.dumps(good) + "\n{bad json\n"
            plain.write_text(text, encoding="utf-8")
            with gzip.open(gz, "wt", encoding="utf-8") as f:
                f.write(text)

            for path in (plain, gz):
                candidates, stats, issues = collect_valid_candidates(path)
                self.assertEqual([c["candidate_id"] for c in candidates], ["CAND_0000001"])
                self.assertEqual(stats["malformed_records"], 1)
                self.assertEqual(issues[0]["type"], "malformed_json")

    def test_duplicate_ids_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidates.jsonl"
            row = {
                "candidate_id": "CAND_0000001",
                "profile": {},
                "career_history": [],
                "education": [],
                "skills": [],
                "redrob_signals": {},
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            _, stats, issues = collect_valid_candidates(path)
            self.assertEqual(stats["duplicate_candidate_ids"], 1)
            self.assertEqual(issues[-1]["type"], "duplicate_candidate_id")


if __name__ == "__main__":
    unittest.main()

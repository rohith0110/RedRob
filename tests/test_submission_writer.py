import csv
import tempfile
import unittest
from pathlib import Path

from src.baseline_reasoning import reasoning_for
from src.submission_validation import validate_submission
from src.submission_writer import write_submission


class SubmissionWriterTest(unittest.TestCase):
    def test_generated_production_csv_passes_validator(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "baseline_submission.csv"
            rows = [
                {
                    "candidate_id": f"CAND_{i:07d}",
                    "final_score": float(101 - i),
                    "top_positive_evidence": "ranking evidence",
                    "top_negative_evidence": "",
                }
                for i in range(1, 101)
            ]
            write_submission(rows, out)
            self.assertEqual(validate_submission(out), [])
            with out.open(encoding="utf-8", newline="") as f:
                self.assertEqual(next(csv.reader(f)), ["candidate_id", "rank", "score", "reasoning"])

    def test_reasoning_handles_nan_negative_and_preserves_caps(self):
        # pandas read_csv yields float NaN for empty cells; it must not surface as "Weakness: nan".
        text = reasoning_for({"top_positive_evidence": "Python ranking evidence", "top_negative_evidence": float("nan")})
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("Weakness", text)
        self.assertIn("Python", text)  # capitalize() would have lowercased this
        with_weak = reasoning_for({"top_positive_evidence": "ranking", "top_negative_evidence": "severe anomalies"})
        self.assertIn("Weakness: severe anomalies.", with_weak)


if __name__ == "__main__":
    unittest.main()

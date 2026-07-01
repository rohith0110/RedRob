import csv
import tempfile
import unittest
from pathlib import Path

from src.submission_validation import validate_submission

ORGANIZER = Path("data/validate_submission.py")
ROOT = Path("validate_submission.py")


def _write(rows):
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    with tmp.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return tmp


def _valid_rows():
    rows = [["candidate_id", "rank", "score", "reasoning"]]
    for i in range(1, 101):
        rows.append([f"CAND_{i:07d}", str(i), f"{100 - i}.000000", "ok"])
    return rows


class ValidatorParityTest(unittest.TestCase):
    def test_root_validator_is_organizer_verbatim(self):
        # The executed gate must be the unchanged organizer validator, not a rewrite.
        self.assertEqual(
            ROOT.read_bytes(), ORGANIZER.read_bytes(),
            "root validate_submission.py drifted from the organizer copy in data/",
        )

    def test_canonical_submission_is_valid(self):
        tmp = _write(_valid_rows())
        self.addCleanup(tmp.unlink)
        self.assertEqual(validate_submission(tmp), [])

    def test_rejects_non_canonical_rank(self):
        # Organizer rejects zero-padded ranks; a permissive rewrite did not.
        rows = _valid_rows()
        rows[1][1] = "01"
        tmp = _write(rows)
        self.addCleanup(tmp.unlink)
        self.assertTrue(validate_submission(tmp))


if __name__ == "__main__":
    unittest.main()

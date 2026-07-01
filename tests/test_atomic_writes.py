import tempfile
import unittest
from pathlib import Path

from src.atomic_writes import atomic_publish
from src.checkpointing import load_state, update_stage


class AtomicWritesTest(unittest.TestCase):
    def test_failed_publish_preserves_previous_output_and_marks_stage_failed(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "artifact.txt"
            state = Path(td) / "state.json"
            target.write_text("previous", encoding="utf-8")

            def broken_writer(tmp):
                tmp.write_text("partial", encoding="utf-8")
                raise RuntimeError("boom")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                atomic_publish(target, broken_writer)
            update_stage(state, "scores", "failed", "fingerprint", "config", [], 0, metadata={"error": "boom"})

            self.assertEqual(target.read_text(encoding="utf-8"), "previous")
            self.assertEqual(load_state(state)["stages"]["scores"]["status"], "failed")
            self.assertEqual(list(Path(td).glob(f".{target.name}.*")), [])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import yaml

from src.runtime_config import load_runtime_config


class RuntimeConfigTest(unittest.TestCase):
    def test_processing_mode_defaults_to_memory_and_resolves_reference_date(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "version": "phase2",
                        "processing_mode": "memory",
                        "chunk_size": 5000,
                        "dataset_reference_date": "2026-06-30",
                    }
                ),
                encoding="utf-8",
            )
            config = load_runtime_config(path)
            self.assertEqual(config.processing_mode, "memory")
            self.assertEqual(config.chunk_size, 5000)
            self.assertEqual(config.dataset_reference_date, "2026-06-30")

    def test_invalid_processing_mode_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "version": "phase2",
                        "processing_mode": "streaming",
                        "chunk_size": 5000,
                        "dataset_reference_date": "2026-06-30",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "processing_mode"):
                load_runtime_config(path)

    def test_invalid_reference_date_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "version": "phase2",
                        "processing_mode": "memory",
                        "chunk_size": 5000,
                        "dataset_reference_date": "2026-02-31",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dataset_reference_date"):
                load_runtime_config(path)


if __name__ == "__main__":
    unittest.main()

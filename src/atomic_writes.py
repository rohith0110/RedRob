import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


def atomic_publish(path: str | Path, write_temp: Callable[[Path], None], validate_temp: Callable[[Path], None] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix or ".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_temp(tmp)
        if validate_temp:
            validate_temp(tmp)
        try:
            with tmp.open("rb") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        os.replace(tmp, path)
        return path
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def write_text_atomic(path: str | Path, text: str) -> Path:
    def writer(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

    return atomic_publish(path, writer, lambda tmp: tmp.read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, data: object) -> Path:
    text = json.dumps(data, indent=2, default=str)
    return write_text_atomic(path, text)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> Path:
    def writer(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def validator(tmp: Path) -> None:
        with tmp.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)

    return atomic_publish(path, writer, validator)


def validate_csv(path: str | Path, expected_rows: int | None = None, required_header: list[str] | None = None) -> None:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if required_header and header != required_header:
            raise ValueError(f"CSV header mismatch: {header}")
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"CSV row count mismatch: expected {expected_rows}, found {len(rows)}")


def write_csv_atomic(path: str | Path, write_file: Callable, expected_rows: int | None = None, required_header: list[str] | None = None, validator: Callable[[Path], None] | None = None) -> Path:
    def writer(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            write_file(f)
            f.flush()
            os.fsync(f.fileno())

    def validate(tmp: Path) -> None:
        validate_csv(tmp, expected_rows=expected_rows, required_header=required_header)
        if validator:
            validator(tmp)

    return atomic_publish(path, writer, validate)


def write_parquet_atomic(df: pd.DataFrame, path: str | Path, expected_rows: int | None = None, required_columns: list[str] | None = None) -> Path:
    def writer(tmp: Path) -> None:
        df.to_parquet(tmp, index=False)

    def validator(tmp: Path) -> None:
        read_back = pd.read_parquet(tmp)
        if expected_rows is not None and len(read_back) != expected_rows:
            raise ValueError(f"Parquet row count mismatch: expected {expected_rows}, found {len(read_back)}")
        missing = set(required_columns or []) - set(read_back.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {sorted(missing)}")

    return atomic_publish(path, writer, validator)


def copy_atomic(src: str | Path, dst: str | Path, validator: Callable[[Path], None] | None = None) -> Path:
    src = Path(src)

    def writer(tmp: Path) -> None:
        shutil.copyfile(src, tmp)

    return atomic_publish(dst, writer, validator)

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from .hashing import sha256_file, sha256_json

RUNTIME_CONFIG_PATH = Path("configs/runtime.yaml")
PROCESSING_MODES = {"memory", "chunked"}


@dataclass(frozen=True)
class RuntimeConfig:
    path: str
    raw_sha256: str
    resolved_sha256: str
    version: str
    processing_mode: str
    chunk_size: int
    dataset_reference_date: str


def load_runtime_config(path: str | Path = RUNTIME_CONFIG_PATH) -> RuntimeConfig:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing runtime config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    unknown = set(data) - {"version", "processing_mode", "chunk_size", "dataset_reference_date"}
    if unknown:
        raise ValueError(f"runtime config has unknown keys: {sorted(unknown)}")
    processing_mode = data.get("processing_mode", "memory")
    if processing_mode not in PROCESSING_MODES:
        raise ValueError("processing_mode must be one of: chunked, memory")
    chunk_size = data.get("chunk_size", 5000)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    dataset_reference_date = data.get("dataset_reference_date")
    if not dataset_reference_date:
        raise ValueError("dataset_reference_date is required for reproducible ranking")
    try:
        dataset_reference_date = str(date.fromisoformat(str(dataset_reference_date)[:10]))
    except ValueError as exc:
        raise ValueError("dataset_reference_date must be an ISO date (YYYY-MM-DD)") from exc
    resolved = MappingProxyType({
        "version": str(data.get("version", "phase2-preflight")),
        "processing_mode": processing_mode,
        "chunk_size": chunk_size,
        "dataset_reference_date": dataset_reference_date,
    })
    return RuntimeConfig(
        path=str(path),
        raw_sha256=sha256_file(path),
        resolved_sha256=sha256_json(dict(resolved)),
        version=resolved["version"],
        processing_mode=processing_mode,
        chunk_size=chunk_size,
        dataset_reference_date=dataset_reference_date,
    )


def runtime_config_manifest(config: RuntimeConfig) -> dict:
    return {
        "path": config.path,
        "raw_sha256": config.raw_sha256,
        "resolved_sha256": config.resolved_sha256,
        "version": config.version,
        "processing_mode": config.processing_mode,
        "chunk_size": config.chunk_size,
        "dataset_reference_date": config.dataset_reference_date,
    }

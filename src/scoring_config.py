from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .hashing import sha256_file, sha256_json

WEIGHTS_PATH = Path("configs/baseline_weights.yaml")
RUBRIC_PATH = Path("configs/role_rubric.yaml")
WEIGHT_KEYS = {
    "career_text_relevance",
    "production_evaluation",
    "python_engineering",
    "experience_plausibility",
    "location_logistics",
    "behavioral_availability",
}
TOLERANCE = 1e-6


@dataclass(frozen=True)
class TextCategory:
    positive_phrases: tuple[str, ...]
    aliases: tuple[str, ...]
    negative_phrases: tuple[str, ...]
    field_weights: Mapping[str, float]
    hit_divisors: Mapping[str, int]
    positive_evidence: str
    weak_evidence: str = ""


@dataclass(frozen=True)
class ScoringConfig:
    weights_path: str
    rubric_path: str
    weights_raw_sha256: str
    rubric_raw_sha256: str
    resolved_sha256: str
    scoring_code_version: str
    formula_metadata: Mapping[str, Any]
    positive_weights: Mapping[str, float]
    anomaly_penalty: Mapping[str, float]
    low_relevance_threshold: float
    low_relevance_max_score: float
    role_relevance: TextCategory
    production_evaluation: TextCategory
    python_engineering: TextCategory
    experience_bands: tuple[Mapping[str, float], ...]
    location: Mapping[str, Any]
    availability: Mapping[str, float]


def _load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing scoring config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _reject_unknown(name: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def _number(name: str, value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _strings(name: str, values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return tuple(values)


def _category(name: str, data: Any) -> TextCategory:
    if not isinstance(data, dict):
        raise ValueError(f"role_rubric concept_categories.{name} must be a mapping")
    _reject_unknown(
        f"role_rubric concept_categories.{name}",
        data,
        {"positive_phrases", "aliases", "negative_phrases", "field_weights", "hit_divisor", "hit_divisors", "positive_evidence", "weak_evidence"},
    )
    weights = data.get("field_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{name}.field_weights must be a mapping")
    resolved_weights = {str(k): _number(f"{name}.field_weights.{k}", v) for k, v in weights.items()}
    total = sum(resolved_weights.values())
    if abs(total - 1.0) > TOLERANCE:
        raise ValueError(f"{name}.field_weights must sum to 1.0")
    if "hit_divisors" in data:
        raw_divisors = data["hit_divisors"]
        if not isinstance(raw_divisors, dict):
            raise ValueError(f"{name}.hit_divisors must be a mapping")
        missing = set(resolved_weights) - set(raw_divisors)
        if missing:
            raise ValueError(f"{name}.hit_divisors missing fields: {sorted(missing)}")
        divisors = {}
        for field, value in raw_divisors.items():
            if field not in resolved_weights:
                raise ValueError(f"{name}.hit_divisors has unknown field: {field}")
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}.hit_divisors.{field} must be a positive integer")
            divisors[field] = value
    else:
        hit_divisor = data.get("hit_divisor")
        if not isinstance(hit_divisor, int) or hit_divisor <= 0:
            raise ValueError(f"{name}.hit_divisor must be a positive integer")
        divisors = {field: hit_divisor for field in resolved_weights}
    evidence = data.get("positive_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"{name}.positive_evidence must be a non-empty string")
    weak = data.get("weak_evidence", "")
    if not isinstance(weak, str):
        raise ValueError(f"{name}.weak_evidence must be a string")
    aliases = data.get("aliases") or []
    negative = data.get("negative_phrases") or []
    if not isinstance(aliases, list) or not all(isinstance(v, str) for v in aliases):
        raise ValueError(f"{name}.aliases must be a list of strings")
    if not isinstance(negative, list) or not all(isinstance(v, str) for v in negative):
        raise ValueError(f"{name}.negative_phrases must be a list of strings")
    return TextCategory(
        positive_phrases=_strings(f"{name}.positive_phrases", data.get("positive_phrases")),
        aliases=tuple(aliases),
        negative_phrases=tuple(negative),
        field_weights=MappingProxyType(resolved_weights),
        hit_divisors=MappingProxyType(divisors),
        positive_evidence=evidence,
        weak_evidence=weak,
    )


def _resolved_hash(config: dict) -> str:
    return sha256_json(config)


def load_scoring_config(weights_path: str | Path = WEIGHTS_PATH, rubric_path: str | Path = RUBRIC_PATH) -> ScoringConfig:
    weights_path = Path(weights_path)
    rubric_path = Path(rubric_path)
    weights_raw = _load_yaml(weights_path)
    rubric_raw = _load_yaml(rubric_path)
    _reject_unknown("baseline_weights", weights_raw, {"version", "positive_score_weights", "anomaly_penalty", "low_relevance_cap"})
    positive = weights_raw.get("positive_score_weights")
    if not isinstance(positive, dict):
        raise ValueError("baseline_weights.positive_score_weights must be a mapping")
    missing = WEIGHT_KEYS - set(positive)
    if missing:
        raise ValueError(f"baseline_weights.positive_score_weights missing keys: {sorted(missing)}")
    _reject_unknown("baseline_weights.positive_score_weights", positive, WEIGHT_KEYS)
    positive_weights = {key: _number(f"positive_score_weights.{key}", positive[key]) for key in sorted(WEIGHT_KEYS)}
    if abs(sum(positive_weights.values()) - 1.0) > TOLERANCE:
        raise ValueError("baseline positive score weights must sum to 1.0")

    penalties = weights_raw.get("anomaly_penalty")
    if not isinstance(penalties, dict):
        raise ValueError("baseline_weights.anomaly_penalty must be a mapping")
    _reject_unknown("baseline_weights.anomaly_penalty", penalties, {"none", "minor", "severe", "default"})
    anomaly_penalty = {key: _number(f"anomaly_penalty.{key}", penalties[key], 0.0, 1.0) for key in ("none", "minor", "severe", "default")}

    cap = weights_raw.get("low_relevance_cap")
    if not isinstance(cap, dict):
        raise ValueError("baseline_weights.low_relevance_cap must be a mapping")
    _reject_unknown("baseline_weights.low_relevance_cap", cap, {"threshold", "max_score"})
    low_threshold = _number("low_relevance_cap.threshold", cap["threshold"], 0.0, 1.0)
    low_max = _number("low_relevance_cap.max_score", cap["max_score"], 0.0, 100.0)

    _reject_unknown("role_rubric", rubric_raw, {"version", "concept_categories", "experience_bands", "location_logistics", "availability"})
    categories = rubric_raw.get("concept_categories")
    if not isinstance(categories, dict):
        raise ValueError("role_rubric.concept_categories must be a mapping")
    _reject_unknown("role_rubric.concept_categories", categories, {"role_relevance", "production_evaluation", "python_engineering"})
    role_relevance = _category("role_relevance", categories.get("role_relevance"))
    production = _category("production_evaluation", categories.get("production_evaluation"))
    python = _category("python_engineering", categories.get("python_engineering"))

    bands = rubric_raw.get("experience_bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("role_rubric.experience_bands must be a non-empty list")
    resolved_bands = []
    for idx, band in enumerate(bands):
        if not isinstance(band, dict):
            raise ValueError(f"experience_bands[{idx}] must be a mapping")
        _reject_unknown(f"experience_bands[{idx}]", band, {"min_years", "max_years", "score"})
        resolved_bands.append(MappingProxyType({
            "min_years": _number(f"experience_bands[{idx}].min_years", band["min_years"], 0.0, 100.0),
            "max_years": _number(f"experience_bands[{idx}].max_years", band["max_years"], 0.0, 100.0),
            "score": _number(f"experience_bands[{idx}].score", band["score"]),
        }))

    location = rubric_raw.get("location_logistics")
    if not isinstance(location, dict):
        raise ValueError("role_rubric.location_logistics must be a mapping")
    _reject_unknown("role_rubric.location_logistics", location, {"preferred_modes", "preferred_country", "preferred_score", "fallback_score"})
    resolved_location = MappingProxyType({
        "preferred_modes": _strings("location_logistics.preferred_modes", location["preferred_modes"]),
        "preferred_country": str(location["preferred_country"]).lower(),
        "preferred_score": _number("location_logistics.preferred_score", location["preferred_score"]),
        "fallback_score": _number("location_logistics.fallback_score", location["fallback_score"]),
    })

    availability = rubric_raw.get("availability")
    if not isinstance(availability, dict):
        raise ValueError("role_rubric.availability must be a mapping")
    _reject_unknown("role_rubric.availability", availability, {"open_to_work", "response_rate", "response_time", "notice_period", "max_response_time_hours", "max_notice_days"})
    resolved_availability = MappingProxyType({
        "open_to_work": _number("availability.open_to_work", availability["open_to_work"]),
        "response_rate": _number("availability.response_rate", availability["response_rate"]),
        "response_time": _number("availability.response_time", availability["response_time"]),
        "notice_period": _number("availability.notice_period", availability["notice_period"]),
        "max_response_time_hours": _number("availability.max_response_time_hours", availability["max_response_time_hours"], 1.0, 10000.0),
        "max_notice_days": _number("availability.max_notice_days", availability["max_notice_days"], 1.0, 10000.0),
    })
    if abs(sum(resolved_availability[k] for k in ("open_to_work", "response_rate", "response_time", "notice_period")) - 1.0) > TOLERANCE:
        raise ValueError("availability evidence weights must sum to 1.0")

    resolved = {
        "weights": weights_raw,
        "rubric": rubric_raw,
        "scoring_code_version": "phase1.5-config-driven-v1",
        "score_formula": "max(0, (weighted_components - anomaly_penalty) * 100), capped when role relevance is below threshold",
    }
    resolved_sha = _resolved_hash(resolved)
    return ScoringConfig(
        weights_path=str(weights_path),
        rubric_path=str(rubric_path),
        weights_raw_sha256=sha256_file(weights_path),
        rubric_raw_sha256=sha256_file(rubric_path),
        resolved_sha256=resolved_sha,
        scoring_code_version=resolved["scoring_code_version"],
        formula_metadata=MappingProxyType({"version": resolved["scoring_code_version"], "description": resolved["score_formula"]}),
        positive_weights=MappingProxyType(positive_weights),
        anomaly_penalty=MappingProxyType(anomaly_penalty),
        low_relevance_threshold=low_threshold,
        low_relevance_max_score=low_max,
        role_relevance=role_relevance,
        production_evaluation=production,
        python_engineering=python,
        experience_bands=tuple(resolved_bands),
        location=resolved_location,
        availability=resolved_availability,
    )


def config_manifest(config: ScoringConfig) -> dict:
    return {
        "paths": {"baseline_weights": config.weights_path, "role_rubric": config.rubric_path},
        "raw_sha256": {"baseline_weights": config.weights_raw_sha256, "role_rubric": config.rubric_raw_sha256},
        "resolved_sha256": config.resolved_sha256,
        "scoring_code_version": config.scoring_code_version,
        "formula_metadata": dict(config.formula_metadata),
    }

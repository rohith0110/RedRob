from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .hashing import sha256_file, sha256_json

ROLE_RUBRIC_PATH = Path("configs/role_rubric.yaml")
EVIDENCE_PATTERNS_PATH = Path("configs/evidence_patterns.yaml")
SCORING_WEIGHTS_PATH = Path("configs/scoring_weights.yaml")
CREDIBILITY_RULES_PATH = Path("configs/credibility_rules.yaml")
BEHAVIORAL_RULES_PATH = Path("configs/behavioral_rules.yaml")
SEMANTIC_CONFIG_PATH = Path("configs/semantic_config.yaml")


@dataclass(frozen=True)
class Phase2ConfigBundle:
    paths: Mapping[str, str]
    raw_sha256: Mapping[str, str]
    bundle_hashes: Mapping[str, str]
    role_rubric: Mapping[str, Any]
    evidence_patterns: Mapping[str, Any]
    scoring_weights: Mapping[str, Any]
    credibility_rules: Mapping[str, Any]
    behavioral_rules: Mapping[str, Any]
    semantic_config: Mapping[str, Any]


def _load_yaml_map(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing phase2 config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _validate_evidence_patterns(data: Mapping[str, Any]) -> None:
    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("evidence_patterns.categories must be a non-empty mapping")
    required = {
        "aliases",
        "phrase_patterns",
        "polarity",
        "source_field_priority",
        "category_weight",
        "confidence_rules",
        "max_contribution_cap",
        "recency_factor",
        "require_corroboration",
    }
    for name, category in categories.items():
        if not isinstance(category, dict):
            raise ValueError(f"evidence_patterns.categories.{name} must be a mapping")
        missing = required - set(category)
        if missing:
            raise ValueError(f"evidence_patterns.categories.{name} missing keys: {sorted(missing)}")


def _validate_semantic_config(data: Mapping[str, Any]) -> None:
    required = {"version", "enabled", "artifact_path", "manifest_path", "model_identifier"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"semantic_config missing keys: {sorted(missing)}")
    if not isinstance(data.get("enabled"), bool):
        raise ValueError("semantic_config.enabled must be a boolean")


def _validate_credibility_rules(data: Mapping[str, Any]) -> None:
    multiplier_bounds = data.get("multiplier_bounds")
    if not isinstance(multiplier_bounds, dict):
        raise ValueError("credibility_rules.multiplier_bounds must be a mapping")
    rules = data.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("credibility_rules.rules must be a non-empty mapping")
    base_required = {"enabled", "severity", "score_contribution", "explanation_label", "candidate_facing_reasoning"}
    severities = {"minor", "moderate", "severe"}
    for rule_id, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"credibility_rules.rules.{rule_id} must be a mapping")
        missing = base_required - set(rule)
        if missing:
            raise ValueError(f"credibility_rules.rules.{rule_id} missing keys: {sorted(missing)}")
        if rule["severity"] not in severities:
            raise ValueError(f"credibility_rules.rules.{rule_id}.severity must be one of {sorted(severities)}")
    title_rule = rules.get("title_description_contradiction")
    if not isinstance(title_rule, dict):
        raise ValueError("credibility_rules.rules.title_description_contradiction must be declared")
    title_required = {
        "supported_title_families",
        "contradiction_handling_policy",
        "minimum_evidence_threshold",
        "supporting_description_terms",
    }
    missing_title = title_required - set(title_rule)
    if missing_title:
        raise ValueError(f"credibility_rules.rules.title_description_contradiction missing keys: {sorted(missing_title)}")


def load_phase2_config(
    role_rubric_path: str | Path = ROLE_RUBRIC_PATH,
    evidence_patterns_path: str | Path = EVIDENCE_PATTERNS_PATH,
    scoring_weights_path: str | Path = SCORING_WEIGHTS_PATH,
    credibility_rules_path: str | Path = CREDIBILITY_RULES_PATH,
    behavioral_rules_path: str | Path = BEHAVIORAL_RULES_PATH,
    semantic_config_path: str | Path = SEMANTIC_CONFIG_PATH,
) -> Phase2ConfigBundle:
    paths = {
        "role_rubric": Path(role_rubric_path),
        "evidence_patterns": Path(evidence_patterns_path),
        "scoring_weights": Path(scoring_weights_path),
        "credibility_rules": Path(credibility_rules_path),
        "behavioral_rules": Path(behavioral_rules_path),
        "semantic_config": Path(semantic_config_path),
    }
    loaded = {name: _load_yaml_map(path) for name, path in paths.items()}
    _validate_evidence_patterns(loaded["evidence_patterns"])
    _validate_semantic_config(loaded["semantic_config"])
    _validate_credibility_rules(loaded["credibility_rules"])
    bundle_hashes = {
        "evidence": sha256_json({
            "role_rubric": loaded["role_rubric"],
            "evidence_patterns": loaded["evidence_patterns"],
        }),
        "scoring": sha256_json({
            "role_rubric": loaded["role_rubric"],
            "scoring_weights": loaded["scoring_weights"],
        }),
        "credibility": sha256_json({
            "role_rubric": loaded["role_rubric"],
            "credibility_rules": loaded["credibility_rules"],
        }),
        "behavioral": sha256_json({
            "role_rubric": loaded["role_rubric"],
            "behavioral_rules": loaded["behavioral_rules"],
        }),
        "semantic": sha256_json({
            "role_rubric": loaded["role_rubric"],
            "semantic_config": loaded["semantic_config"],
        }),
    }
    return Phase2ConfigBundle(
        paths=MappingProxyType({name: str(path) for name, path in paths.items()}),
        raw_sha256=MappingProxyType({name: sha256_file(path) for name, path in paths.items()}),
        bundle_hashes=MappingProxyType(bundle_hashes),
        role_rubric=MappingProxyType(loaded["role_rubric"]),
        evidence_patterns=MappingProxyType(loaded["evidence_patterns"]),
        scoring_weights=MappingProxyType(loaded["scoring_weights"]),
        credibility_rules=MappingProxyType(loaded["credibility_rules"]),
        behavioral_rules=MappingProxyType(loaded["behavioral_rules"]),
        semantic_config=MappingProxyType(loaded["semantic_config"]),
    )


def phase2_config_manifest(config: Phase2ConfigBundle) -> dict[str, Any]:
    return {
        "paths": dict(config.paths),
        "raw_sha256": dict(config.raw_sha256),
        "bundle_hashes": dict(config.bundle_hashes),
        "semantic_config": dict(config.semantic_config),
        "credibility_rule_ids": list(dict(config.credibility_rules.get("rules", {})).keys()),
    }

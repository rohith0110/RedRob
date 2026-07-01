import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import parse_date, run_audit
from .atomic_writes import copy_atomic
from .baseline_scoring import run_scoring
from .behavioral import run_behavioral_analysis
from .checkpointing import begin_stage_attempt, record_stage_skip, stage_validation_reason, update_stage
from .credibility import run_credibility_analysis
from .evidence import run_evidence_extraction
from .hashing import file_fingerprint, sha256_file, sha256_json
from .logging_utils import RunLogger
from .manifest import base_manifest, write_manifest
from .normalization import run_normalization
from .paths import ensure_run_dirs
from .phase2_config import load_phase2_config, phase2_config_manifest
from .reasoning_v2 import run_reasoning_generation
from .runtime_config import load_runtime_config, runtime_config_manifest
from .scoring_config import config_manifest, load_scoring_config
from .submission_v2 import write_v2_submission
from .submission_validation import require_valid_submission
from .submission_writer import write_submission
from .v2_scoring import run_v2_scoring

STAGE_VERSIONS = {
    "audit": "phase1.5-audit-v1",
    "normalized": "phase2-normalization-v2",
    "scores": "phase2-scoring-v2",
    "submission": "phase1.5-submission-v1",
    "evidence": "phase2-evidence-v1",
    "credibility": "phase2-credibility-v1",
    "behavioral": "phase2-behavioral-v1",
    "scores_v2": "phase2-scores-v1",
    "reasoning_v2": "phase2-reasoning-v1",
    "submission_v2": "phase2-submission-v1",
}


def utc_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", default="./submissions/baseline_submission.csv")
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default="./runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--allow-small-sample", action="store_true")
    p.add_argument("--baseline-weights", default="configs/baseline_weights.yaml")
    p.add_argument("--role-rubric", default="configs/role_rubric.yaml")
    p.add_argument("--runtime-config", default="configs/runtime.yaml")
    p.add_argument("--mode", choices=("baseline", "v2"), default="baseline")
    p.add_argument("--evidence-profile", action="store_true")
    return p


def config_hash(config_paths):
    return sha256_json({str(p): p.read_text(encoding="utf-8") for p in config_paths if p.exists()})


def artifact_fingerprint(paths: list[str | Path], row_count: int | None = None) -> dict:
    return {
        "row_count": row_count,
        "files": {str(Path(p)): sha256_file(p) for p in paths if Path(p).exists()},
    }


def stage_fingerprint(stage: str, payload: dict) -> str:
    return sha256_json({"stage": stage, "version": STAGE_VERSIONS[stage], **payload})


def source_hashes(paths: list[str]) -> dict:
    return {path: sha256_file(path) for path in paths}


def combined_config_hash(*values: str) -> str:
    return sha256_json([value for value in values if value])


def score_stage_payload(
    upstream_normalized: dict,
    resolved_config_hash: str,
    seed: int,
    source_hashes: dict[str, str],
    schema_version: str = "baseline-score-breakdown-v1",
) -> dict:
    return {
        "upstream_normalized": upstream_normalized,
        "resolved_config_hash": resolved_config_hash,
        "source_hashes": source_hashes,
        "seed": seed,
        "schema_version": schema_version,
    }


def should_skip(state_path, stage, fp_hash, cfg_hash, args, logger, upstream_reran=False):
    if args.force:
        logger(stage, "force_rerun", "force requested; rerunning stage")
        return False
    if upstream_reran:
        logger(stage, "rerun", "upstream stage reran; refreshing downstream stage")
        return False
    valid, reason = stage_validation_reason(state_path, stage, fp_hash, cfg_hash)
    if args.resume and valid:
        record_stage_skip(state_path, stage, fp_hash, cfg_hash, reason)
        logger(stage, "resume_skip", reason)
        return True
    logger(stage, "rerun", reason if args.resume else "resume not requested; running stage")
    return False


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "v2" and args.out == "./submissions/baseline_submission.csv":
        args.out = "./submissions/v2_submission.csv"
    run_id = args.run_id or utc_run_id()
    run_path = ensure_run_dirs(Path(args.run_dir) / run_id)
    state_path = run_path / "state.json"
    logger = RunLogger(run_id, run_path, args.log_level)
    scoring_config = load_scoring_config(args.baseline_weights, args.role_rubric)
    runtime_config = load_runtime_config(args.runtime_config)
    phase2_config = load_phase2_config(args.role_rubric)
    logger("config", "resolved", f"{scoring_config.scoring_code_version} {scoring_config.resolved_sha256}")
    logger("config", "runtime", f"{runtime_config.version} {runtime_config.processing_mode}")
    config_paths = [
        Path(args.role_rubric),
        Path(args.baseline_weights),
        Path(args.runtime_config),
        Path("configs/evidence_patterns.yaml"),
        Path("configs/scoring_weights.yaml"),
        Path("configs/credibility_rules.yaml"),
        Path("configs/behavioral_rules.yaml"),
        Path("configs/semantic_config.yaml"),
        Path("configs/logging.yaml"),
    ]
    raw_input = file_fingerprint(args.candidates)
    raw_input_hash = sha256_json(raw_input)
    logging_cfg_hash = config_hash([Path("configs/logging.yaml")])
    runtime_cfg_hash = runtime_config.resolved_sha256
    no_cfg_hash = "no-stage-config"
    dataset_reference_date = parse_date(runtime_config.dataset_reference_date)
    audit_cfg_hash = combined_config_hash(logging_cfg_hash, runtime_cfg_hash)
    phase2_evidence_cfg_hash = combined_config_hash(phase2_config.bundle_hashes["evidence"], runtime_cfg_hash)
    phase2_credibility_cfg_hash = combined_config_hash(phase2_config.bundle_hashes["credibility"], runtime_cfg_hash)
    phase2_behavioral_cfg_hash = combined_config_hash(phase2_config.bundle_hashes["behavioral"], runtime_cfg_hash)
    phase2_scoring_cfg_hash = combined_config_hash(phase2_config.bundle_hashes["scoring"], phase2_config.bundle_hashes["semantic"])
    manifest = base_manifest(run_id, args.candidates, vars(args), config_paths)
    manifest["resolved_scoring_configuration"] = config_manifest(scoring_config)
    manifest["resolved_runtime_configuration"] = runtime_config_manifest(runtime_config)
    manifest["resolved_phase2_configuration"] = phase2_config_manifest(phase2_config)
    manifest["fingerprint_schema_versions"] = dict(STAGE_VERSIONS)
    manifest_path = run_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    t0 = time.perf_counter()
    current_stage = None
    current_fp = None
    current_cfg = no_cfg_hash
    reran_stages = set()

    try:
        audit_fp = stage_fingerprint("audit", {
            "raw_input": raw_input,
            "dataset_reference_date": runtime_config.dataset_reference_date,
            "source_hashes": source_hashes(["src/audit.py", "src/io.py"]),
            "schema_version": "audit-summary-v1",
        })
        current_stage, current_fp, current_cfg = "audit", audit_fp, audit_cfg_hash
        if not should_skip(state_path, "audit", audit_fp, audit_cfg_hash, args, logger):
            reran_stages.add("audit")
            start = time.perf_counter()
            begin_stage_attempt(state_path, "audit", audit_fp, audit_cfg_hash)
            logger("audit", "started", "input validation started")
            audit = run_audit(args.candidates, run_path, logger=logger, dataset_reference_date=dataset_reference_date)
            if audit["duplicate_candidate_ids"]:
                raise ValueError(f"duplicate candidate IDs found: {audit['duplicate_candidate_ids']}")
            logger("audit", "completed", "audit completed", processed_count=audit["total_records"])
            manifest["elapsed_time_by_stage"]["audit"] = round(time.perf_counter() - start, 3)
            update_stage(state_path, "audit", "completed", audit_fp, audit_cfg_hash, [run_path / "audit" / "audit_summary.json", run_path / "audit" / "audit_summary.md", run_path / "audit" / "field_profile.json", run_path / "audit" / "issues.jsonl"], audit["valid_records"], metadata={"dataset_reference_date": runtime_config.dataset_reference_date})
        else:
            audit = json.loads((run_path / "audit" / "audit_summary.json").read_text(encoding="utf-8"))

        audit_artifacts = artifact_fingerprint([run_path / "audit" / "audit_summary.json", run_path / "audit" / "issues.jsonl"], audit["valid_records"])
        normalized_fp = stage_fingerprint("normalized", {
            "raw_input_hash": raw_input_hash,
            "upstream_audit": audit_artifacts,
            "processing_mode": runtime_config.processing_mode,
            "dataset_reference_date": runtime_config.dataset_reference_date,
            "source_hashes": source_hashes(["src/normalization.py", "src/audit.py", "src/io.py", "src/runtime_config.py"]),
            "schema_version": "normalized-candidate-v2",
        })
        current_stage, current_fp, current_cfg = "normalized", normalized_fp, runtime_cfg_hash
        if not should_skip(state_path, "normalized", normalized_fp, runtime_cfg_hash, args, logger, upstream_reran="audit" in reran_stages):
            reran_stages.add("normalized")
            start = time.perf_counter()
            begin_stage_attempt(state_path, "normalized", normalized_fp, runtime_cfg_hash)
            logger("normalization", "started", "normalization started")
            normalized = run_normalization(
                args.candidates,
                run_path,
                logger=logger,
                processing_mode=runtime_config.processing_mode,
                chunk_size=runtime_config.chunk_size,
                dataset_reference_date=dataset_reference_date,
            )
            logger("normalization", "completed", "normalization completed", processed_count=normalized["normalized_output_row_count"])
            manifest["elapsed_time_by_stage"]["normalization"] = round(time.perf_counter() - start, 3)
            update_stage(
                state_path,
                "normalized",
                "completed",
                normalized_fp,
                runtime_cfg_hash,
                [Path(normalized["jsonl_path"]), Path(normalized["parquet_path"]), Path(normalized["candidate_context_path"]), run_path / "normalized" / "normalization_summary.json"],
                normalized["normalized_output_row_count"],
                metadata=runtime_config_manifest(runtime_config),
            )
        else:
            normalized = json.loads((run_path / "normalized" / "normalization_summary.json").read_text(encoding="utf-8"))

        normalized_artifacts = artifact_fingerprint(
            [normalized["jsonl_path"], normalized["parquet_path"], normalized["candidate_context_path"]],
            normalized["normalized_output_row_count"],
        )
        baseline_normalized_artifacts = artifact_fingerprint(
            [normalized["jsonl_path"], normalized["parquet_path"]],
            normalized["normalized_output_row_count"],
        )

        if args.mode == "baseline":
            scores_fp = stage_fingerprint(
                "scores",
                score_stage_payload(
                    upstream_normalized=baseline_normalized_artifacts,
                    resolved_config_hash=scoring_config.resolved_sha256,
                    seed=args.seed,
                    source_hashes=source_hashes(["src/baseline_scoring.py", "src/scoring_config.py", "src/cli.py"]),
                ),
            )
            current_stage, current_fp, current_cfg = "scores", scores_fp, scoring_config.resolved_sha256
            if not should_skip(state_path, "scores", scores_fp, scoring_config.resolved_sha256, args, logger, upstream_reran="normalized" in reran_stages):
                reran_stages.add("scores")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "scores", scores_fp, scoring_config.resolved_sha256)
                logger("scoring", "started", "scoring started")
                scored = run_scoring(run_path, logger=logger, config=scoring_config)
                manifest["elapsed_time_by_stage"]["scoring"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "scores", "completed", scores_fp, scoring_config.resolved_sha256, [Path(scored["score_breakdown_csv"]), Path(scored["score_breakdown_parquet"]), run_path / "scores" / "top_500_baseline.csv", run_path / "reports" / "baseline_summary.md"], scored["scoring_output_row_count"], metadata=config_manifest(scoring_config))
            else:
                scored = {"scoring_output_row_count": len(pd.read_csv(run_path / "scores" / "baseline_score_breakdown.csv"))}

            scoring_artifacts = artifact_fingerprint([run_path / "scores" / "baseline_score_breakdown.csv", run_path / "scores" / "baseline_score_breakdown.parquet"], scored["scoring_output_row_count"])
            submission_fp = stage_fingerprint("submission", {
                "upstream_scoring": scoring_artifacts,
                "source_hashes": source_hashes(["src/submission_writer.py", "src/baseline_reasoning.py", "src/submission_validation.py", "validate_submission.py"]),
                "output_schema_version": "submission-v1",
                "official_output": str(Path(args.out).resolve()),
            })
            current_stage, current_fp, current_cfg = "submission", submission_fp, no_cfg_hash
            if not should_skip(state_path, "submission", submission_fp, no_cfg_hash, args, logger, upstream_reran="scores" in reran_stages):
                reran_stages.add("submission")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "submission", submission_fp, no_cfg_hash)
                score_rows = pd.read_csv(run_path / "scores" / "baseline_score_breakdown.csv").to_dict("records")
                if len(score_rows) < 100 and not args.allow_small_sample:
                    raise ValueError("fewer than 100 candidates requires --allow-small-sample")
                run_submission = run_path / "submissions" / ("sample_baseline_submission.csv" if len(score_rows) < 100 else "baseline_submission.csv")
                write_submission(score_rows, run_submission, limit=min(100, len(score_rows)))
                out = Path(args.out)
                output_paths = [run_submission]
                if len(score_rows) >= 100:
                    copy_atomic(run_submission, out, require_valid_submission)
                    output_paths.append(out)
                    logger("submission", "validation_completed", "submission validation completed", output_path=out)
                manifest["elapsed_time_by_stage"]["submission"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "submission", "completed", submission_fp, no_cfg_hash, output_paths, min(100, len(score_rows)))

            manifest["scoring_output_row_count"] = scored["scoring_output_row_count"]
            manifest["artifact_paths"] = {
                "audit_summary": str(run_path / "audit" / "audit_summary.json"),
                "normalized_jsonl": str(run_path / "normalized" / "candidates_normalized.jsonl"),
                "normalized_parquet": str(run_path / "normalized" / "candidates_normalized.parquet"),
                "score_breakdown_csv": str(run_path / "scores" / "baseline_score_breakdown.csv"),
                "submission": str(Path(args.out)),
            }
        else:
            evidence_fp = stage_fingerprint("evidence", {
                "upstream_normalized": normalized_artifacts,
                "processing_mode": runtime_config.processing_mode,
                "profile_enabled": bool(args.evidence_profile),
                "resolved_evidence_bundle_hash": phase2_config.bundle_hashes["evidence"],
                "dataset_reference_date": runtime_config.dataset_reference_date,
                "source_hashes": source_hashes(["src/evidence.py", "src/phase2_config.py"]),
                "schema_version": "evidence-ledger-v1",
            })
            current_stage, current_fp, current_cfg = "evidence", evidence_fp, phase2_evidence_cfg_hash
            if not should_skip(state_path, "evidence", evidence_fp, phase2_evidence_cfg_hash, args, logger, upstream_reran="normalized" in reran_stages):
                reran_stages.add("evidence")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "evidence", evidence_fp, phase2_evidence_cfg_hash)
                evidence = run_evidence_extraction(run_path, logger=logger, config=phase2_config, dataset_reference_date=dataset_reference_date, profile=bool(args.evidence_profile))
                manifest["elapsed_time_by_stage"]["evidence"] = round(time.perf_counter() - start, 3)
                evidence_outputs = [Path(evidence["evidence_ledger_jsonl"]), Path(evidence["evidence_summary_parquet"]), Path(evidence["evidence_summary_csv"]), Path(evidence["evidence_quality_report_json"]), Path(evidence["evidence_report_md"])]
                if evidence.get("evidence_profile_json"):
                    evidence_outputs.extend([Path(evidence["evidence_profile_json"]), Path(evidence["evidence_profile_md"])])
                update_stage(state_path, "evidence", "completed", evidence_fp, phase2_evidence_cfg_hash, evidence_outputs, evidence["candidate_count"], metadata=phase2_config_manifest(phase2_config) | {"dataset_reference_date": runtime_config.dataset_reference_date, "profile_enabled": bool(args.evidence_profile)})
            else:
                evidence = {"candidate_count": len(pd.read_parquet(run_path / "evidence" / "evidence_summary.parquet"))}

            evidence_artifacts = artifact_fingerprint([run_path / "evidence" / "evidence_summary.parquet", run_path / "evidence" / "evidence_ledger.jsonl"], evidence["candidate_count"])
            credibility_fp = stage_fingerprint("credibility", {
                "upstream_evidence": evidence_artifacts,
                "resolved_credibility_bundle_hash": phase2_config.bundle_hashes["credibility"],
                "dataset_reference_date": runtime_config.dataset_reference_date,
                "source_hashes": source_hashes(["src/credibility.py", "src/phase2_config.py"]),
                "schema_version": "credibility-breakdown-v1",
            })
            current_stage, current_fp, current_cfg = "credibility", credibility_fp, phase2_credibility_cfg_hash
            if not should_skip(state_path, "credibility", credibility_fp, phase2_credibility_cfg_hash, args, logger, upstream_reran="evidence" in reran_stages):
                reran_stages.add("credibility")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "credibility", credibility_fp, phase2_credibility_cfg_hash)
                credibility = run_credibility_analysis(run_path, logger=logger, config=phase2_config, dataset_reference_date=dataset_reference_date)
                manifest["elapsed_time_by_stage"]["credibility"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "credibility", "completed", credibility_fp, phase2_credibility_cfg_hash, [Path(credibility["credibility_breakdown_parquet"]), Path(credibility["credibility_breakdown_csv"]), Path(credibility["credibility_rules_triggered_jsonl"]), Path(credibility["credibility_report_md"])], len(pd.read_parquet(credibility["credibility_breakdown_parquet"])), metadata=phase2_config_manifest(phase2_config) | {"dataset_reference_date": runtime_config.dataset_reference_date})

            behavioral_fp = stage_fingerprint("behavioral", {
                "upstream_evidence": evidence_artifacts,
                "resolved_behavioral_bundle_hash": phase2_config.bundle_hashes["behavioral"],
                "dataset_reference_date": runtime_config.dataset_reference_date,
                "source_hashes": source_hashes(["src/behavioral.py", "src/phase2_config.py"]),
                "schema_version": "behavioral-breakdown-v1",
            })
            current_stage, current_fp, current_cfg = "behavioral", behavioral_fp, phase2_behavioral_cfg_hash
            if not should_skip(state_path, "behavioral", behavioral_fp, phase2_behavioral_cfg_hash, args, logger, upstream_reran="evidence" in reran_stages):
                reran_stages.add("behavioral")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "behavioral", behavioral_fp, phase2_behavioral_cfg_hash)
                behavioral = run_behavioral_analysis(run_path, logger=logger, config=phase2_config, dataset_reference_date=dataset_reference_date)
                manifest["elapsed_time_by_stage"]["behavioral"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "behavioral", "completed", behavioral_fp, phase2_behavioral_cfg_hash, [Path(behavioral["availability_breakdown_parquet"]), Path(behavioral["availability_breakdown_csv"]), Path(behavioral["behavioral_report_md"])], len(pd.read_parquet(behavioral["availability_breakdown_parquet"])), metadata=phase2_config_manifest(phase2_config) | {"dataset_reference_date": runtime_config.dataset_reference_date})

            credibility_artifacts = artifact_fingerprint([run_path / "credibility" / "credibility_breakdown.parquet"], evidence["candidate_count"])
            behavioral_artifacts = artifact_fingerprint([run_path / "behavioral" / "availability_breakdown.parquet"], evidence["candidate_count"])
            scores_v2_fp = stage_fingerprint("scores_v2", {
                "upstream_evidence": evidence_artifacts,
                "upstream_credibility": credibility_artifacts,
                "upstream_behavioral": behavioral_artifacts,
                "resolved_scoring_bundle_hash": phase2_config.bundle_hashes["scoring"],
                "resolved_semantic_bundle_hash": phase2_config.bundle_hashes["semantic"],
                "source_hashes": source_hashes(["src/v2_scoring.py", "src/phase2_config.py"]),
                "seed": args.seed,
                "schema_version": "score-breakdown-v2",
            })
            current_stage, current_fp, current_cfg = "scores_v2", scores_v2_fp, phase2_scoring_cfg_hash
            if not should_skip(state_path, "scores_v2", scores_v2_fp, phase2_scoring_cfg_hash, args, logger, upstream_reran=("evidence" in reran_stages or "credibility" in reran_stages or "behavioral" in reran_stages)):
                reran_stages.add("scores_v2")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "scores_v2", scores_v2_fp, phase2_scoring_cfg_hash)
                scores_v2 = run_v2_scoring(run_path, logger=logger, config=phase2_config)
                manifest["elapsed_time_by_stage"]["scores_v2"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "scores_v2", "completed", scores_v2_fp, phase2_scoring_cfg_hash, [Path(scores_v2["score_breakdown_v2_parquet"]), Path(scores_v2["score_breakdown_v2_csv"]), Path(scores_v2["top_1000_diagnostics_csv"]), Path(scores_v2["cohort_summary_json"]), Path(scores_v2["phase2_ranking_report_md"]), Path(scores_v2["semantic_status_json"])], len(pd.read_parquet(scores_v2["score_breakdown_v2_parquet"])), metadata=phase2_config_manifest(phase2_config))

            scores_v2_artifacts = artifact_fingerprint([run_path / "scores" / "score_breakdown_v2.parquet"], evidence["candidate_count"])
            reasoning_fp = stage_fingerprint("reasoning_v2", {
                "upstream_scores_v2": scores_v2_artifacts,
                "upstream_evidence": evidence_artifacts,
                "upstream_credibility": credibility_artifacts,
                "upstream_behavioral": behavioral_artifacts,
                "resolved_reasoning_bundle_hash": phase2_config.bundle_hashes["evidence"],
                "source_hashes": source_hashes(["src/reasoning_v2.py"]),
                "schema_version": "reasoning-v2",
            })
            current_stage, current_fp, current_cfg = "reasoning_v2", reasoning_fp, phase2_config.bundle_hashes["evidence"]
            if not should_skip(state_path, "reasoning_v2", reasoning_fp, phase2_config.bundle_hashes["evidence"], args, logger, upstream_reran="scores_v2" in reran_stages):
                reran_stages.add("reasoning_v2")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "reasoning_v2", reasoning_fp, phase2_config.bundle_hashes["evidence"])
                reasoning = run_reasoning_generation(run_path)
                manifest["elapsed_time_by_stage"]["reasoning_v2"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "reasoning_v2", "completed", reasoning_fp, phase2_config.bundle_hashes["evidence"], [Path(reasoning["reasoning_parquet"]), Path(reasoning["reasoning_jsonl"]), Path(reasoning["reasoning_style_lint_jsonl"]), Path(reasoning["reasoning_grounding_report_json"]), Path(reasoning["reasoning_grounding_report_md"]), Path(reasoning["reasoning_diversity_report_json"]), Path(reasoning["reasoning_diversity_report_md"]), Path(reasoning["reasoning_style_quality_report_json"]), Path(reasoning["reasoning_style_quality_report_md"])], len(pd.read_parquet(reasoning["reasoning_parquet"])))

            submission_v2_fp = stage_fingerprint("submission_v2", {
                "upstream_scores_v2": scores_v2_artifacts,
                "upstream_reasoning_v2": artifact_fingerprint([run_path / "reasoning" / "reasoning_v2.parquet"], min(100, evidence["candidate_count"])),
                "source_hashes": source_hashes(["src/submission_v2.py", "src/submission_validation.py", "validate_submission.py"]),
                "output_schema_version": "submission-v1",
                "official_output": str(Path(args.out).resolve()),
            })
            current_stage, current_fp, current_cfg = "submission_v2", submission_v2_fp, no_cfg_hash
            if not should_skip(state_path, "submission_v2", submission_v2_fp, no_cfg_hash, args, logger, upstream_reran="reasoning_v2" in reran_stages):
                reran_stages.add("submission_v2")
                start = time.perf_counter()
                begin_stage_attempt(state_path, "submission_v2", submission_v2_fp, no_cfg_hash)
                score_rows = pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet")
                if len(score_rows) < 100 and not args.allow_small_sample:
                    raise ValueError("fewer than 100 candidates requires --allow-small-sample")
                run_submission = run_path / "submissions" / ("sample_v2_submission.csv" if len(score_rows) < 100 else "v2_submission.csv")
                write_v2_submission(run_path, run_submission, limit=min(100, len(score_rows)))
                out = Path(args.out)
                output_paths = [run_submission]
                if len(score_rows) >= 100:
                    copy_atomic(run_submission, out, require_valid_submission)
                    output_paths.append(out)
                    logger("submission_v2", "validation_completed", "submission validation completed", output_path=out)
                manifest["elapsed_time_by_stage"]["submission_v2"] = round(time.perf_counter() - start, 3)
                update_stage(state_path, "submission_v2", "completed", submission_v2_fp, no_cfg_hash, output_paths, min(100, len(score_rows)))

            manifest["scoring_output_row_count"] = len(pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet"))
            manifest["artifact_paths"] = {
                "audit_summary": str(run_path / "audit" / "audit_summary.json"),
                "normalized_jsonl": str(run_path / "normalized" / "candidates_normalized.jsonl"),
                "normalized_parquet": str(run_path / "normalized" / "candidates_normalized.parquet"),
                "candidate_context_jsonl": str(run_path / "normalized" / "candidate_context.jsonl"),
                "evidence_summary_parquet": str(run_path / "evidence" / "evidence_summary.parquet"),
                "credibility_breakdown_parquet": str(run_path / "credibility" / "credibility_breakdown.parquet"),
                "availability_breakdown_parquet": str(run_path / "behavioral" / "availability_breakdown.parquet"),
                "score_breakdown_v2_csv": str(run_path / "scores" / "score_breakdown_v2.csv"),
                "reasoning_parquet": str(run_path / "reasoning" / "reasoning_v2.parquet"),
                "reasoning_style_lint_jsonl": str(run_path / "reasoning" / "reasoning_style_lint.jsonl"),
                "submission": str(Path(args.out)),
            }

        manifest["source_dataset_row_count"] = audit["total_records"]
        manifest["normalized_output_row_count"] = normalized["normalized_output_row_count"]
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["final_status"] = "completed"
        logger("run", "runtime_summary", f"completed in {time.perf_counter() - t0:.3f}s")
        return 0
    except Exception as exc:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["final_status"] = "failed"
        logger("run", "error", str(exc), level="ERROR")
        if current_stage and current_fp:
            update_stage(state_path, current_stage, "failed", current_fp, current_cfg, [], 0, metadata={"error": str(exc)[:500]})
        raise
    finally:
        write_manifest(manifest_path, manifest)
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

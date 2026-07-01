from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.behavioral import run_behavioral_analysis
from src.credibility import run_credibility_analysis
from src.evidence import run_evidence_extraction
from src.io import validate_candidate_shape
from src.normalization import run_normalization
from src.paths import ensure_run_dirs
from src.reasoning_v2 import run_reasoning_generation
from src.submission_v2 import write_v2_submission
from src.v2_scoring import run_v2_scoring

MAX_SANDBOX_CANDIDATES = 100
DISPLAY_SCORE_COLUMNS = {
    "career_relevance_score": "Career relevance",
    "retrieval_ranking_depth_score": "Retrieval and ranking depth",
    "production_evaluation_score": "Production and evaluation",
    "product_shipper_score": "Product shipper",
    "experience_recent_coding_score": "Experience and coding recency",
    "corroborated_skill_score": "Corroborated skill",
    "preferred_differentiator_score": "Preferred differentiators",
    "location_logistics_score": "Location and logistics",
    "semantic_score": "Semantic score",
}


def should_run_ranking(*, has_upload: bool, run_clicked: bool) -> bool:
    return has_upload and run_clicked


def _decode_rows(payload: bytes, filename: str) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    text = payload.decode("utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
            return parsed["candidates"]
        if isinstance(parsed, dict):
            return [parsed]
    raise ValueError("Upload a .json or .jsonl file.")


def _validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No candidates found in the uploaded file.")
    if len(rows) > MAX_SANDBOX_CANDIDATES:
        raise ValueError(f"Sandbox accepts at most {MAX_SANDBOX_CANDIDATES} candidates per run.")

    issues: list[str] = []
    seen_ids: set[str] = set()
    for row_num, candidate in enumerate(rows, 1):
        shape_issues = validate_candidate_shape(candidate, row_num)
        if shape_issues:
            for issue in shape_issues[:3]:
                field = issue.get("field")
                if field:
                    issues.append(f"Row {row_num}: missing required field `{field}`.")
                else:
                    issues.append(f"Row {row_num}: missing `candidate_id`.")
            continue
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_ids:
            issues.append(f"Row {row_num}: duplicate candidate_id `{candidate_id}`.")
        seen_ids.add(candidate_id)

    if issues:
        raise ValueError(" ".join(issues[:5]))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_ledger_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["evidence_id"]] = row
    return rows


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in ("", None):
        return []
    return [str(item) for item in json.loads(value)]


def _score_components(score_row: pd.Series, behavioral_row: pd.Series) -> dict[str, float]:
    components = {
        label: round(float(score_row[column]), 6)
        for column, label in DISPLAY_SCORE_COLUMNS.items()
        if column in score_row.index
    }
    components["Availability modifier"] = round(float(behavioral_row["availability_multiplier"]), 6)
    components["Credibility modifier"] = round(float(score_row["credibility_multiplier"]), 6)
    return components


def _evidence_rows(evidence_ids: list[str], ledger: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for evidence_id in evidence_ids:
        item = ledger.get(evidence_id)
        if not item:
            continue
        rows.append(
            {
                "category": str(item.get("normalized_category", "")).replace("_", " "),
                "source_type": str(item.get("source_type", "")).replace("_", " "),
                "excerpt": str(item.get("exact_source_excerpt", "")).strip(),
            }
        )
    return rows


def rank_uploaded_candidates(payload: bytes, filename: str) -> dict[str, Any]:
    rows = _validate_rows(_decode_rows(payload, filename))
    with tempfile.TemporaryDirectory(prefix="redrob-sandbox-") as td:
        tmp_root = Path(td)
        input_path = tmp_root / "candidates.jsonl"
        _write_jsonl(input_path, rows)

        run_path = ensure_run_dirs(tmp_root / "run")
        run_normalization(input_path, run_path)
        run_evidence_extraction(run_path)
        run_credibility_analysis(run_path)
        run_behavioral_analysis(run_path)
        run_v2_scoring(run_path)
        run_reasoning_generation(run_path)

        submission_path = tmp_root / "sandbox_ranked_candidates.csv"
        write_v2_submission(run_path, submission_path, limit=min(100, len(rows)))

        submission = pd.read_csv(submission_path)
        scores = pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet").set_index("candidate_id")
        credibility = pd.read_parquet(run_path / "credibility" / "credibility_breakdown.parquet").set_index("candidate_id")
        behavioral = pd.read_parquet(run_path / "behavioral" / "availability_breakdown.parquet").set_index("candidate_id")
        ledger = _load_ledger_by_id(run_path / "evidence" / "evidence_ledger.jsonl")

        output_rows = []
        for row in submission.to_dict("records"):
            candidate_id = row["candidate_id"]
            score_row = scores.loc[candidate_id]
            behavioral_row = behavioral.loc[candidate_id]
            credibility_row = credibility.loc[candidate_id]
            positive_ids = _json_list(score_row["top_positive_evidence_ids"])
            negative_ids = _json_list(score_row["top_negative_evidence_ids"])
            output_rows.append(
                {
                    "candidate_id": candidate_id,
                    "rank": int(row["rank"]),
                    "score": f"{float(row['score']):.6f}",
                    "reasoning": row["reasoning"],
                    "score_components": _score_components(score_row, behavioral_row),
                    "credibility_modifier": round(float(credibility_row["credibility_multiplier"]), 6),
                    "availability_modifier": round(float(behavioral_row["availability_multiplier"]), 6),
                    "location_logistics_score": round(float(behavioral_row["location_logistics_score"]), 6),
                    "positive_evidence": _evidence_rows(positive_ids[:3], ledger),
                    "negative_evidence": _evidence_rows(negative_ids[:2], ledger),
                }
            )

        return {
            "candidate_count": len(rows),
            "rows": output_rows,
            "download_csv": submission_path.read_text(encoding="utf-8"),
        }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Redrob Ranking Sandbox", layout="wide")
    st.title("Redrob Ranking Sandbox")
    st.caption("Local sample runner that uses the same ranking pipeline as the submission CLI.")
    st.info("Use this for small sample inspection only. Official submissions require 100 ranked rows from the full dataset.")

    uploaded = st.file_uploader("Upload candidate JSON or JSONL", type=["json", "jsonl"])
    run_clicked = st.button("Run ranking", type="primary", disabled=uploaded is None)
    if not uploaded:
        st.markdown("For a quick demo, upload `sample_data/demo_candidates.json` or `sample_data/demo_candidates.jsonl`.")
        return
    if not should_run_ranking(has_upload=True, run_clicked=run_clicked):
        st.caption("File ready. Click `Run ranking` to process it.")
        return

    try:
        result = rank_uploaded_candidates(uploaded.getvalue(), uploaded.name)
    except Exception as exc:  # pragma: no cover - streamlit UI path
        st.error(str(exc))
        return

    st.success(f"Processed {result['candidate_count']} candidates.")
    st.download_button(
        "Download ranked CSV",
        data=result["download_csv"],
        file_name="sandbox_ranked_candidates.csv",
        mime="text/csv",
    )

    table_rows = []
    for row in result["rows"]:
        table_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": row["score"],
                "credibility_modifier": row["credibility_modifier"],
                "availability_modifier": row["availability_modifier"],
                "reasoning": row["reasoning"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    for row in result["rows"]:
        with st.expander(f"Rank {row['rank']} - {row['candidate_id']}"):
            st.markdown(f"**Overall score:** `{row['score']}`")
            st.markdown(f"**Credibility modifier:** `{row['credibility_modifier']}`")
            st.markdown(f"**Availability modifier:** `{row['availability_modifier']}`")
            st.markdown(f"**Location and logistics score:** `{row['location_logistics_score']}`")
            st.markdown("**Reasoning**")
            st.write(row["reasoning"])
            st.markdown("**Score components**")
            st.json(row["score_components"])
            st.markdown("**Positive evidence excerpts**")
            st.json(row["positive_evidence"] or [])
            st.markdown("**Negative evidence excerpts**")
            st.json(row["negative_evidence"] or [])


if __name__ == "__main__":
    main()

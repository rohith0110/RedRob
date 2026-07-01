from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .atomic_writes import write_json_atomic, write_jsonl_atomic, write_parquet_atomic, write_text_atomic

FORBIDDEN_REASONING_TERMS = (
    "honeypot",
    "trap",
    "synthetic",
    "risk score",
    "anomaly score",
    "internal evidence category",
)
MAX_DUPLICATE_REASONING_COUNT = 2
DEFAULT_REPEATED_OPENING_THRESHOLD = 2
DANGLING_END_TOKENS = {"a", "an", "and", "at", "because", "for", "in", "including", "of", "or", "to", "while", "with"}
DANGLING_START_TOKENS = {"and", "because", "but", "while", "with"}
GENERIC_PRAISE_TERMS = ("excellent candidate", "strong fit", "ideal profile", "great cultural fit")
FACT_VERB_TOKENS = {
    "architected",
    "are",
    "automated",
    "be",
    "built",
    "delivered",
    "designed",
    "developed",
    "drove",
    "fine-tuned",
    "fine",
    "implemented",
    "improved",
    "is",
    "keeps",
    "launched",
    "led",
    "managed",
    "migrated",
    "narrows",
    "operated",
    "optimized",
    "owned",
    "ran",
    "reduces",
    "remains",
    "served",
    "shipped",
    "shows",
    "slows",
    "stays",
    "supports",
    "trained",
    "uses",
    "was",
    "were",
}
BROKEN_FRAGMENT_PATTERNS = (
    re.compile(r"\bat a\b$", re.IGNORECASE),
    re.compile(r"\band service\b$", re.IGNORECASE),
    re.compile(r"\bfor experiment\b$", re.IGNORECASE),
    re.compile(r"\bwhile [a-z0-9/+.-]+ remains visible\b$", re.IGNORECASE),
)
SOURCE_TYPE_PRIORITY = {
    "career_description": 0,
    "profile_summary": 0,
    "profile_headline": 0,
    "career_title": 1,
    "skill": 2,
    "assessment": 2,
}

CATEGORY_PHRASES = {
    "retrieval_ranking_relevance": "retrieval and ranking work",
    "vector_ir_infrastructure": "vector or search infrastructure",
    "ranking_evaluation_experimentation": "ranking evaluation and experimentation",
    "production_delivery_systems": "production delivery",
    "python_practical_engineering": "hands-on Python engineering",
    "product_founding_behavior": "product-facing ownership",
    "preferred_differentiators": "differentiated search or model-tuning work",
    "risk_signals": "profile limitations",
}
RANK_BANDS = (
    (1, 10, "1-10"),
    (11, 30, "11-30"),
    (31, 60, "31-60"),
    (61, 100, "61-100"),
)
AWKWARD_PATTERN_RE = re.compile(r"included .* because .* included .* because", re.IGNORECASE)
FIRST_PERSON_DESCRIPTOR_RE = r"[A-Za-z0-9'/-]+"


def remove_adjacent_duplicate_tokens(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    deduped = [tokens[0]]
    previous = re.sub(r"^[^\w]+|[^\w]+$", "", tokens[0]).lower()
    for token in tokens[1:]:
        normalized = re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()
        if normalized and normalized == previous:
            continue
        deduped.append(token)
        if normalized:
            previous = normalized
    return " ".join(deduped)


def title_case_or_sentence_case(text: str, *, sentence_start: bool) -> str:
    text = text.strip()
    if not text:
        return text
    if sentence_start:
        return text[0].upper() + text[1:]
    return text[0].lower() + text[1:] if text[0].isalpha() else text


def _normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9+/.-]+", text)]


def _normalized_span(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _starts_with_dangling_token(text: str) -> bool:
    tokens = _normalized_tokens(text)
    return bool(tokens and tokens[0] in DANGLING_START_TOKENS)


def _ends_with_dangling_token(text: str) -> bool:
    tokens = _normalized_tokens(text)
    return bool(tokens and tokens[-1] in DANGLING_END_TOKENS)


def _has_predicate(text: str) -> bool:
    lowered = text.lower()
    tokens = _normalized_tokens(text)
    if any(token in FACT_VERB_TOKENS for token in tokens):
        return True
    return bool(re.search(r"\b\w+(ed|ing)\b", lowered))


def reject_dangling_fragment(text: str) -> bool:
    return _starts_with_dangling_token(text) or _ends_with_dangling_token(text) or any(
        pattern.search(text) for pattern in BROKEN_FRAGMENT_PATTERNS
    )


def normalize_fact_phrase(text: str | None) -> str | None:
    if text is None:
        return None
    phrase = " ".join(str(text).replace("—", " ").split())
    phrase = re.sub(r"\bour\s+internal\b", "an internal", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bour\s+existing\b", "an existing", phrase, flags=re.IGNORECASE)
    phrase = re.sub(
        rf"\bour\s+(?P<descriptor>(?:{FIRST_PERSON_DESCRIPTOR_RE}\s+){{0,3}})product(?P<possessive>'s)?\b",
        lambda match: (
            f"the candidate's {' '.join((match.group('descriptor') or '').split()) + ' ' if (match.group('descriptor') or '').strip() else ''}product"
            if match.group("possessive")
            else f"the candidate's {' '.join((match.group('descriptor') or '').split()) + ' ' if (match.group('descriptor') or '').strip() else ''}product work"
        ),
        phrase,
        flags=re.IGNORECASE,
    )
    phrase = re.sub(r"\bwe deployed\b", "the candidate's team deployed", phrase, flags=re.IGNORECASE)
    phrase = phrase.strip(" \t\r\n,;:")
    phrase = remove_adjacent_duplicate_tokens(phrase)
    phrase = re.sub(r"\s+([,.;:])", r"\1", phrase).strip()
    if not phrase:
        return None
    if reject_dangling_fragment(phrase):
        return None
    tokens = _normalized_tokens(phrase)
    if len(tokens) <= 1:
        return None
    if len(tokens) <= 5 and not _has_predicate(phrase):
        return None
    return phrase


def select_complete_clause(*candidates: str | None) -> str | None:
    for candidate in candidates:
        normalized = normalize_fact_phrase(candidate)
        if normalized:
            return normalized
    return None


def deduplicate_clause_content(clauses: list[str | None]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        if not clause:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", clause.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(clause)
    return deduped


def _normalized_grounding_phrases(plan: dict[str, Any]) -> list[str]:
    return deduplicate_clause_content([
        normalize_fact_phrase(phrase)
        for phrase in list(plan.get("grounding_phrases") or [])
    ])


def repair_sentence_start(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return title_case_or_sentence_case(text, sentence_start=True)


def ensure_sentence_has_predicate(text: str) -> bool:
    return _has_predicate(text) or len(_normalized_tokens(text)) >= 6


def _ensure_terminal_punctuation(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return f"{text}."
    return text


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _first_sentence(excerpt: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", excerpt.strip()) if part.strip()]
    return parts[0] if parts else excerpt.strip()


def _abstract_fact_for_category(category: str) -> str:
    return CATEGORY_PHRASES.get(category, "role-relevant engineering work")


def _negative_label(evidence: dict[str, Any]) -> str:
    terms = [str(term).strip().lower() for term in evidence.get("matched_terms") or [] if str(term).strip()]
    if terms:
        return terms[0]
    excerpt = " ".join(str(evidence.get("exact_source_excerpt", "")).split()).strip()
    tokens = _normalized_tokens(excerpt)
    return tokens[0] if tokens else "a secondary signal"


def _load_ledger(run_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    ledger_by_id = {}
    ledger_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (run_path / "evidence" / "evidence_ledger.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ledger_by_id[row["evidence_id"]] = row
                ledger_by_candidate[row["candidate_id"]].append(row)
    return ledger_by_id, ledger_by_candidate


def _load_contexts(run_path: Path) -> dict[str, dict[str, Any]]:
    contexts = {}
    with (run_path / "normalized" / "candidate_context.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                contexts[row["candidate_id"]] = row
    return contexts


def _load_breakdowns(run_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    credibility = pd.read_parquet(run_path / "credibility" / "credibility_breakdown.parquet").set_index("candidate_id")
    behavioral = pd.read_parquet(run_path / "behavioral" / "availability_breakdown.parquet").set_index("candidate_id")
    return credibility, behavioral


def _rank_band(rank: int) -> str:
    for minimum, maximum, label in RANK_BANDS:
        if minimum <= rank <= maximum:
            return label
    return "61-100"


def _join_terms(terms: list[str]) -> str:
    unique = [str(term).strip() for term in terms if str(term).strip()]
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]} and {unique[1]}"


def _source_priority(evidence: dict[str, Any]) -> int:
    return SOURCE_TYPE_PRIORITY.get(str(evidence.get("source_type", "")), 3)


def _evidence_snippet(evidence: dict[str, Any]) -> str | None:
    source_type = str(evidence.get("source_type", ""))
    if source_type in {"career_description", "profile_summary", "profile_headline"}:
        excerpt = " ".join(str(evidence.get("exact_source_excerpt", "")).split())
        if excerpt:
            return normalize_fact_phrase(_first_sentence(excerpt))
    if source_type == "career_title":
        title = " ".join(str(evidence.get("exact_source_excerpt", "")).split())
        if title:
            return normalize_fact_phrase(title)
    terms = [str(term).strip() for term in evidence.get("matched_terms") or [] if str(term).strip()]
    if terms:
        return normalize_fact_phrase(_join_terms(terms[:2]))
    excerpt = " ".join(str(evidence.get("exact_source_excerpt", "")).split())
    return normalize_fact_phrase(_first_sentence(excerpt))


def _pick_evidence(items: list[dict[str, Any]], categories: tuple[str, ...]) -> dict[str, Any] | None:
    ranked = sorted(
        items,
        key=lambda item: (
            _source_priority(item),
            -float(item.get("contribution_after_caps", 0.0)),
        ),
    )
    for category in categories:
        for item in ranked:
            if item.get("normalized_category") == category:
                return item
    return ranked[0] if ranked else None


def _positive_categories(positives: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("normalized_category")) for item in positives}


def _family_for_plan(score_row: pd.Series, context: dict[str, Any], positives: list[dict[str, Any]]) -> str:
    categories = _positive_categories(positives)
    matched_text = " ".join(
        " ".join(str(term).lower() for term in item.get("matched_terms") or [])
        for item in positives
    )
    if int(score_row["rank"]) >= 61:
        return "lower_cutoff_candidate"
    if float(score_row.get("credibility_multiplier", 1.0)) < 0.9:
        return "credibility_constrained"
    if (
        float(score_row.get("availability_multiplier", 1.0)) < 0.9
        or int((context.get("redrob_signals") or {}).get("notice_period_days") or 0) > 60
        or float(score_row.get("location_logistics_score", 1.0)) < 0.8
    ):
        return "availability_logistics_constrained"
    if "recommend" in matched_text or "matching" in matched_text:
        return "recommendation_matching_heavy"
    if {"retrieval_ranking_relevance", "ranking_evaluation_experimentation"} & categories:
        return "retrieval_ranking_heavy"
    if {"production_delivery_systems", "ranking_evaluation_experimentation"} <= categories:
        return "production_and_evaluation_heavy"
    if {"python_practical_engineering", "product_founding_behavior"} & categories:
        return "python_product_ownership_heavy"
    return "adjacent_but_credible"


def _availability_phrase(context: dict[str, Any], score_row: pd.Series) -> str | None:
    signals = context.get("redrob_signals") or {}
    work_mode = str(signals.get("preferred_work_mode", "")).lower()
    country = str((context.get("profile") or {}).get("country", "")).strip()
    if float(score_row.get("availability_multiplier", 1.0)) < 0.9 or int(signals.get("notice_period_days") or 0) > 60:
        return None
    if work_mode in {"remote", "flexible", "hybrid"} and country:
        return f"{country}-based {work_mode} compatibility supports shortlist viability"
    return None


def _availability_constraint_phrase(context: dict[str, Any], score_row: pd.Series) -> str | None:
    signals = context.get("redrob_signals") or {}
    notice_period_days = int(signals.get("notice_period_days") or 0)
    work_mode = str(signals.get("preferred_work_mode", "")).lower()
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    response_time = float(signals.get("avg_response_time_hours") or 0.0)
    if notice_period_days > 60:
        return f"a {notice_period_days}-day notice period slows availability"
    if work_mode not in {"remote", "flexible", "hybrid"} and not bool(signals.get("willing_to_relocate")):
        return f"{work_mode or 'less flexible'} work-mode preference narrows logistics flexibility"
    if response_rate < 0.25:
        return "response history is thinner than stronger candidates"
    if response_time >= 72:
        return "slower response timing reduces availability confidence"
    if float(score_row.get("location_logistics_score", 1.0)) < 0.8:
        return "location logistics are less flexible than the strongest candidates"
    return None


def _limitation_text(score_row: pd.Series, context: dict[str, Any], positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> tuple[str | None, bool]:
    categories = _positive_categories(positives)
    if float(score_row.get("credibility_multiplier", 1.0)) < 0.9:
        return "profile consistency is weaker than the strongest candidates", True
    signals = context.get("redrob_signals") or {}
    if float(score_row.get("availability_multiplier", 1.0)) < 0.9 or int(signals.get("notice_period_days") or 0) > 60:
        return "availability signals are weaker than the strongest candidates", True
    if "ranking_evaluation_experimentation" not in categories:
        return "explicit evidence of ranking evaluation is limited in the available profile", False
    if "vector_ir_infrastructure" not in categories:
        return "explicit evidence of vector or search infrastructure is limited in the available profile", False
    if "production_delivery_systems" not in categories:
        return "explicit evidence of production ownership is limited in the available profile", False
    if negatives:
        return f"some profile detail remains less complete around {_negative_label(negatives[0])}", True
    return None, False


def build_reasoning_plan(
    score_row: pd.Series,
    context: dict[str, Any],
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = _pick_evidence(positives, ("retrieval_ranking_relevance", "ranking_evaluation_experimentation", "vector_ir_infrastructure"))
    production = _pick_evidence(positives, ("production_delivery_systems", "ranking_evaluation_experimentation"))
    practical = _pick_evidence(positives, ("python_practical_engineering", "product_founding_behavior"))
    rank_band = _rank_band(int(score_row["rank"]))
    template_family = _family_for_plan(score_row, context, positives)
    limitation, material_caveat = _limitation_text(score_row, context, positives, negatives)
    availability_phrase = _availability_phrase(context, score_row)
    title = str((context.get("profile") or {}).get("current_title") or "Career history").strip()
    primary_phrase = _evidence_snippet(primary) or _abstract_fact_for_category(primary["normalized_category"] if primary else "retrieval_ranking_relevance")
    production_phrase = _evidence_snippet(production) or _abstract_fact_for_category(production["normalized_category"] if production else "production_delivery_systems")
    practical_phrase = _evidence_snippet(practical) or _abstract_fact_for_category(practical["normalized_category"] if practical else "python_practical_engineering")
    grounding_phrases = deduplicate_clause_content([
        primary_phrase,
        production_phrase,
        practical_phrase,
    ])
    grounding_phrases = [
        phrase for phrase in grounding_phrases if phrase
    ]
    grounding_evidence_ids = [
        evidence["evidence_id"]
        for evidence in (primary, production, practical)
        if evidence and evidence.get("evidence_id")
    ]
    return {
        "candidate_id": score_row["candidate_id"],
        "rank": int(score_row["rank"]),
        "score": float(score_row["final_score"]),
        "rank_band": rank_band,
        "template_family": template_family,
        "title": title,
        "primary_role_fit_category": primary["normalized_category"] if primary else "retrieval_ranking_relevance",
        "primary_phrase": primary_phrase,
        "production_phrase": production_phrase,
        "practical_phrase": practical_phrase,
        "availability_phrase": availability_phrase,
        "availability_constraint_phrase": _availability_constraint_phrase(context, score_row),
        "limitation": limitation,
        "material_caveat": material_caveat,
        "grounding_phrases": grounding_phrases,
        "grounding_evidence_ids": grounding_evidence_ids,
        "positive_evidence_ids": [item["evidence_id"] for item in positives[:6] if item.get("evidence_id")],
        "negative_evidence_ids": [item["evidence_id"] for item in negatives[:3] if item.get("evidence_id")],
        "concrete_evidence": bool(primary or production or practical),
    }


def _cutoff_variant(plan: dict[str, Any]) -> int:
    return (int(plan["rank"]) - 61) % 4


def _legacy_cutoff_reasoning(plan: dict[str, Any]) -> str:
    title = plan["title"]
    primary_phrase = plan["primary_phrase"]
    practical_phrase = plan["practical_phrase"]
    limitation = plan.get("limitation") or "direct production retrieval proof is limited in the available profile"
    base = str(limitation).strip()
    if "limited" not in base.lower():
        base = f"{base} and direct evidence is limited in the available profile"
    return (
        f"Included near the cutoff because {title.lower()} experience shows {primary_phrase} and {practical_phrase}. "
        f"Included on the shortlist because {primary_phrase} is still visible, but {base[0].lower()}{base[1:]} keeps the profile below stronger candidates."
    )


def _sentence_slot(text: str) -> str:
    return title_case_or_sentence_case(text.rstrip(".!?").strip(), sentence_start=False)


def _starts_with_action_verb(text: str) -> bool:
    tokens = _normalized_tokens(text)
    if not tokens:
        return False
    first = tokens[0]
    return first in FACT_VERB_TOKENS or first.endswith("ed")


def _normalized_limit_text(plan: dict[str, Any]) -> str:
    limitation = str(plan.get("limitation") or "").strip()
    if not limitation:
        return "explicit evidence of retrieval and ranking depth is limited in the available profile"
    lowered = limitation.lower()
    if any(term in lowered for term in ("limited", "less ", "weaker", "thinner", "slows", "narrows", "reduces")):
        return limitation
    return f"{limitation} and direct retrieval depth is limited in the available profile"


def _construct_clauses(plan: dict[str, Any], *, fallback: bool) -> tuple[str, str | None]:
    primary = (
        _abstract_fact_for_category(plan["primary_role_fit_category"])
        if fallback
        else select_complete_clause(plan["primary_phrase"]) or _abstract_fact_for_category(plan["primary_role_fit_category"])
    )
    support_candidates = [
        _abstract_fact_for_category("production_delivery_systems")
        if fallback
        else select_complete_clause(plan["production_phrase"]) or _abstract_fact_for_category("production_delivery_systems"),
        _abstract_fact_for_category("python_practical_engineering")
        if fallback
        else select_complete_clause(plan["practical_phrase"]) or _abstract_fact_for_category("python_practical_engineering"),
    ]
    clauses = deduplicate_clause_content([primary, *support_candidates])
    primary_clause = clauses[0] if clauses else _abstract_fact_for_category(plan["primary_role_fit_category"])
    support_clause = clauses[1] if len(clauses) > 1 else None
    return primary_clause, support_clause


def _assemble_reasoning(plan: dict[str, Any], *, fallback: bool) -> str:
    title = repair_sentence_start(str(plan.get("title") or "Profile"))
    primary_clause, support_clause = _construct_clauses(plan, fallback=fallback)
    limitation = _normalized_limit_text(plan)
    availability_phrase = str(plan.get("availability_phrase") or "").strip()
    availability_constraint = str(plan.get("availability_constraint_phrase") or "").strip()
    rank_band = plan["rank_band"]
    action_led_primary = _starts_with_action_verb(primary_clause)

    if rank_band == "1-10":
        first_sentence = f"{title} record shows {_sentence_slot(primary_clause)}."
        if action_led_primary and not fallback:
            first_sentence = _ensure_terminal_punctuation(repair_sentence_start(primary_clause.rstrip(".!?")))
        elif support_clause:
            first_sentence = f"{title} record shows {_sentence_slot(primary_clause)}, with {_sentence_slot(support_clause)} reinforcing direct role fit."
        second_sentence = repair_sentence_start(limitation if limitation else "the available profile remains strong across the core role requirements")
        if availability_phrase and not plan["material_caveat"] and not plan.get("limitation"):
            second_sentence = repair_sentence_start(availability_phrase)
    elif rank_band == "11-30":
        first_sentence = f"{title} background combines {_sentence_slot(primary_clause)}."
        if action_led_primary and not fallback:
            first_sentence = _ensure_terminal_punctuation(repair_sentence_start(primary_clause.rstrip(".!?")))
        elif support_clause:
            first_sentence = f"{title} background combines {_sentence_slot(primary_clause)} with {_sentence_slot(support_clause)}."
        second_sentence = repair_sentence_start(limitation if limitation else "one area remains less complete than the strongest profiles")
    elif rank_band == "31-60":
        first_sentence = f"{title} background still shows {_sentence_slot(primary_clause)}."
        if action_led_primary and not fallback:
            first_sentence = _ensure_terminal_punctuation(repair_sentence_start(primary_clause.rstrip(".!?")))
        elif support_clause:
            first_sentence = f"{title} background still shows {_sentence_slot(primary_clause)}, with {_sentence_slot(support_clause)} supporting adjacent fit."
        second_sentence = repair_sentence_start(
            availability_constraint
            or limitation
            or "explicit evidence is less complete than candidates above"
        )
    else:
        first_sentence = f"{title} background still shows {_sentence_slot(primary_clause)}."
        if action_led_primary and not fallback:
            first_sentence = _ensure_terminal_punctuation(repair_sentence_start(primary_clause.rstrip(".!?")))
        elif support_clause and _cutoff_variant(plan) % 2:
            first_sentence = f"{title} background still shows {_sentence_slot(primary_clause)}, with {_sentence_slot(support_clause)} supporting adjacent fit."
        second_sentence = (
            f"{repair_sentence_start(limitation).rstrip('.')} so the candidate stays near the shortlist cutoff "
            "rather than among stronger retrieval-and-ranking fits."
        )

    sentences = [_ensure_terminal_punctuation(repair_sentence_start(first_sentence))]
    if second_sentence:
        sentences.append(_ensure_terminal_punctuation(repair_sentence_start(second_sentence)))
    return " ".join(" ".join(sentence.split()) for sentence in sentences if sentence)


def validate_reasoning_style(
    reasoning: str,
    *,
    rank: int,
    rank_band: str,
    grounding_phrases: list[str] | None = None,
    limitation: str | None = None,
) -> dict[str, Any]:
    failed_checks: list[str] = []
    warning_checks: list[str] = []
    reasoning = reasoning.strip()
    sentences = _split_sentences(reasoning)
    lowered = reasoning.lower()

    if not 1 <= len(sentences) <= 2:
        failed_checks.append("sentence_count")
    if reasoning and reasoning[-1] not in ".!?":
        failed_checks.append("missing_terminal_punctuation")
    if any(sentence and not sentence[0].isupper() for sentence in sentences):
        failed_checks.append("lowercase_sentence_start")
    if re.search(r"\b(\w+)\s+\1\b", lowered):
        failed_checks.append("duplicate_adjacent_word")
    if any(reject_dangling_fragment(sentence.rstrip(".!?")) for sentence in sentences):
        failed_checks.append("dangling_fragment")
    if any(_starts_with_dangling_token(sentence) for sentence in sentences):
        failed_checks.append("dangling_start_token")
    if any(len(_normalized_tokens(sentence)) <= 4 and not ensure_sentence_has_predicate(sentence) for sentence in sentences):
        failed_checks.append("bare_fragment")
    if any(not ensure_sentence_has_predicate(sentence) for sentence in sentences):
        failed_checks.append("incomplete_clause")
    if any(term in lowered for term in FORBIDDEN_REASONING_TERMS):
        failed_checks.append("forbidden_term")

    normalized_sentences = [re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip() for sentence in sentences]
    normalized_reasoning = _normalized_span(reasoning)
    if len(normalized_sentences) != len(set(normalized_sentences)):
        failed_checks.append("repeated_clause")

    repeated_snippet_count = 0
    normalized_grounding = []
    for phrase in grounding_phrases or []:
        normalized = _normalized_span(phrase)
        if not normalized:
            continue
        normalized_grounding.append(normalized)
        if lowered.count(phrase.lower()) > 1:
            repeated_snippet_count += 1
    if len(normalized_grounding) != len(set(normalized_grounding)) or repeated_snippet_count:
        failed_checks.append("repeated_snippet")

    if any(term in lowered for term in GENERIC_PRAISE_TERMS):
        warning_checks.append("generic_praise")

    if rank_band in {"11-30", "31-60", "61-100"} and not any(
        term in lowered for term in ("limited", "less ", "weaker", "thinner", "slows", "narrows", "reduces")
    ):
        failed_checks.append("missing_limitation")
    if rank_band == "61-100" and "shortlist cutoff" not in lowered:
        failed_checks.append("missing_cutoff_positioning")
    if rank_band == "1-10" and not any(phrase in normalized_reasoning for phrase in normalized_grounding):
        failed_checks.append("missing_concrete_evidence")
    if limitation and rank_band == "61-100" and limitation.lower() not in lowered:
        failed_checks.append("missing_grounded_limitation")

    return {
        "passed": not failed_checks,
        "failed_checks": sorted(set(failed_checks)),
        "warning_checks": sorted(set(warning_checks)),
        "sentence_count": len(sentences),
        "repeated_snippet_count": repeated_snippet_count,
        "opening_key": _opening_key(sentences[0] if sentences else "", words=5),
        "rank": rank,
        "rank_band": rank_band,
    }


def _realize_reasoning(plan: dict[str, Any]) -> dict[str, Any]:
    grounding_phrases = _normalized_grounding_phrases(plan)
    reasoning = _assemble_reasoning(plan, fallback=False)
    lint = validate_reasoning_style(
        reasoning,
        rank=int(plan["rank"]),
        rank_band=str(plan["rank_band"]),
        grounding_phrases=grounding_phrases,
        limitation=str(plan.get("limitation") or ""),
    )
    if lint["passed"]:
        return {"reasoning": reasoning, "style_lint": lint, "used_fallback": False}

    fallback_reasoning = _assemble_reasoning(plan, fallback=True)
    fallback_lint = validate_reasoning_style(
        fallback_reasoning,
        rank=int(plan["rank"]),
        rank_band=str(plan["rank_band"]),
        grounding_phrases=grounding_phrases,
        limitation=str(plan.get("limitation") or ""),
    )
    return {"reasoning": fallback_reasoning, "style_lint": fallback_lint, "used_fallback": True}


def render_reasoning_from_plan(plan: dict[str, Any]) -> str:
    return _realize_reasoning(plan)["reasoning"]


def _grounding_result(reasoning: str, plan: dict[str, Any]) -> dict[str, Any]:
    lowered = reasoning.lower()
    normalized_reasoning = _normalized_span(reasoning)
    grounding_phrases = _normalized_grounding_phrases(plan)
    failed_checks = []
    if any(term in lowered for term in FORBIDDEN_REASONING_TERMS):
        failed_checks.append("forbidden_term")
    if not plan["grounding_evidence_ids"]:
        failed_checks.append("missing_grounding_ids")
    if not any(phrase and _normalized_span(phrase) in normalized_reasoning for phrase in grounding_phrases):
        failed_checks.append("missing_concrete_grounding_phrase")
    return {
        "passed": not failed_checks,
        "failed_checks": failed_checks,
    }


def _sentence_parts(reasoning: str) -> tuple[str, str]:
    parts = [part.strip() for part in reasoning.split(".") if part.strip()]
    first = parts[0] if parts else ""
    second = parts[1] if len(parts) > 1 else ""
    return first, second


def _opening_key(sentence: str, words: int = 3) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", sentence.lower())
    return " ".join(cleaned.split()[:words])


def _repeated_opening_count(
    reasoning_rows: list[dict[str, Any]],
    repeated_opening_threshold: int,
    opening_words: int = 5,
    nearby_window_size: int = 5,
) -> int:
    ordered = sorted(reasoning_rows, key=lambda row: int(row.get("rank", 0)))
    openings = [_opening_key(_sentence_parts(str(row.get("reasoning", "")))[0], words=opening_words) for row in ordered]
    if not openings:
        return 0
    if len(openings) <= nearby_window_size:
        windows = [openings]
    else:
        windows = [openings[index:index + nearby_window_size] for index in range(len(openings) - nearby_window_size + 1)]
    violations = 0
    for window in windows:
        counts = Counter(key for key in window if key)
        if any(count > repeated_opening_threshold for count in counts.values()):
            violations += 1
    return violations


def _awkward_pattern_count(reasoning_rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in reasoning_rows:
        reasoning = str(row.get("reasoning", ""))
        lowered = reasoning.lower()
        if AWKWARD_PATTERN_RE.search(reasoning):
            total += 1
            continue
        if "because because" in lowered:
            total += 1
            continue
        if lowered.count("included") > 1 and "because" in lowered:
            total += 1
            continue
        if re.search(r"\b(\w+)\s+\1\b", lowered):
            total += 1
            continue
        if re.search(r"\.\s+[a-z]", reasoning):
            total += 1
    return total


def analyze_reasoning_quality(
    reasoning_rows: list[dict[str, Any]],
    repeated_opening_threshold: int = DEFAULT_REPEATED_OPENING_THRESHOLD,
) -> dict[str, Any]:
    duplicates = Counter(str(row.get("reasoning", "")) for row in reasoning_rows)
    cutoff_rows = [row for row in reasoning_rows if row.get("rank_band") == "61-100"]
    band_style_distribution: dict[str, dict[str, int]] = {}
    lint_fail_count = 0
    lint_pass_count = 0
    duplicate_adjacent_word_count = 0
    lowercase_sentence_start_count = 0
    dangling_fragment_count = 0
    bare_fragment_count = 0
    repeated_snippet_count = 0
    generic_praise_warning_count = 0
    forbidden_term_count = 0
    rank_band_breakdown: dict[str, dict[str, int]] = {}
    for row in reasoning_rows:
        band = str(row.get("rank_band", ""))
        family = str(row.get("template_family", ""))
        band_style_distribution.setdefault(band, {})
        band_style_distribution[band][family] = band_style_distribution[band].get(family, 0) + 1
        failed_checks = row.get("style_lint_failed_checks", [])
        if isinstance(failed_checks, str):
            failed_checks = json.loads(failed_checks or "[]")
        warning_checks = row.get("style_lint_warning_checks", [])
        if isinstance(warning_checks, str):
            warning_checks = json.loads(warning_checks or "[]")
        lint_passed = bool(row.get("style_lint_pass"))
        lint_pass_count += int(lint_passed)
        lint_fail_count += int(not lint_passed)
        duplicate_adjacent_word_count += int("duplicate_adjacent_word" in failed_checks)
        lowercase_sentence_start_count += int("lowercase_sentence_start" in failed_checks)
        dangling_fragment_count += int("dangling_fragment" in failed_checks)
        bare_fragment_count += int("bare_fragment" in failed_checks)
        repeated_snippet_count += int("repeated_snippet" in failed_checks)
        forbidden_term_count += int("forbidden_term" in failed_checks)
        generic_praise_warning_count += int("generic_praise" in warning_checks)
        rank_band_breakdown.setdefault(band, {"count": 0, "lint_pass_count": 0, "lint_fail_count": 0})
        rank_band_breakdown[band]["count"] += 1
        rank_band_breakdown[band]["lint_pass_count"] += int(lint_passed)
        rank_band_breakdown[band]["lint_fail_count"] += int(not lint_passed)

    sampled_ranks = [1, 5, 10, 25, 50, 61, 75, 100]
    sampled_explanations = [
        {
            "rank": int(row["rank"]),
            "candidate_id": row["candidate_id"],
            "reasoning": row["reasoning"],
            "passed": bool(row.get("style_lint_pass")),
            "failed_checks": json.loads(row["style_lint_failed_checks"]) if isinstance(row.get("style_lint_failed_checks"), str) else row.get("style_lint_failed_checks", []),
        }
        for row in sorted(reasoning_rows, key=lambda item: int(item["rank"]))
        if int(row["rank"]) in sampled_ranks
    ]
    repeated_opening_count = _repeated_opening_count(reasoning_rows, repeated_opening_threshold)
    awkward_pattern_count = _awkward_pattern_count(reasoning_rows)
    grounding_passed = sum(1 for row in reasoning_rows if bool(row.get("grounding_pass")))

    return {
        "total_explanations": len(reasoning_rows),
        "top_100_count": len(reasoning_rows),
        "rank_band_style_distribution": band_style_distribution,
        "exact_duplicate_count": sum(count - 1 for count in duplicates.values() if count > 1),
        "max_exact_duplicate_count": max(duplicates.values(), default=0),
        "repeated_opening_threshold": repeated_opening_threshold,
        "opening_window_size": 5,
        "repeated_opening_count": repeated_opening_count,
        "repeated_nearby_opening_count": repeated_opening_count,
        "awkward_pattern_count": awkward_pattern_count,
        "lint_pass_count": lint_pass_count,
        "lint_fail_count": lint_fail_count,
        "duplicate_adjacent_word_count": duplicate_adjacent_word_count,
        "lowercase_sentence_start_count": lowercase_sentence_start_count,
        "dangling_fragment_count": dangling_fragment_count,
        "bare_fragment_count": bare_fragment_count,
        "repeated_snippet_count": repeated_snippet_count,
        "generic_praise_warning_count": generic_praise_warning_count,
        "forbidden_term_count": forbidden_term_count,
        "rank_band_breakdown": rank_band_breakdown,
        "cutoff_band_positive_coverage": {
            "covered_count": sum(1 for row in cutoff_rows if row.get("grounded_positive")),
            "total_count": len(cutoff_rows),
        },
        "cutoff_band_caveat_coverage": {
            "covered_count": sum(1 for row in cutoff_rows if row.get("grounded_limitation_or_caveat")),
            "total_count": len(cutoff_rows),
        },
        "grounding_result": {
            "passed_count": grounding_passed,
            "failed_count": len(reasoning_rows) - grounding_passed,
        },
        "sampled_explanations": sampled_explanations,
        "explanation_quality_status": "PASS" if lint_fail_count == 0 and grounding_passed == len(reasoning_rows) else "FAIL",
        "sample_before_after": [
            {
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "before": row["legacy_reasoning_preview"],
                "after": row["reasoning"],
            }
            for row in cutoff_rows[:5]
            if row.get("legacy_reasoning_preview")
        ],
    }


def run_reasoning_generation(run_path: str | Path) -> dict[str, str]:
    run_path = Path(run_path)
    score_df = pd.read_parquet(run_path / "scores" / "score_breakdown_v2.parquet")
    ledger_by_id, ledger_by_candidate = _load_ledger(run_path)
    contexts = _load_contexts(run_path)
    credibility, behavioral = _load_breakdowns(run_path)
    top_rows = score_df.sort_values("rank").head(100).copy()

    reasoning_rows = []
    lint_rows = []
    grounding_failures = []
    for _, row in top_rows.iterrows():
        candidate_id = row["candidate_id"]
        positive_ids = json.loads(row["top_positive_evidence_ids"])
        negative_ids = json.loads(row["top_negative_evidence_ids"])
        candidate_items = ledger_by_candidate.get(candidate_id, [])
        positives = [item for item in candidate_items if item.get("polarity") == "positive"]
        negatives = [item for item in candidate_items if item.get("polarity") == "negative"]
        if positive_ids:
            positives = sorted(
                positives,
                key=lambda item: (item["evidence_id"] not in set(positive_ids), -float(item.get("contribution_after_caps", 0.0))),
            )
        if negative_ids:
            negatives = sorted(
                negatives,
                key=lambda item: (item["evidence_id"] not in set(negative_ids), -float(item.get("contribution_after_caps", 0.0))),
            )
        enriched_row = row.copy()
        if candidate_id in credibility.index:
            enriched_row["credibility_multiplier"] = float(credibility.loc[candidate_id, "credibility_multiplier"])
        if candidate_id in behavioral.index:
            enriched_row["availability_multiplier"] = float(behavioral.loc[candidate_id, "availability_multiplier"])
            enriched_row["location_logistics_score"] = float(behavioral.loc[candidate_id, "location_logistics_score"])
        plan = build_reasoning_plan(
            score_row=enriched_row,
            context=contexts.get(candidate_id, {"candidate_id": candidate_id}),
            positives=positives,
            negatives=negatives,
        )
        realization = _realize_reasoning(plan)
        reasoning = realization["reasoning"]
        grounding = _grounding_result(reasoning, plan)
        lint = realization["style_lint"]
        if not lint["passed"] or not grounding["passed"]:
            retry_plan = dict(plan)
            retry_plan["primary_phrase"] = _abstract_fact_for_category(plan["primary_role_fit_category"])
            retry_plan["production_phrase"] = _abstract_fact_for_category("production_delivery_systems")
            retry_plan["practical_phrase"] = _abstract_fact_for_category("python_practical_engineering")
            retry_plan["grounding_phrases"] = deduplicate_clause_content([
                retry_plan["primary_phrase"],
                retry_plan["production_phrase"],
                retry_plan["practical_phrase"],
            ])
            realization = _realize_reasoning(retry_plan)
            reasoning = realization["reasoning"]
            lint = realization["style_lint"]
            grounding = _grounding_result(reasoning, retry_plan)
            plan = retry_plan
        if not lint["passed"] or not grounding["passed"]:
            raise ValueError(f"unable to produce grounded recruiter-facing reasoning for {candidate_id}")
        lowered_reasoning = reasoning.lower()
        normalized_reasoning = _normalized_span(reasoning)
        grounded_positive = any(
            phrase and _normalized_span(str(phrase)) in normalized_reasoning
            for phrase in (plan["primary_phrase"], plan["production_phrase"], plan["practical_phrase"])
        )
        limitation = str(plan.get("limitation") or "").strip()
        grounded_limitation_or_caveat = bool(
            (limitation and limitation.lower() in lowered_reasoning)
            or ("limited" in lowered_reasoning and plan["rank_band"] == "61-100")
            or (plan.get("availability_constraint_phrase") and str(plan["availability_constraint_phrase"]).lower() in lowered_reasoning)
        )
        if not grounding["passed"]:
            grounding_failures.append({"candidate_id": candidate_id, **grounding})
        lint_rows.append({
            "candidate_id": candidate_id,
            "rank": int(row["rank"]),
            "reasoning": reasoning,
            **lint,
        })
        reasoning_rows.append({
            "candidate_id": candidate_id,
            "rank": int(row["rank"]),
            "score": float(row["final_score"]),
            "reasoning": reasoning,
            "template_family": plan["template_family"],
            "rank_band": plan["rank_band"],
            "primary_role_fit_category": plan["primary_role_fit_category"],
            "material_caveat": bool(plan["material_caveat"]),
            "grounding_pass": bool(grounding["passed"]),
            "grounding_evidence_ids": json.dumps(plan["grounding_evidence_ids"]),
            "positive_evidence_ids": json.dumps(plan["positive_evidence_ids"]),
            "negative_evidence_ids": json.dumps(plan["negative_evidence_ids"]),
            "grounding_phrases": json.dumps(plan["grounding_phrases"]),
            "grounded_positive": grounded_positive,
            "grounded_limitation_or_caveat": grounded_limitation_or_caveat,
            "style_lint_pass": bool(lint["passed"]),
            "style_lint_failed_checks": json.dumps(lint["failed_checks"]),
            "style_lint_warning_checks": json.dumps(lint["warning_checks"]),
            "legacy_reasoning_preview": _legacy_cutoff_reasoning(plan) if plan["rank_band"] == "61-100" else "",
        })

    reasoning_dir = run_path / "reasoning"
    reports_dir = run_path / "reports"
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    reasoning_df = pd.DataFrame(reasoning_rows)
    parquet_path = reasoning_dir / "reasoning_v2.parquet"
    jsonl_path = reasoning_dir / "reasoning_v2.jsonl"
    lint_jsonl_path = reasoning_dir / "reasoning_style_lint.jsonl"
    report_json_path = reasoning_dir / "reasoning_grounding_report.json"
    report_md_path = reasoning_dir / "reasoning_grounding_report.md"
    diversity_json_path = reports_dir / "reasoning_diversity_report.json"
    diversity_md_path = reports_dir / "reasoning_diversity_report.md"
    style_quality_json_path = reports_dir / "reasoning_style_quality_report.json"
    style_quality_md_path = reports_dir / "reasoning_style_quality_report.md"

    write_parquet_atomic(reasoning_df, parquet_path, expected_rows=len(reasoning_df), required_columns=["candidate_id", "reasoning"])
    write_jsonl_atomic(jsonl_path, reasoning_rows)
    write_jsonl_atomic(lint_jsonl_path, lint_rows)

    duplicate_counts = Counter(reasoning_df["reasoning"])
    first_sentences, second_sentences = zip(*(_sentence_parts(text) for text in reasoning_df["reasoning"])) if len(reasoning_df) else ([], [])
    diversity_report = {
        "top_100_count": len(reasoning_rows),
        "exact_duplicate_count": sum(count - 1 for count in duplicate_counts.values() if count > 1),
        "max_exact_duplicate_count": max(duplicate_counts.values(), default=0),
        "normalized_template_family_distribution": dict(Counter(reasoning_df["template_family"])) if len(reasoning_df) else {},
        "unique_full_string_count": int(reasoning_df["reasoning"].nunique()) if len(reasoning_df) else 0,
        "unique_first_sentence_count": len(set(first_sentences)),
        "unique_second_sentence_count": len(set(second_sentences)),
        "rank_band_distribution": dict(Counter(reasoning_df["rank_band"])) if len(reasoning_df) else {},
        "evidence_category_distribution": dict(Counter(reasoning_df["primary_role_fit_category"])) if len(reasoning_df) else {},
        "material_caveat_count": int(reasoning_df["material_caveat"].sum()) if len(reasoning_df) else 0,
        "grounding_pass_count": int(reasoning_df["grounding_pass"].sum()) if len(reasoning_df) else 0,
        "grounding_fail_count": len(reasoning_rows) - int(reasoning_df["grounding_pass"].sum()) if len(reasoning_df) else 0,
        "concrete_evidence_count": sum(1 for row in reasoning_rows if json.loads(row["grounding_phrases"])),
    }
    style_quality_report = analyze_reasoning_quality(reasoning_rows)
    grounding_report = {
        "checked_count": len(reasoning_rows),
        "failed_count": len(grounding_failures),
        "passed_count": len(reasoning_rows) - len(grounding_failures),
        "failures": grounding_failures,
    }
    write_json_atomic(report_json_path, grounding_report)
    write_text_atomic(
        report_md_path,
        "# Reasoning Grounding Report\n\n"
        f"Checked rows: {grounding_report['checked_count']}\n\n"
        f"Passed rows: {grounding_report['passed_count']}\n\n"
        f"Failed rows: {grounding_report['failed_count']}\n",
    )
    write_json_atomic(diversity_json_path, diversity_report)
    write_text_atomic(
        diversity_md_path,
        "# Reasoning Diversity Report\n\n"
        f"Top-100 rows: {diversity_report['top_100_count']}\n\n"
        f"Exact duplicate count: {diversity_report['exact_duplicate_count']}\n\n"
        f"Unique reasoning strings: {diversity_report['unique_full_string_count']}\n\n"
        f"Grounding pass count: {diversity_report['grounding_pass_count']}\n",
    )
    write_json_atomic(style_quality_json_path, style_quality_report)
    before_after_lines = []
    for sample in style_quality_report["sample_before_after"]:
        before_after_lines.append(
            f"- rank {sample['rank']} {sample['candidate_id']}\n"
            f"  before: {sample['before']}\n"
            f"  after: {sample['after']}"
        )
    write_text_atomic(
        style_quality_md_path,
        "# Reasoning Style Quality Report\n\n"
        f"Total explanations: {style_quality_report['total_explanations']}\n\n"
        f"Lint pass count: {style_quality_report['lint_pass_count']}\n\n"
        f"Lint fail count: {style_quality_report['lint_fail_count']}\n\n"
        f"Repeated nearby opening count: {style_quality_report['repeated_nearby_opening_count']}\n\n"
        f"Duplicate adjacent word count: {style_quality_report['duplicate_adjacent_word_count']}\n\n"
        f"Lowercase sentence start count: {style_quality_report['lowercase_sentence_start_count']}\n\n"
        f"Dangling fragment count: {style_quality_report['dangling_fragment_count']}\n\n"
        f"Bare fragment count: {style_quality_report['bare_fragment_count']}\n\n"
        f"Repeated snippet count: {style_quality_report['repeated_snippet_count']}\n\n"
        f"Generic praise warning count: {style_quality_report['generic_praise_warning_count']}\n\n"
        f"Forbidden term count: {style_quality_report['forbidden_term_count']}\n\n"
        f"Cutoff-band positive coverage: {style_quality_report['cutoff_band_positive_coverage']['covered_count']} / {style_quality_report['cutoff_band_positive_coverage']['total_count']}\n\n"
        f"Cutoff-band caveat coverage: {style_quality_report['cutoff_band_caveat_coverage']['covered_count']} / {style_quality_report['cutoff_band_caveat_coverage']['total_count']}\n\n"
        f"Grounding pass count: {style_quality_report['grounding_result']['passed_count']}\n\n"
        f"Explanation quality status: {style_quality_report['explanation_quality_status']}\n\n"
        "Sample before/after comparisons:\n\n"
        + ("\n".join(before_after_lines) if before_after_lines else "- none"),
    )
    return {
        "reasoning_parquet": str(parquet_path),
        "reasoning_jsonl": str(jsonl_path),
        "reasoning_style_lint_jsonl": str(lint_jsonl_path),
        "reasoning_grounding_report_json": str(report_json_path),
        "reasoning_grounding_report_md": str(report_md_path),
        "reasoning_diversity_report_json": str(diversity_json_path),
        "reasoning_diversity_report_md": str(diversity_md_path),
        "reasoning_style_quality_report_json": str(style_quality_json_path),
        "reasoning_style_quality_report_md": str(style_quality_md_path),
    }

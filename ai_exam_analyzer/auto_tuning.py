"""Automatic dataset-aware parameter tuning for the UI workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_exam_analyzer.llm_clients import build_llm_client, call_json_schema
from ai_exam_analyzer.model_profiles import QUALITY_PROFILE_LABELS, QUALITY_PROFILE_OPTIONS, get_quality_cost_profile
from ai_exam_analyzer.preprocessing import compute_preprocessing_assessment
from ai_exam_analyzer.cost_tracking import add_records, estimate_tokens_from_text, format_eur, make_cost_record


def _load_context_doc() -> str:
    path = Path(__file__).with_name("auto_tuning_context.md")
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _dataset_profile(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = max(1, len(questions))
    assessments = [compute_preprocessing_assessment(q) for q in questions]
    avg_quality = sum(float((a.get("qualityScore") or 0.0)) for a in assessments) / float(total)
    force_manual = sum(1 for a in assessments if bool((a.get("gates") or {}).get("forceManualReview")))
    blocked_auto = sum(1 for a in assessments if not bool((a.get("gates") or {}).get("allowAutoChange", True)))
    reason_counts: Dict[str, int] = {}
    for a in assessments:
        for reason in (a.get("reasons") or []):
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    return {
        "question_count": len(questions),
        "avg_quality_score": round(avg_quality, 4),
        "force_manual_review_ratio": round(force_manual / float(total), 4),
        "auto_change_blocked_ratio": round(blocked_auto / float(total), 4),
        "top_quality_reasons": [{"reason": k, "count": v, "ratio": round(v / float(total), 4)} for k, v in top_reasons],
    }


def _knowledge_profile(knowledge_base: Optional[Any], questions: List[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
    if knowledge_base is None:
        return {"enabled": False}

    chunks = list(getattr(knowledge_base, "chunks", []) or [])
    images = list(getattr(knowledge_base, "images", []) or [])
    sources = {str(getattr(c, "source", "")) for c in chunks if str(getattr(c, "source", ""))}
    avg_chunk_len = (sum(int(getattr(c, "length", 0) or 0) for c in chunks) / max(1, len(chunks))) if chunks else 0.0

    retrieval_quality_mean = 0.0
    retrieval_nonempty_ratio = 0.0
    if chunks and hasattr(knowledge_base, "retrieve"):
        sample = questions[: min(8, len(questions))]
        qualities: List[float] = []
        nonempty = 0
        for q in sample:
            text = str(q.get("questionText") or "").strip()
            if not text:
                continue
            hits, quality = knowledge_base.retrieve(
                text,
                top_k=int(current.get("knowledge_top_k", 6)),
                min_score=float(current.get("knowledge_min_score", 0.06)),
                max_chars=int(current.get("knowledge_max_chars", 4000)),
            )
            qualities.append(float(quality or 0.0))
            if hits:
                nonempty += 1
        if qualities:
            retrieval_quality_mean = sum(qualities) / len(qualities)
            retrieval_nonempty_ratio = nonempty / len(qualities)

    return {
        "enabled": True,
        "chunk_count": len(chunks),
        "source_count": len(sources),
        "image_count": len(images),
        "avg_chunk_length": round(float(avg_chunk_len), 2),
        "sample_retrieval_quality_mean": round(float(retrieval_quality_mean), 4),
        "sample_retrieval_nonempty_ratio": round(float(retrieval_nonempty_ratio), 4),
    }


def _schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "properties": {
                    "trigger_answer_conf": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "trigger_topic_conf": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "apply_change_min_conf_b": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "low_conf_maintenance_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "knowledge_top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    "knowledge_max_chars": {"type": "integer", "minimum": 500, "maximum": 12000},
                    "knowledge_min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "enable_review_pass": {"type": "boolean"},
                    "enable_reconstruction_pass": {"type": "boolean"},
                    "enable_repeat_reconstruction": {"type": "boolean"},
                },
                "required": [
                    "trigger_answer_conf",
                    "trigger_topic_conf",
                    "apply_change_min_conf_b",
                    "low_conf_maintenance_threshold",
                    "knowledge_top_k",
                    "knowledge_max_chars",
                    "knowledge_min_score",
                    "enable_review_pass",
                    "enable_reconstruction_pass",
                    "enable_repeat_reconstruction",
                ],
                "additionalProperties": False,
            },
            "report_short": {"type": "string"},
            "reasoning": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["settings", "report_short", "reasoning"],
        "additionalProperties": False,
    }


def estimate_analysis_costs(*, provider: str, questions: List[Dict[str, Any]], settings: Dict[str, Any], models: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    del provider
    models = models or {}
    q_count = len(questions)
    sample_text = "\n".join(str(q.get("questionText") or "") for q in questions[: min(20, q_count)])
    avg_question_tokens = max(80, estimate_tokens_from_text(sample_text) // max(1, min(20, q_count)))
    knowledge_tokens = max(0, int(settings.get("knowledge_max_chars", 0) or 0) // 4)
    base_input = avg_question_tokens + knowledge_tokens + 900
    pass_b_ratio = 0.55
    review_ratio = 0.18 if bool(settings.get("enable_review_pass")) else 0.0
    reconstruction_ratio = 1.0 if bool(settings.get("enable_reconstruction_pass")) else 0.0
    explainer_ratio = 1.0 if bool(settings.get("enable_explainer_pass", True)) else 0.0
    records = [
        make_cost_record(stage="base_pass_a", model=models.get("passA") or models.get("pass_a") or "gpt-5.4-nano", input_tokens=q_count * base_input, output_tokens=q_count * 900, estimated=True),
        make_cost_record(stage="base_pass_b_estimated", model=models.get("passB") or models.get("pass_b") or models.get("passA") or "gpt-5.4-nano", input_tokens=int(q_count * pass_b_ratio * (base_input + 700)), output_tokens=int(q_count * pass_b_ratio * 750), estimated=True),
        make_cost_record(stage="review_pass_estimated", model=models.get("review") or models.get("passB") or "gpt-5.4-nano", input_tokens=int(q_count * review_ratio * (base_input + 1200)), output_tokens=int(q_count * review_ratio * 900), estimated=True),
        make_cost_record(stage="reconstruction_pass", model=models.get("reconstruction") or "gpt-5.4-nano", input_tokens=int(q_count * reconstruction_ratio * (base_input + 1000)), output_tokens=int(q_count * reconstruction_ratio * 1000), estimated=True),
        make_cost_record(stage="explainer_pass", model=models.get("explainer") or models.get("passA") or "gpt-5.4-nano", input_tokens=int(q_count * explainer_ratio * (base_input + 800)), output_tokens=int(q_count * explainer_ratio * 1100), estimated=True),
    ]
    summary = add_records(records)
    summary["assumptions"] = {
        "question_count": q_count,
        "avg_question_tokens": avg_question_tokens,
        "knowledge_tokens_per_question": knowledge_tokens,
        "pass_b_run_ratio": pass_b_ratio,
        "review_run_ratio": review_ratio,
        "currency": "EUR",
        "note": "Schätzung auf Zeichen-/Token-Heuristik, statischer Provider-Preistabelle und USD-EUR-Umrechnung; tatsächliche Tokens werden im Lauf aus API-Usage getrackt.",
    }
    return summary


def _models_for_profile(*, provider: str, profile_name: str) -> Dict[str, str]:
    profile = get_quality_cost_profile(provider=provider, profile=profile_name)
    return {
        "passA": profile.pass_a_model,
        "passB": profile.pass_b_model,
        "review": profile.review_model,
        "reconstruction": profile.reconstruction_model,
        "explainer": profile.explainer_model,
    }


def _settings_for_profile(*, provider: str, profile_name: str, current: Dict[str, Any]) -> Dict[str, Any]:
    profile = get_quality_cost_profile(provider=provider, profile=profile_name)
    return {
        **current,
        "trigger_answer_conf": profile.trigger_answer_conf,
        "trigger_topic_conf": profile.trigger_topic_conf,
        "apply_change_min_conf_b": profile.apply_change_min_conf_b,
        "low_conf_maintenance_threshold": profile.low_conf_maintenance_threshold,
        "knowledge_top_k": profile.knowledge_top_k,
        "knowledge_max_chars": profile.knowledge_max_chars,
        "knowledge_min_score": profile.knowledge_min_score,
        "enable_review_pass": profile.enable_review_pass,
        "enable_reconstruction_pass": profile.enable_reconstruction_pass,
        # Keep explainer visible in tuning estimates by default; callers can
        # explicitly set enable_explainer_pass=False to price a run without it.
        "enable_explainer_pass": bool(current.get("enable_explainer_pass", True)),
    }


def estimate_quality_profile_costs(*, provider: str, questions: List[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
    estimates: Dict[str, Any] = {}
    for profile_name in QUALITY_PROFILE_OPTIONS:
        estimate = estimate_analysis_costs(
            provider=provider,
            questions=questions,
            settings=_settings_for_profile(provider=provider, profile_name=profile_name, current=current),
            models=_models_for_profile(provider=provider, profile_name=profile_name),
        )
        estimates[profile_name] = {
            "label": QUALITY_PROFILE_LABELS.get(profile_name, profile_name),
            "models": _models_for_profile(provider=provider, profile_name=profile_name),
            "settings": _settings_for_profile(provider=provider, profile_name=profile_name, current=current),
            "estimate": estimate,
            "total": estimate.get("total") or {},
        }
    return estimates


def recommend_settings(*, provider: str, api_key: str, model: str, topic_tree: Any, questions: List[Dict[str, Any]], current: Dict[str, Any], knowledge_base: Optional[Any] = None, models: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    llm = build_llm_client(provider=provider, api_key=api_key)
    sample = questions[: min(12, len(questions))]
    sample_payload = []
    for q in sample:
        sample_payload.append({
            "questionId": q.get("questionId"),
            "questionText": (q.get("questionText") or "")[:500],
            "answers": [str((a or {}).get("text") or "")[:140] for a in (q.get("answers") or [])[:6]],
            "hasCorrect": bool(q.get("correctAnswers") or q.get("correctAnswerIndices")),
        })

    user = {
        "topic_tree": topic_tree,
        "dataset_stats": _dataset_profile(questions),
        "knowledge_stats": _knowledge_profile(knowledge_base, questions, current),
        "sample_questions": sample_payload,
        "current_settings": current,
    }
    context_doc = _load_context_doc()
    system = (
        "Du optimierst Parameter für einen automatischen Prüfungsfragen-Analyzer. "
        "Wähle robuste Einstellungen für heterogene Datensätze mit Blick auf Qualität vor Aggressivität. "
        "Setze die Datensatzanalyse (dataset_stats + sample_questions) und die Knowledge-Analyse (knowledge_stats) explizit in Relation zur Workflow-Dokumentation. "
        "Optimiere insbesondere knowledge_top_k, knowledge_max_chars, knowledge_min_score passend zur beobachteten Retrieval-Qualität. "
        "Nutze konservative Entscheidungen bei Unsicherheit und begründe parameterweise. "
        "Liefere kurze, konkrete Begründungen.\n\n"
        "=== Workflow-Kontextdokumentation ===\n"
        f"{context_doc}"
    )
    out = call_json_schema(
        llm,
        model=model,
        system=system,
        user=json.dumps(user, ensure_ascii=False),
        schema=_schema(),
        format_name="auto_tuning_recommendation",
        temperature=0.1,
        reasoning_effort="medium",
        max_output_tokens=1200,
    )
    tuning_request_cost = make_cost_record(stage="auto_tuning_request", model=model, usage=out.pop("_llm_usage", None))
    settings = out.get("settings") or {}
    report = str(out.get("report_short") or "").strip()
    reasons = [str(x).strip() for x in (out.get("reasoning") or []) if str(x).strip()]
    if reasons:
        report = (report + "\n\n" if report else "") + "\n".join([f"- {x}" for x in reasons[:6]])
    effective_current = {**current}
    # If the UI did not expose explainer during tuning, still estimate it so the
    # user sees the cost impact of enabling that pass in the breakdown.
    effective_current.setdefault("enable_explainer_pass", True)
    estimate = estimate_analysis_costs(provider=provider, questions=questions, settings={**effective_current, **settings}, models=models or {"passB": model})
    estimate["profileEstimates"] = estimate_quality_profile_costs(provider=provider, questions=questions, current={**effective_current, **settings})
    estimate["tuningRequest"] = tuning_request_cost
    total_cost = float((estimate.get("total") or {}).get("costEur") or 0.0)
    tuning_cost = float(tuning_request_cost.get("costEur") or 0.0)
    report = (report + "\n\n" if report else "") + (
        f"Geschätzte Gesamtkosten der Analyse: {format_eur(total_cost)} (Details in cost_estimate).\n"
        f"Kosten dieser Parameter-Abfrage: {format_eur(tuning_cost)}."
    )
    return settings, report, estimate

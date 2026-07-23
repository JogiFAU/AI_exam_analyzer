"""Automatic dataset-aware parameter tuning for the UI workflow."""

from __future__ import annotations

import json
import math
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
                    "enable_reconstruction_pass": {"type": "boolean", "enum": [False]},
                    "enable_repeat_reconstruction": {"type": "boolean"},
                    "enable_explainer_pass": {"type": "boolean", "enum": [True]},
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
                    "enable_explainer_pass",
                ],
                "additionalProperties": False,
            },
            "report_short": {"type": "string"},
            "reasoning": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["settings", "report_short", "reasoning"],
        "additionalProperties": False,
    }


def _zero_cost_stage(stage: str, *, note: str) -> Dict[str, Any]:
    record = make_cost_record(stage=stage, model="deterministic", input_tokens=0, output_tokens=0, estimated=True)
    record["note"] = note
    return record


def _question_payload_token_stats(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_counts: List[int] = []
    for q in questions:
        # Estimate the payload size closer to the real pass payload than only
        # questionText: answers, ids, correct-answer markers, image references,
        # and existing annotations all add prompt tokens.
        token_counts.append(estimate_tokens_from_text(json.dumps(q, ensure_ascii=False, default=str)))
    if not token_counts:
        token_counts = [80]
    mean = sum(token_counts) / float(len(token_counts))
    variance = sum((x - mean) ** 2 for x in token_counts) / float(len(token_counts))
    return {
        "mean": max(80.0, mean),
        "stddev": math.sqrt(max(0.0, variance)),
        "min": min(token_counts),
        "max": max(token_counts),
    }


def _attach_estimate_stddev(record: Dict[str, Any], *, input_stddev_tokens: float, output_stddev_tokens: float, systematic_ratio: float = 0.30) -> Dict[str, Any]:
    token_variance_cost = make_cost_record(
        stage=record.get("stage") or "unknown",
        model=record.get("model") or "mixed",
        input_tokens=int(max(0.0, input_stddev_tokens)),
        output_tokens=int(max(0.0, output_stddev_tokens)),
        estimated=True,
    )
    token_cost_stddev = float(token_variance_cost.get("costEur") or 0.0)
    systematic_cost_stddev = float(record.get("costEur") or 0.0) * max(0.0, systematic_ratio)
    stddev_eur = math.sqrt((token_cost_stddev ** 2) + (systematic_cost_stddev ** 2))
    record["standardDeviationEur"] = round(stddev_eur, 8)
    record["standardDeviationFormatted"] = format_eur(record["standardDeviationEur"])
    return record


def _attach_summary_stddev(summary: Dict[str, Any]) -> Dict[str, Any]:
    records = list(summary.get("records") or [])
    by_stage = summary.get("byStage") or {}
    total_variance = 0.0
    for stage, bucket in by_stage.items():
        stage_variance = 0.0
        for record in records:
            if str(record.get("stage") or "unknown") == stage:
                stage_variance += float(record.get("standardDeviationEur") or 0.0) ** 2
        stddev = math.sqrt(stage_variance)
        bucket["standardDeviationEur"] = round(stddev, 8)
        bucket["standardDeviationFormatted"] = format_eur(stddev)
        total_variance += stage_variance
    total_stddev = math.sqrt(total_variance)
    total = summary.get("total") or {}
    total["standardDeviationEur"] = round(total_stddev, 8)
    total["standardDeviationFormatted"] = format_eur(total_stddev)
    summary["total"] = total
    return summary


def estimate_analysis_costs(*, provider: str, questions: List[Dict[str, Any]], settings: Dict[str, Any], models: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    del provider
    models = models or {}
    q_count = len(questions)
    token_stats = _question_payload_token_stats(questions)
    avg_question_tokens = int(math.ceil(float(token_stats["mean"])))
    question_stddev_tokens = float(token_stats["stddev"])
    knowledge_tokens = max(0, int(settings.get("knowledge_max_chars", 0) or 0) // 4)

    # The earlier estimate used questionText only and a small fixed overhead,
    # which missed answer payloads, structured-output schemas, topic candidates,
    # system prompts, provider wrappers, and occasional retry/repair context. Use
    # a calibration multiplier so the displayed point estimate is deliberately
    # closer to observed billed totals while the stddev communicates spread.
    prompt_schema_overhead_tokens = 1800
    calibration_multiplier = 1.65
    base_input = int((avg_question_tokens + knowledge_tokens + prompt_schema_overhead_tokens) * calibration_multiplier)

    # Estimate run ratios from the same knobs that decide whether expensive LLM
    # passes run in the processor. This keeps tuning totals aligned with enabled
    # and disabled cost points instead of pricing every optional pass unconditionally.
    pass_b_ratio = max(0.20, min(0.95, 1.0 - float(settings.get("trigger_answer_conf", 0.82) or 0.82) + 0.45))
    review_ratio = 0.0
    if bool(settings.get("enable_review_pass", True)):
        review_ratio = max(0.08, min(0.70, 0.55 - float(settings.get("low_conf_maintenance_threshold", 0.72) or 0.72) * 0.38))
    reconstruction_ratio = 1.0 if bool(settings.get("enable_reconstruction_pass", False)) else 0.0
    explainer_ratio = 1.0 if bool(settings.get("enable_explainer_pass", True)) else 0.0
    cluster_refinement_ratio = 0.30 if bool(settings.get("enable_llm_abstraction_cluster_refinement", True)) else 0.0

    def llm_or_disabled(
        stage: str,
        *,
        enabled: bool,
        model: str,
        run_ratio: float,
        input_per_call: int,
        output_per_call: int,
        disabled_note: str,
        output_stddev_per_call: int,
    ) -> Dict[str, Any]:
        if enabled:
            expected_calls = q_count * max(0.0, run_ratio)
            record = make_cost_record(
                stage=stage,
                model=model,
                input_tokens=int(expected_calls * input_per_call),
                output_tokens=int(expected_calls * output_per_call),
                estimated=True,
            )
            input_stddev = math.sqrt(max(1.0, expected_calls)) * (question_stddev_tokens + input_per_call * 0.18)
            output_stddev = math.sqrt(max(1.0, expected_calls)) * output_stddev_per_call
            record["estimatedCallCount"] = round(expected_calls, 4)
            return _attach_estimate_stddev(record, input_stddev_tokens=input_stddev, output_stddev_tokens=output_stddev)
        record = _zero_cost_stage(stage, note=disabled_note)
        record["standardDeviationEur"] = 0.0
        record["standardDeviationFormatted"] = format_eur(0.0)
        record["estimatedCallCount"] = 0.0
        return record

    records = [
        _zero_cost_stage("initialization_and_loading", note="Datei-/Schema-/Knowledge-Initialisierung; keine LLM-Tokens."),
        _zero_cost_stage("preprocessing_gates", note="Deterministische Qualitäts-/Gate-Prüfung; keine LLM-Tokens."),
        _zero_cost_stage("retrieval_and_context_building", note="Knowledge-Retrieval und Kontextaufbau; keine LLM-Tokens."),
        llm_or_disabled(
            "pass_a",
            enabled=True,
            model=models.get("passA") or models.get("pass_a") or "gpt-5.4-nano",
            run_ratio=1.0,
            input_per_call=base_input,
            output_per_call=1250,
            output_stddev_per_call=450,
            disabled_note="Pass A ist Basisbestandteil und wird nicht deaktiviert.",
        ),
        llm_or_disabled(
            "pass_b",
            enabled=pass_b_ratio > 0,
            model=models.get("passB") or models.get("pass_b") or models.get("passA") or "gpt-5.4-nano",
            run_ratio=pass_b_ratio,
            input_per_call=base_input + 1100,
            output_per_call=1000,
            output_stddev_per_call=400,
            disabled_note="Pass B ist durch die Tuning-Parameter faktisch deaktiviert; keine LLM-Tokens.",
        ),
        llm_or_disabled(
            "review",
            enabled=review_ratio > 0,
            model=models.get("review") or models.get("passB") or "gpt-5.4-nano",
            run_ratio=review_ratio,
            input_per_call=base_input + 1500,
            output_per_call=1100,
            output_stddev_per_call=500,
            disabled_note="Review-Pass ist deaktiviert; keine LLM-Tokens.",
        ),
        _zero_cost_stage("content_and_image_clustering", note="Deterministisches Text-/Bild-Clustering; keine LLM-Tokens."),
        _zero_cost_stage("abstraction_clustering", note="Deterministisches Abstraktions-Clustering; keine LLM-Tokens."),
        llm_or_disabled(
            "abstraction_cluster_refinement",
            enabled=cluster_refinement_ratio > 0,
            model=models.get("cluster_refinement") or models.get("clusterRefinement") or models.get("passA") or "gpt-5.4-nano",
            run_ratio=cluster_refinement_ratio,
            input_per_call=base_input + 1400,
            output_per_call=750,
            output_stddev_per_call=350,
            disabled_note="LLM-Abstraktionscluster-Refinement ist deaktiviert; keine LLM-Tokens.",
        ),
        _zero_cost_stage("repeat_reconstruction", note="Repeat-Reconstruction gleicht Frage-/Antwortmuster deterministisch über Cluster/Jahrgänge ab; keine LLM-Tokens."),
        llm_or_disabled(
            "reconstruction",
            enabled=reconstruction_ratio > 0,
            model=models.get("reconstruction") or "gpt-5.4-nano",
            run_ratio=reconstruction_ratio,
            input_per_call=base_input + 1600,
            output_per_call=1300,
            output_stddev_per_call=550,
            disabled_note="Rekonstruktions-Pass ist deaktiviert; keine LLM-Tokens.",
        ),
        llm_or_disabled(
            "explainer",
            enabled=explainer_ratio > 0,
            model=models.get("explainer") or models.get("passA") or "gpt-5.4-nano",
            run_ratio=explainer_ratio,
            input_per_call=base_input + 1200,
            output_per_call=1500,
            output_stddev_per_call=650,
            disabled_note="Explainer-Pass ist deaktiviert; keine LLM-Tokens.",
        ),
        _zero_cost_stage("output_and_cost_report", note="Ausgabe-/Kostenreport-Schreiben; keine LLM-Tokens."),
    ]
    for record in records:
        record.setdefault("standardDeviationEur", 0.0)
        record.setdefault("standardDeviationFormatted", format_eur(0.0))
    summary = _attach_summary_stddev(add_records(records))
    summary["assumptions"] = {
        "question_count": q_count,
        "avg_question_tokens": avg_question_tokens,
        "question_token_stddev": round(question_stddev_tokens, 4),
        "question_token_min": int(token_stats["min"]),
        "question_token_max": int(token_stats["max"]),
        "knowledge_tokens_per_question": knowledge_tokens,
        "prompt_schema_overhead_tokens": prompt_schema_overhead_tokens,
        "calibration_multiplier": calibration_multiplier,
        "standard_deviation_includes": ["question_payload_spread", "expected_output_spread", "30_percent_systematic_estimation_uncertainty"],
        "pass_b_run_ratio": round(pass_b_ratio, 4),
        "review_run_ratio": round(review_ratio, 4),
        "reconstruction_run_ratio": reconstruction_ratio,
        "explainer_run_ratio": explainer_ratio,
        "cluster_refinement_run_ratio": cluster_refinement_ratio,
        "currency": "EUR",
        "all_cost_points_listed": True,
        "note": "Schätzung listet alle Workflow-Kostenpunkte. Deaktivierte oder deterministische Schritte bleiben als 0-€-Records in der Aufschlüsselung sichtbar. Die Schätzung nutzt vollständige Frage-Payloads statt nur questionText, enthält Prompt-/Schema-Overhead und einen Kalibrierungsfaktor gegen systematische Untererfassung; tatsächliche Tokens werden im Lauf aus API-Usage inklusive Retry-Versuchen getrackt.",
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
        "cluster_refinement": profile.cluster_refinement_model,
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
        "enable_reconstruction_pass": False,
        "enable_llm_abstraction_cluster_refinement": profile.enable_llm_abstraction_cluster_refinement,
        "enable_explainer_pass": profile.enable_explainer_pass,
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
        "Aktiviere den teuren Reconstruction-Pass nicht; der Standard-Workflow nutzt stattdessen den didaktisch wertvollen Explainer-Pass. "
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
    # Auto-Tuning darf den teuren Reconstruction-Pass nicht mehr empfehlen;
    # der Standard-Workflow erzeugt stattdessen didaktische Explainer.
    settings["enable_reconstruction_pass"] = False
    settings["enable_explainer_pass"] = True
    report = str(out.get("report_short") or "").strip()
    reasons = [str(x).strip() for x in (out.get("reasoning") or []) if str(x).strip()]
    if reasons:
        report = (report + "\n\n" if report else "") + "\n".join([f"- {x}" for x in reasons[:6]])
    effective_current = {**current}
    effective_current["enable_reconstruction_pass"] = False
    effective_current["enable_explainer_pass"] = True
    estimate = estimate_analysis_costs(provider=provider, questions=questions, settings={**effective_current, **settings}, models=models or {"passB": model})
    estimate["profileEstimates"] = estimate_quality_profile_costs(provider=provider, questions=questions, current={**effective_current, **settings})
    estimate["tuningRequest"] = tuning_request_cost
    total_payload = estimate.get("total") or {}
    total_cost = float(total_payload.get("costEur") or 0.0)
    total_stddev = float(total_payload.get("standardDeviationEur") or 0.0)
    tuning_cost = float(tuning_request_cost.get("costEur") or 0.0)
    report = (report + "\n\n" if report else "") + (
        f"Geschätzte Gesamtkosten der Analyse: {format_eur(total_cost)} "
        f"± {format_eur(total_stddev)} Standardabweichung (Details in cost_estimate).\n"
        f"Kosten dieser Parameter-Abfrage: {format_eur(tuning_cost)}."
    )
    return settings, report, estimate

"""Automatic dataset-aware parameter tuning for the UI workflow."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ai_exam_analyzer.llm_clients import build_llm_client, call_json_schema


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


def recommend_settings(*, provider: str, api_key: str, model: str, topic_tree: Any, questions: List[Dict[str, Any]], current: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
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
        "dataset_stats": {"question_count": len(questions)},
        "sample_questions": sample_payload,
        "current_settings": current,
    }
    system = (
        "Du optimierst Parameter für einen automatischen Prüfungsfragen-Analyzer. "
        "Wähle robuste Einstellungen für heterogene Datensätze mit Blick auf Qualität vor Aggressivität. "
        "Liefer kurze, konkrete Begründungen."
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
    settings = out.get("settings") or {}
    report = str(out.get("report_short") or "").strip()
    reasons = [str(x).strip() for x in (out.get("reasoning") or []) if str(x).strip()]
    if reasons:
        report = (report + "\n\n" if report else "") + "\n".join([f"- {x}" for x in reasons[:6]])
    return settings, report


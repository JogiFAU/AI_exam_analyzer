"""Deterministic pre-checks and gate decisions for dataset quality signals."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_IMAGE_HINT_RE = re.compile(r"\b(bild|abbildung|grafik|schema|figure)\b", re.IGNORECASE)
_UNCERTAIN_NOTE_RE = re.compile(
    r"\b(irgendwas|vielleicht|kann\s+sich\s+jemand\s+erinnern|unsicher|notiz)\b",
    re.IGNORECASE,
)

# Rule classes for downstream gating.
_HARD_BLOCKERS = {"missing_correct_indices", "invalid_answer_option"}
_CONTEXT_BLOCKERS = {"missing_required_image_asset"}
_SOFT_BLOCKERS = {"insufficient_question_context", "non_exam_question_or_uncertain_source"}


def _question_word_count(question: Dict[str, Any]) -> int:
    text = (question.get("questionText") or "").strip()
    return len([p for p in text.split() if p])


def compute_quality_maintenance_reasons(question: Dict[str, Any]) -> List[str]:
    """Return deterministic maintenance reasons derived from raw data quality issues."""
    reasons: List[str] = []

    answers = question.get("answers") or []
    correct_indices = question.get("correctIndices") or []
    has_images = bool((question.get("imageUrls") or []) or (question.get("imageFiles") or []))

    if not correct_indices:
        reasons.append("missing_correct_indices")

    if any((a.get("text") or "").strip() in {"", "?"} for a in answers):
        reasons.append("invalid_answer_option")

    question_blob = "\n".join([
        str(question.get("questionText") or ""),
        str(question.get("questionHtml") or ""),
    ])
    if _IMAGE_HINT_RE.search(question_blob) and not has_images:
        reasons.append("missing_required_image_asset")

    if _question_word_count(question) <= 3:
        reasons.append("insufficient_question_context")

    source_blob_parts = [question_blob]
    for answer in answers:
        source_blob_parts.append(str(answer.get("text") or ""))
    source_blob = "\n".join(source_blob_parts)
    if _UNCERTAIN_NOTE_RE.search(source_blob):
        reasons.append("non_exam_question_or_uncertain_source")

    # deterministic ordering
    return list(dict.fromkeys(reasons))


def compute_preprocessing_assessment(question: Dict[str, Any]) -> Dict[str, Any]:
    """Compute structured preprocessing assessment and execution gates.

    Returns a dictionary with reasons, classes, quality score and gate decisions.
    """
    reasons = compute_quality_maintenance_reasons(question)

    hard_blockers = [r for r in reasons if r in _HARD_BLOCKERS]
    context_blockers = [r for r in reasons if r in _CONTEXT_BLOCKERS]
    soft_blockers = [r for r in reasons if r in _SOFT_BLOCKERS]

    answers = question.get("answers") or []
    question_text = (question.get("questionText") or "").strip()

    # Extremely malformed entries: skip LLM and mark for manual work.
    run_llm = bool(question_text) and bool(answers)

    # No automatic dataset mutation for hard blockers or missing required image assets.
    allow_auto_change = not bool(hard_blockers or context_blockers)

    # Force manual review when hard/context blockers exist.
    force_manual_review = bool(hard_blockers or context_blockers)

    # Simple transparent quality score.
    penalty = 0.0
    penalty += 0.38 * len(hard_blockers)
    penalty += 0.24 * len(context_blockers)
    penalty += 0.10 * len(soft_blockers)
    quality_score = max(0.0, round(1.0 - min(1.0, penalty), 4))

    return {
        "reasons": reasons,
        "classes": {
            "hardBlockers": hard_blockers,
            "contextBlockers": context_blockers,
            "softBlockers": soft_blockers,
        },
        "gates": {
            "runLlm": run_llm,
            "allowAutoChange": allow_auto_change,
            "forceManualReview": force_manual_review,
        },
        "qualityScore": quality_score,
    }

"""Deterministic pre-checks for dataset quality signals."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_IMAGE_HINT_RE = re.compile(r"\b(bild|abbildung|grafik|schema|figure)\b", re.IGNORECASE)
_UNCERTAIN_NOTE_RE = re.compile(
    r"\b(irgendwas|vielleicht|kann\s+sich\s+jemand\s+erinnern|unsicher|notiz)\b",
    re.IGNORECASE,
)


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


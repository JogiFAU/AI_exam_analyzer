"""Decision policies for confidence, change application, and review triggers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def compose_confidence(
    *,
    answer_conf: float,
    topic_conf: float,
    retrieval_quality: float,
    verifier_agreed: Optional[bool],
    evidence_count: int,
) -> float:
    """Calibrated confidence heuristic with evidence prior."""
    agreement = 1.0 if verifier_agreed is True else (0.45 if verifier_agreed is None else 0.2)
    evidence_prior = 1.0 if evidence_count >= 3 else (0.8 if evidence_count == 2 else (0.55 if evidence_count == 1 else 0.35))
    score = (
        0.34 * float(answer_conf)
        + 0.24 * float(topic_conf)
        + 0.2 * float(retrieval_quality)
        + 0.14 * agreement
        + 0.08 * evidence_prior
    )
    return max(0.0, min(1.0, round(score, 4)))


def should_apply_pass_b_change(
    *,
    current_indices: list[int],
    verified_indices: list[int],
    cannot_judge: bool,
    agree_with_change: bool,
    confidence_b: float,
    apply_min_conf_b: float,
    retrieval_quality: float,
    evidence_count: int,
) -> bool:
    if cannot_judge:
        return False
    if not agree_with_change:
        return False
    if not verified_indices or verified_indices == current_indices:
        return False
    if confidence_b < apply_min_conf_b:
        return False
    if evidence_count <= 0 and retrieval_quality < 0.08:
        return False
    return True


def should_run_review_pass(
    *,
    args: Any,
    maintenance: Dict[str, Any],
    ai_disagrees_with_dataset: bool,
    final_combined_confidence: float,
    pass_a_topic_key: str,
    final_topic_key: str,
) -> bool:
    if not bool(getattr(args, "enable_review_pass", False)):
        return False

    severity = int(maintenance.get("severity", 1))
    needs_maintenance = bool(maintenance.get("needsMaintenance"))
    min_sev = int(getattr(args, "review_min_maintenance_severity", 2))

    if needs_maintenance and severity >= min_sev:
        return True
    if ai_disagrees_with_dataset and final_combined_confidence < 0.85:
        return True
    if pass_a_topic_key != final_topic_key:
        return True
    if final_combined_confidence < max(0.45, float(getattr(args, "low_conf_maintenance_threshold", 0.65)) - 0.1):
        return True
    return False

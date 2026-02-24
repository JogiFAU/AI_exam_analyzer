"""Provider-specific workflow profiles and adaptive heuristics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowProfile:
    provider: str
    retrieval_quality_target: float
    retrieval_retry_top_k_boost: int
    retrieval_retry_char_boost: int
    retrieval_retry_min_score_factor: float
    force_pass_b_when_low_retrieval: bool
    force_pass_b_retrieval_threshold: float


def build_workflow_profile(provider: str) -> WorkflowProfile:
    p = (provider or "openai").strip().lower()
    if p == "gemini":
        return WorkflowProfile(
            provider="gemini",
            retrieval_quality_target=0.38,
            retrieval_retry_top_k_boost=4,
            retrieval_retry_char_boost=2400,
            retrieval_retry_min_score_factor=0.8,
            force_pass_b_when_low_retrieval=True,
            force_pass_b_retrieval_threshold=0.26,
        )

    return WorkflowProfile(
        provider="openai",
        retrieval_quality_target=0.0,
        retrieval_retry_top_k_boost=0,
        retrieval_retry_char_boost=0,
        retrieval_retry_min_score_factor=1.0,
        force_pass_b_when_low_retrieval=False,
        force_pass_b_retrieval_threshold=0.0,
    )

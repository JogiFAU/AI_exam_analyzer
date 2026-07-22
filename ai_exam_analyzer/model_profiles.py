"""Workflow defaults tuned to selected provider/model capabilities and cost preference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


QUALITY_PROFILE_OPTIONS = [
    "highest_quality",
    "quality",
    "cost_optimized",
    "fully_cost_optimized",
]

QUALITY_PROFILE_LABELS: Dict[str, str] = {
    "highest_quality": "Höchste Qualität",
    "quality": "Eher Qualität",
    "cost_optimized": "Eher kostenoptimiert",
    "fully_cost_optimized": "Voll kostenoptimiert",
}


@dataclass(frozen=True)
class WorkflowBudget:
    knowledge_top_k: int
    knowledge_max_chars: int
    knowledge_min_score: float


@dataclass(frozen=True)
class QualityCostProfile:
    pass_a_model: str
    pass_b_model: str
    review_model: str
    reconstruction_model: str
    explainer_model: str
    cluster_refinement_model: str
    pass_a_temperature: float
    pass_b_reasoning_effort: str
    trigger_answer_conf: float
    trigger_topic_conf: float
    apply_change_min_conf_b: float
    low_conf_maintenance_threshold: float
    knowledge_top_k: int
    knowledge_max_chars: int
    knowledge_min_score: float
    enable_review_pass: bool
    enable_reconstruction_pass: bool
    enable_llm_abstraction_cluster_refinement: bool


# Model choices verified against the official model docs/pricing on 2026-07-22:
# - OpenAI: GPT-5.6 Sol (quality), GPT-5.6 Terra (balance), GPT-5.6 Luna and GPT-5.4 nano (cost).
# - Gemini: Gemini 3.1 Pro Preview (quality), Gemini 3.5 Flash (balanced price/performance), Gemini 3.1 Flash-Lite (cost).
_PROVIDER_PROFILES: Dict[str, Dict[str, QualityCostProfile]] = {
    "openai": {
        "highest_quality": QualityCostProfile("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-sol", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-terra", 0.0, "xhigh", 0.86, 0.90, 0.86, 0.72, 10, 8000, 0.04, True, True, True),
        "quality": QualityCostProfile("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-terra", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-luna", 0.0, "high", 0.82, 0.87, 0.82, 0.68, 8, 6000, 0.05, False, True, True),
        "cost_optimized": QualityCostProfile("gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.6-luna", "gpt-5.6-luna", "gpt-5.4-nano", "gpt-5.4-nano", 0.0, "medium", 0.78, 0.83, 0.78, 0.62, 5, 3200, 0.07, False, True, True),
        "fully_cost_optimized": QualityCostProfile("gpt-5.4-nano", "gpt-5.4-nano", "gpt-5.4-nano", "gpt-5.4-nano", "gpt-5.4-nano", "gpt-5.4-nano", 0.0, "low", 0.72, 0.78, 0.74, 0.56, 3, 1800, 0.09, False, False, False),
    },
    "gemini": {
        "highest_quality": QualityCostProfile("gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash", 1.0, "high", 0.88, 0.91, 0.87, 0.74, 10, 9000, 0.035, True, True, True),
        "quality": QualityCostProfile("gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash", 1.0, "medium", 0.84, 0.88, 0.84, 0.70, 8, 6500, 0.05, False, True, True),
        "cost_optimized": QualityCostProfile("gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", 1.0, "medium", 0.80, 0.85, 0.80, 0.65, 5, 3500, 0.07, False, True, True),
        "fully_cost_optimized": QualityCostProfile("gemini-3.1-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", 1.0, "low", 0.74, 0.80, 0.75, 0.58, 3, 2000, 0.09, False, False, False),
    },
}


def normalize_quality_profile(profile: str) -> str:
    normalized = (profile or "quality").strip().lower().replace("-", "_")
    return normalized if normalized in QUALITY_PROFILE_OPTIONS else "quality"


def get_quality_cost_profile(*, provider: str, profile: str) -> QualityCostProfile:
    provider_norm = (provider or "openai").strip().lower()
    if provider_norm not in _PROVIDER_PROFILES:
        provider_norm = "openai"
    return _PROVIDER_PROFILES[provider_norm][normalize_quality_profile(profile)]


def apply_quality_cost_profile(args: Any, *, include_optional_toggles: bool = True) -> Any:
    profile = get_quality_cost_profile(
        provider=str(getattr(args, "llm_provider", "openai")),
        profile=str(getattr(args, "quality_cost_profile", "quality")),
    )
    args.passA_model = profile.pass_a_model
    args.passB_model = profile.pass_b_model
    args.review_model = profile.review_model
    args.reconstruction_model = profile.reconstruction_model
    args.explainer_model = profile.explainer_model
    args.cluster_refinement_model = profile.cluster_refinement_model
    args.passA_temperature = profile.pass_a_temperature
    args.passB_reasoning_effort = profile.pass_b_reasoning_effort
    args.trigger_answer_conf = profile.trigger_answer_conf
    args.trigger_topic_conf = profile.trigger_topic_conf
    args.apply_change_min_conf_b = profile.apply_change_min_conf_b
    args.low_conf_maintenance_threshold = profile.low_conf_maintenance_threshold
    args.knowledge_top_k = profile.knowledge_top_k
    args.knowledge_max_chars = profile.knowledge_max_chars
    args.knowledge_min_score = profile.knowledge_min_score
    if include_optional_toggles:
        args.enable_review_pass = profile.enable_review_pass
        args.enable_reconstruction_pass = profile.enable_reconstruction_pass
        args.enable_llm_abstraction_cluster_refinement = profile.enable_llm_abstraction_cluster_refinement
    return args


def derive_workflow_budget(*, provider: str, pass_a_model: str, pass_b_model: str, default_top_k: int, default_max_chars: int, default_min_score: float) -> WorkflowBudget:
    provider_norm = (provider or "openai").strip().lower()
    model_hint = f"{pass_a_model} {pass_b_model}".lower()
    top_k = int(default_top_k)
    max_chars = int(default_max_chars)
    min_score = float(default_min_score)
    if provider_norm == "gemini":
        top_k = max(top_k, 8)
        max_chars = max(max_chars, 6500)
        min_score = max(0.03, min(min_score, 0.05))
        if "pro" in model_hint:
            top_k = max(top_k, 10)
            max_chars = max(max_chars, 8000)
    else:
        top_k = min(max(1, top_k), 8)
        max_chars = min(max(1200, max_chars), 5000)
    return WorkflowBudget(top_k, max_chars, round(min_score, 4))


def apply_model_optimized_defaults(args: Any) -> Any:
    if hasattr(args, "quality_cost_profile"):
        return apply_quality_cost_profile(args)
    budget = derive_workflow_budget(
        provider=str(getattr(args, "llm_provider", "openai")),
        pass_a_model=str(getattr(args, "passA_model", "")),
        pass_b_model=str(getattr(args, "passB_model", "")),
        default_top_k=int(getattr(args, "knowledge_top_k", 6)),
        default_max_chars=int(getattr(args, "knowledge_max_chars", 4000)),
        default_min_score=float(getattr(args, "knowledge_min_score", 0.06)),
    )
    args.knowledge_top_k = budget.knowledge_top_k
    args.knowledge_max_chars = budget.knowledge_max_chars
    args.knowledge_min_score = budget.knowledge_min_score
    return args

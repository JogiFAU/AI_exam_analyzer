"""Workflow defaults tuned to selected provider/model capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkflowBudget:
    knowledge_top_k: int
    knowledge_max_chars: int
    knowledge_min_score: float


def derive_workflow_budget(*, provider: str, pass_a_model: str, pass_b_model: str, default_top_k: int, default_max_chars: int, default_min_score: float) -> WorkflowBudget:
    provider_norm = (provider or "openai").strip().lower()
    model_hint = f"{pass_a_model} {pass_b_model}".lower()

    top_k = int(default_top_k)
    max_chars = int(default_max_chars)
    min_score = float(default_min_score)

    if provider_norm == "gemini":
        # Gemini-Modelle haben typischerweise große Kontextfenster;
        # wir nutzen mehr Retrieval-Kontext bei leicht reduzierter Score-Schwelle.
        top_k = max(top_k, 8)
        max_chars = max(max_chars, 6500)
        min_score = max(0.03, min(min_score, 0.05))
        if "2.5-pro" in model_hint or "1.5-pro" in model_hint:
            top_k = max(top_k, 10)
            max_chars = max(max_chars, 8000)
    else:
        # OpenAI-Defaults konservativ halten, um Prompt-Größe stabil zu halten.
        top_k = min(max(1, top_k), 8)
        max_chars = min(max(1200, max_chars), 5000)

    return WorkflowBudget(
        knowledge_top_k=top_k,
        knowledge_max_chars=max_chars,
        knowledge_min_score=round(min_score, 4),
    )


def apply_model_optimized_defaults(args: Any) -> Any:
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

"""LLM token/cost accounting helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


# Approximate USD prices per 1M tokens. Keep this table deliberately editable in
# one place; unknown/future models fall back to conservative defaults.
MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5.6-sol": {"input": 12.0, "output": 60.0},
    "gpt-5.6-terra": {"input": 3.0, "output": 12.0},
    "gpt-5.6-luna": {"input": 0.6, "output": 2.4},
    "gpt-5.4-mini": {"input": 0.25, "output": 1.0},
    "gpt-5.4-nano": {"input": 0.05, "output": 0.4},
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
    "gemini-3.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-3.1-flash-lite": {"input": 0.1, "output": 0.4},
}
FALLBACK_PRICING_USD_PER_1M = {"input": 1.0, "output": 5.0}
USD_TO_EUR_RATE = 0.8766  # ECB 2026-07-22: 1 EUR = 1.1408 USD


def normalize_model_name(model: str) -> str:
    return (model or "").strip().lower()


def pricing_for_model(model: str) -> Dict[str, float]:
    return MODEL_PRICING_USD_PER_1M.get(normalize_model_name(model), FALLBACK_PRICING_USD_PER_1M)


def estimate_tokens_from_text(text: Any) -> int:
    return max(1, int(len(str(text or "")) / 4) + 1)


def cost_usd(*, model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    pricing = pricing_for_model(model)
    return round(((max(0, input_tokens) * pricing["input"]) + (max(0, output_tokens) * pricing["output"])) / 1_000_000.0, 8)


def usd_to_eur(amount_usd: float) -> float:
    return round(max(0.0, float(amount_usd or 0.0)) * USD_TO_EUR_RATE, 8)


def format_eur(amount: float) -> str:
    formatted = f"{float(amount or 0.0):,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".") + " €"


def empty_cost_record(*, stage: str, model: str, estimated: bool = False) -> Dict[str, Any]:
    return {
        "stage": stage,
        "model": model,
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "costEur": 0.0,
        "costFormatted": format_eur(0.0),
        "currency": "EUR",
        "estimated": bool(estimated),
    }


def make_cost_record(*, stage: str, model: str, usage: Optional[Dict[str, Any]] = None, input_tokens: int = 0, output_tokens: int = 0, estimated: bool = False) -> Dict[str, Any]:
    usage = usage or {}
    in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or input_tokens or 0)
    out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or output_tokens or 0)
    total = int(usage.get("total_tokens") or (in_tok + out_tok))
    usd = cost_usd(model=model, input_tokens=in_tok, output_tokens=out_tok)
    eur = usd_to_eur(usd)
    return {
        "stage": stage,
        "model": model,
        "inputTokens": in_tok,
        "outputTokens": out_tok,
        "totalTokens": total,
        "costEur": eur,
        "costFormatted": format_eur(eur),
        "currency": "EUR",
        "usdToEurRate": USD_TO_EUR_RATE,
        "estimated": bool(estimated),
    }


def add_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, Any] = {}
    total = empty_cost_record(stage="total", model="mixed")
    for record in records:
        stage = str(record.get("stage") or "unknown")
        by_stage.setdefault(stage, empty_cost_record(stage=stage, model="mixed", estimated=bool(record.get("estimated"))))
        for bucket in (by_stage[stage], total):
            bucket["inputTokens"] += int(record.get("inputTokens") or 0)
            bucket["outputTokens"] += int(record.get("outputTokens") or 0)
            bucket["totalTokens"] += int(record.get("totalTokens") or 0)
            bucket["costEur"] = round(float(bucket.get("costEur") or 0.0) + float(record.get("costEur") or 0.0), 8)
            bucket["costFormatted"] = format_eur(float(bucket.get("costEur") or 0.0))
            bucket["currency"] = "EUR"
            bucket["estimated"] = bool(bucket.get("estimated")) or bool(record.get("estimated"))
    return {"total": total, "byStage": by_stage, "records": list(records)}

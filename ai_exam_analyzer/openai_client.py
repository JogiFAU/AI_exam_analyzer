"""OpenAI API helpers."""

import json
import time
from typing import Any, Dict, List, Optional, Union

from ai_exam_analyzer.cost_tracking import add_usage_records, normalize_usage


def is_reasoning_model(model: str) -> bool:
    """Heuristic: o-series + gpt-5* are treated as reasoning models (may reject temperature/top_p)."""
    m = (model or "").lower().strip()
    return m.startswith("o") or m.startswith("gpt-5")


def _normalize_reasoning_effort(model: str, reasoning_effort: Optional[str]) -> Optional[str]:
    effort = (reasoning_effort or "").strip().lower()
    if not effort:
        return None
    if effort == "xhigh" and not (model or "").lower().strip().startswith("gpt-5"):
        return "high"
    return effort


def call_json_schema(
    client: Any,
    *,
    model: str,
    system: str,
    user: Union[str, List[Dict[str, Any]]],
    schema: Dict[str, Any],
    format_name: str,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: int = 900,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Responses API + Structured Outputs (json_schema) with retry/fallback handling."""

    def _extract_output_text(resp: Any) -> str:
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        output = getattr(resp, "output", None)
        if not isinstance(output, list):
            raise RuntimeError("Responses API returned no parseable text output.")

        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                item = item.model_dump() if hasattr(item, "model_dump") else {}
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    content = content.model_dump() if hasattr(content, "model_dump") else {}
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    parts.append(content["text"])

        merged = "".join(parts).strip()
        if not merged:
            raise RuntimeError("Responses API returned no parseable text output.")
        return merged

    def _is_incomplete_error(message: str) -> bool:
        msg = message.lower()
        return (
            "response not completed: incomplete" in msg
            or "response not completed: in_progress" in msg
            or "response not completed: queued" in msg
        )

    def _parse_json_from_response(resp: Any) -> Dict[str, Any]:
        raw_text = _extract_output_text(resp)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            status = str(getattr(resp, "status", ""))
            raise RuntimeError(f"Invalid JSON payload from response status={status}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Structured output must decode to object, got {type(parsed).__name__}.")
        return parsed

    def _poll_response_until_terminal(resp: Any, *, poll_attempts: int = 4, sleep_s: float = 0.35) -> Any:
        status = str(getattr(resp, "status", ""))
        if status in {"completed", "failed", "incomplete", "cancelled"}:
            return resp

        response_id = getattr(resp, "id", None)
        if not response_id:
            return resp

        for _ in range(max(0, int(poll_attempts))):
            time.sleep(max(0.05, float(sleep_s)))
            try:
                resp = client.responses.retrieve(response_id)
            except Exception:
                return resp
            status = str(getattr(resp, "status", ""))
            if status in {"completed", "failed", "incomplete", "cancelled"}:
                return resp
        return resp

    attempt_usages: List[Dict[str, int]] = []

    def _capture_usage(resp: Any) -> Dict[str, int]:
        usage = getattr(resp, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else {})
        normalized = normalize_usage(usage_dict)
        if normalized["total_tokens"]:
            attempt_usages.append(normalized)
        return normalized

    def _attach_total_usage(parsed: Dict[str, Any], usage: Dict[str, int]) -> Dict[str, Any]:
        parsed["_llm_usage"] = add_usage_records(attempt_usages or [usage])
        parsed["_llm_attempt_count"] = max(1, len(attempt_usages))
        return parsed

    def _single_call(send_temperature: bool, tokens: int) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": format_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": tokens,
        }

        if send_temperature and temperature is not None:
            params["temperature"] = temperature

        normalized_effort = _normalize_reasoning_effort(model, reasoning_effort)
        if is_reasoning_model(model) and normalized_effort:
            params["reasoning"] = {"effort": normalized_effort}

        resp = client.responses.create(**params)
        resp = _poll_response_until_terminal(resp)
        usage = _capture_usage(resp)
        status = str(getattr(resp, "status", ""))
        if status == "completed":
            parsed = _parse_json_from_response(resp)
            return _attach_total_usage(parsed, usage)

        # Some providers occasionally mark responses as incomplete even though
        # the structured JSON is already parseable. Use it when possible.
        try:
            parsed = _parse_json_from_response(resp)
            return _attach_total_usage(parsed, usage)
        except Exception:
            pass

        if status != "completed":
            details = getattr(resp, "incomplete_details", None)
            reason = None
            if isinstance(details, dict):
                reason = details.get("reason")
            elif hasattr(details, "model_dump"):
                reason = details.model_dump().get("reason")
            suffix = f" (reason={reason})" if reason else ""
            raise RuntimeError(f"Response not completed: {status}{suffix}")
        return _parse_json_from_response(resp)

    def _call_with_retries(send_temperature: bool) -> Dict[str, Any]:
        current_tokens = max(256, int(max_output_tokens))
        last_error: Optional[Exception] = None

        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                return _single_call(send_temperature=send_temperature, tokens=current_tokens)
            except Exception as exc:  # keep broad: API/network/serialization variants
                msg = str(exc)
                last_error = exc

                is_retryable = (
                    _is_incomplete_error(msg)
                    or "timed out" in msg.lower()
                    or "rate limit" in msg.lower()
                    or "temporarily unavailable" in msg.lower()
                    or "connection" in msg.lower()
                )
                if attempt >= max_retries or not is_retryable:
                    break

                # for incomplete outputs, raise available output budget on retry
                if _is_incomplete_error(msg):
                    current_tokens = min(4000, int(current_tokens * 1.6) + 128)
                time.sleep(0.45 * (attempt + 1))

        if last_error is None:
            raise RuntimeError("Unknown Responses API failure.")
        raise last_error

    if is_reasoning_model(model):
        return _call_with_retries(send_temperature=False)

    try:
        return _call_with_retries(send_temperature=True)
    except Exception as e:
        msg = str(e)
        if "temperature" in msg and ("Unsupported parameter" in msg or "not supported" in msg):
            return _call_with_retries(send_temperature=False)
        raise

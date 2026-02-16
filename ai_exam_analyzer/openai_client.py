"""OpenAI API helpers."""

import json
from typing import Any, Dict, List, Optional, Union


def is_reasoning_model(model: str) -> bool:
    """Heuristic: o-series + gpt-5* are treated as reasoning models (may reject temperature/top_p)."""
    m = (model or "").lower().strip()
    return m.startswith("o") or m.startswith("gpt-5")


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
) -> Dict[str, Any]:
    """Responses API + Structured Outputs (json_schema). Handles temperature for reasoning models."""

    def _do_call(send_temperature: bool) -> Dict[str, Any]:
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
            "max_output_tokens": max_output_tokens,
        }

        if send_temperature and temperature is not None:
            params["temperature"] = temperature

        if is_reasoning_model(model) and reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        resp = client.responses.create(**params)
        if resp.status != "completed":
            raise RuntimeError(f"Response not completed: {resp.status}")
        return json.loads(resp.output_text)

    if is_reasoning_model(model):
        return _do_call(send_temperature=False)

    try:
        return _do_call(send_temperature=True)
    except Exception as e:
        msg = str(e)
        if "temperature" in msg and ("Unsupported parameter" in msg or "not supported" in msg):
            return _do_call(send_temperature=False)
        raise

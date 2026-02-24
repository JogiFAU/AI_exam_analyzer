"""Provider-neutral LLM client adapters."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ai_exam_analyzer.openai_client import call_json_schema as _openai_call_json_schema


@dataclass
class LLMClient:
    provider: str
    client: Any


def _extract_json_object(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return text
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


class GeminiResponsesAdapter:
    """Small adapter for Gemini via google-genai SDK."""

    def __init__(self, api_key: str):
        try:
            from google import genai  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency for Gemini: install `google-genai` package.") from exc
        self._sdk = genai.Client(api_key=api_key)

    def call_json_schema(
        self,
        *,
        model: str,
        system: str,
        user: Union[str, List[Dict[str, Any]]],
        schema: Dict[str, Any],
        format_name: str,
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        max_output_tokens: int,
        max_retries: int,
    ) -> Dict[str, Any]:
        del format_name, reasoning_effort

        contents: List[Dict[str, Any]] = []
        contents.append({"role": "user", "parts": [{"text": system}]})
        if isinstance(user, str):
            contents.append({"role": "user", "parts": [{"text": user}]})
        else:
            parts: List[Dict[str, Any]] = []
            for item in user:
                kind = str(item.get("type") or "")
                if kind == "input_text":
                    parts.append({"text": str(item.get("text") or "")})
                    continue
                if kind == "input_image":
                    image_url = str(item.get("image_url") or "")
                    if image_url.startswith("data:") and ";base64," in image_url:
                        header, b64 = image_url.split(",", 1)
                        mime_type = header[5:].split(";", 1)[0] or "image/png"
                        try:
                            raw = base64.b64decode(b64)
                            parts.append({"inline_data": {"mime_type": mime_type, "data": raw}})
                        except Exception:
                            continue
            if parts:
                contents.append({"role": "user", "parts": parts})

        config: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "max_output_tokens": int(max(256, max_output_tokens)),
        }
        if temperature is not None:
            config["temperature"] = float(temperature)

        last_error: Optional[Exception] = None
        for _ in range(max(0, int(max_retries)) + 1):
            try:
                resp = self._sdk.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                text = str(getattr(resp, "text", "") or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned empty response text.")
                parsed = json.loads(_extract_json_object(text))
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"Gemini structured output must decode to object, got {type(parsed).__name__}.")
                return parsed
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("Unknown Gemini API failure.")
        raise last_error


def build_llm_client(*, provider: str, api_key: Optional[str] = None) -> LLMClient:
    normalized = (provider or "openai").strip().lower()
    if normalized not in {"openai", "gemini"}:
        raise ValueError(f"Unsupported --llm-provider '{provider}'. Supported: openai|gemini")

    if normalized == "openai":
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: install `openai` package (e.g. `pip install openai`).") from exc
        return LLMClient(provider="openai", client=OpenAI(api_key=api_key))

    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return LLMClient(provider="gemini", client=GeminiResponsesAdapter(api_key=gemini_key))


def call_json_schema(
    llm: LLMClient,
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
    if llm.provider == "openai":
        return _openai_call_json_schema(
            llm.client,
            model=model,
            system=system,
            user=user,
            schema=schema,
            format_name=format_name,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )

    return llm.client.call_json_schema(
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name=format_name,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
    )

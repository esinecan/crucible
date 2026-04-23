"""Unified DeepSeek LLM client.

Replaces five httpx duplicates scattered across insight/engine.py,
insight/evaluator.py, reasoning/actions.py, extraction/extractor.py, and
extraction/ontology.py. Modeled on ~/dev5/burokrat/src/burokrat/llm/client.py
but stays on raw httpx (no OpenAI SDK dep) so Crucible keeps its current
dependency footprint.

Design:
- One `LLMClient` per `Config`. Callers pass per-call temperature / max_tokens
  — the shared client owns API credentials and endpoint, not per-site tuning.
- `chat()` returns raw text. `chat_json()` sets response_format=json_object
  (DeepSeek honors this) and falls back to lenient parsing if the server or
  model doesn't respect the flag.
- No retry here; the client raises standard httpx exceptions. Callers that
  need retry (extractor) wrap their own policy around the call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from .config import Config

logger = logging.getLogger(__name__)


def parse_json_lenient(text: str) -> Any:
    """Extract JSON from LLM output tolerant of markdown fences and preamble.

    Lifted from three duplicate copies in engine/extractor/ontology; exposed
    here for callers that don't know if the model honored response_format.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


class LLMClient:
    """DeepSeek chat client. One instance per Config."""

    def __init__(self, config: Config, timeout: float = 60.0):
        self.config = config
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    def _endpoint(self) -> str:
        return f"{self.config.deepseek_base_url}/v1/chat/completions"

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Single chat completion. Returns response text.

        Raises httpx exceptions unchanged — no retry. Wrap in a retry loop at
        the caller if transient failures need to be tolerated.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat_with_messages(
            messages, temperature=temperature, max_tokens=max_tokens,
        )

    def chat_with_messages(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: dict | None = None,
    ) -> str:
        body = {
            "model": self.config.eval_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format
        resp = httpx.post(
            self._endpoint(),
            headers=self._headers,
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> Any:
        """Chat with JSON response mode, lenient fallback on parse failure.

        Returns parsed dict/list, or None if nothing parseable came back.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = self.chat_with_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except httpx.HTTPStatusError as e:
            # Model may not support response_format; retry without it.
            if e.response.status_code == 400:
                logger.debug("response_format rejected, retrying without JSON mode")
                text = self.chat_with_messages(
                    messages, temperature=temperature, max_tokens=max_tokens,
                )
            else:
                raise
        return parse_json_lenient(text)

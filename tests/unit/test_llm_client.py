"""Unit tests for the unified LLMClient.

Network is stubbed via httpx MockTransport — the real DeepSeek endpoint is
never hit. We verify request shape (model, messages, temperature, max_tokens,
response_format) and parse behavior (chat_json fallback to lenient parsing).
"""
from __future__ import annotations

import json

import httpx
import pytest

from crucible.config import Config
from crucible.llm_client import LLMClient, parse_json_lenient


def _config(api_key: str = "test-key", model: str = "test-model") -> Config:
    return Config(
        deepseek_api_key=api_key,
        deepseek_base_url="https://api.example.test",
        eval_model=model,
    )


def _stub_response(monkeypatch, payload: str | dict, *, status: int = 200, capture: list | None = None):
    """Patch httpx.post to return a canned chat-completions response."""
    body = (
        payload if isinstance(payload, str) else json.dumps(payload)
    )

    def _fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        if capture is not None:
            capture.append({"url": url, "headers": headers, "json": json})
        if status >= 400:
            request = httpx.Request("POST", url)
            response = httpx.Response(status, request=request, content=b"err")
            raise httpx.HTTPStatusError("err", request=request, response=response)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": body}}]},
            request=httpx.Request("POST", url),
        )

    import crucible.llm_client as mod
    monkeypatch.setattr(mod.httpx, "post", _fake_post)


class TestParseJsonLenient:
    def test_plain_object(self):
        assert parse_json_lenient('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        assert parse_json_lenient('text\n```json\n{"x":2}\n```') == {"x": 2}

    def test_garbage_returns_none(self):
        assert parse_json_lenient("not json at all") is None


class TestLLMClient:
    def test_chat_returns_content(self, monkeypatch):
        _stub_response(monkeypatch, "the answer")
        client = LLMClient(_config())
        out = client.chat("you are X", "hello", temperature=0.5, max_tokens=200)
        assert out == "the answer"

    def test_chat_passes_request_shape(self, monkeypatch):
        captured: list = []
        _stub_response(monkeypatch, "ok", capture=captured)
        client = LLMClient(_config(model="my-model"))
        client.chat("sys", "usr", temperature=0.3, max_tokens=400)
        body = captured[0]["json"]
        assert body["model"] == "my-model"
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 400
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        # response_format absent for plain chat()
        assert "response_format" not in body
        # Auth header carries the configured key
        assert captured[0]["headers"]["Authorization"] == "Bearer test-key"

    def test_chat_json_sets_response_format(self, monkeypatch):
        captured: list = []
        _stub_response(monkeypatch, '{"x": 1}', capture=captured)
        client = LLMClient(_config())
        result = client.chat_json("sys", "usr")
        assert result == {"x": 1}
        assert captured[0]["json"]["response_format"] == {"type": "json_object"}

    def test_chat_json_falls_back_to_lenient(self, monkeypatch):
        # Server returns markdown-fenced JSON, not raw — lenient parser saves it
        _stub_response(monkeypatch, "prose\n```json\n{\"y\": 2}\n```")
        client = LLMClient(_config())
        result = client.chat_json("sys", "usr")
        assert result == {"y": 2}

    def test_chat_json_retries_without_response_format_on_400(self, monkeypatch):
        """Some DeepSeek deployments reject response_format. Client retries
        without it instead of failing."""
        captured: list = []
        first_call = {"done": False}

        def _fake_post(url, headers=None, json=None, timeout=None, **kwargs):
            captured.append({"json": json})
            if not first_call["done"] and json.get("response_format"):
                first_call["done"] = True
                request = httpx.Request("POST", url)
                response = httpx.Response(400, request=request, content=b"bad")
                raise httpx.HTTPStatusError("bad", request=request, response=response)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"z": 3}'}}]},
                request=httpx.Request("POST", url),
            )

        import crucible.llm_client as mod
        monkeypatch.setattr(mod.httpx, "post", _fake_post)
        client = LLMClient(_config())
        result = client.chat_json("sys", "usr")
        assert result == {"z": 3}
        assert len(captured) == 2
        assert captured[0]["json"].get("response_format") is not None
        assert captured[1]["json"].get("response_format") is None

    def test_chat_propagates_non_400_errors(self, monkeypatch):
        _stub_response(monkeypatch, "boom", status=500)
        client = LLMClient(_config())
        with pytest.raises(httpx.HTTPStatusError):
            client.chat("sys", "usr")

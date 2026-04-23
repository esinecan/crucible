"""Tests for extractor error classification: transient vs permanent.

Transient errors must NOT poison the chunk (no mark_chunk_extracted).
Permanent errors mark it so we don't loop forever on bad input.
"""
from __future__ import annotations

import httpx
import pytest

from crucible.extraction.extractor import (
    TransientExtractionError,
    _is_transient,
)


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(status_code, request=req)
    return httpx.HTTPStatusError("boom", request=req, response=resp)


class TestIsTransient:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable_http_codes(self, code):
        assert _is_transient(_status_error(code))

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_permanent_http_codes(self, code):
        assert not _is_transient(_status_error(code))

    def test_network_errors(self):
        req = httpx.Request("POST", "http://x")
        assert _is_transient(httpx.TimeoutException("slow", request=req))
        assert _is_transient(httpx.ConnectError("unreachable", request=req))
        assert _is_transient(httpx.ReadError("broken pipe", request=req))

    def test_transient_marker_class(self):
        assert _is_transient(TransientExtractionError("flag"))

    def test_plain_exception_is_permanent(self):
        assert not _is_transient(ValueError("bad json"))
        assert not _is_transient(KeyError("missing field"))
        assert not _is_transient(RuntimeError("?"))

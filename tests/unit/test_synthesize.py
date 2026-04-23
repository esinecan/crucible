"""Unit tests for EntitySynthesizer.

Stubs LLMClient + the graph layer so the merge logic, fallback behavior, and
budget-cap accounting are testable without Neo4j or DeepSeek.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crucible.config import Config
from crucible.extraction.synthesize import EntitySynthesizer, _build_user_prompt


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat_json(self, system, user, *, temperature, max_tokens):
        self.calls += 1
        if not self._responses:
            return None
        return self._responses.pop(0)


class _FakeGraph:
    """In-memory graph stand-in covering the four methods the synthesizer
    calls: get_dirty_entities, get_mentions_for_entity, update_entity_synthesis,
    and the bulk embedding pair.
    """

    def __init__(self, dirty, mentions_by_id, names_by_id):
        self._dirty = list(dirty)
        self._mentions = mentions_by_id
        self._names = names_by_id
        self.updates: list[tuple[str, str, list[str], str]] = []
        self.embeddings: list[list[tuple[str, list[float]]]] = []

    def get_dirty_entities(self, corpus_id, limit=0):
        if limit and limit > 0:
            return list(self._dirty[:limit])
        return list(self._dirty)

    def get_mentions_for_entity(self, entity_id):
        return list(self._mentions.get(entity_id, []))

    def update_entity_synthesis(self, entity_id, description, aliases, synced):
        self.updates.append((entity_id, description, aliases, synced))

    def set_entity_embeddings(self, updates):
        self.embeddings.append(list(updates))

    # _embed_in_batches reaches in to read names; emulate the cypher.
    @property
    def _driver(self):
        names = self._names

        class _Session:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def run(self_inner, query, **params):
                return [
                    {"id": eid, "name": names.get(eid, "")}
                    for eid in params.get("ids", [])
                    if eid in names
                ]

        class _Driver:
            def session(self_inner):
                return _Session()

        return _Driver()


def _config() -> Config:
    return Config(
        deepseek_api_key="test",
        deepseek_base_url="https://example.test",
        max_synthesis_calls=500,
    )


def _ent(eid="e1", name="Alice", etype="PERSON"):
    return {"id": eid, "name": name, "entity_type": etype}


def _m(chunk_id, name_as_extracted="Alice", description="", aliases=None):
    return {
        "id": f"m-{chunk_id}",
        "chunk_id": chunk_id,
        "name_as_extracted": name_as_extracted,
        "description": description,
        "aliases": aliases or [],
        "confidence": 1.0,
    }


class TestPromptBuilder:
    def test_includes_entity_header_and_mentions(self):
        out = _build_user_prompt(
            "Alice", "PERSON",
            [_m("ch-1", description="a", aliases=["Al"]),
             _m("ch-2", description="b")],
        )
        assert 'Entity: "Alice" (PERSON)' in out
        assert "ch-1" not in out  # we don't leak chunk ids; only surface forms
        assert "description: a" in out
        assert "description: b" in out
        assert "aliases: Al" in out


class TestSynthesizeDirty:
    def test_no_dirty_entities_returns_zero_stats(self, monkeypatch):
        graph = _FakeGraph(dirty=[], mentions_by_id={}, names_by_id={})
        # Stub embed_batch so any accidental call would not hit the network.
        from crucible.extraction import synthesize as _syn
        monkeypatch.setattr(_syn, "embed_batch", lambda *a, **k: [])
        synth = EntitySynthesizer(_config(), graph, llm_client=_FakeLLM([]))
        stats = synth.synthesize_dirty("corp")
        assert stats == {"dirty_total": 0, "synthesized": 0,
                         "errors": 0, "embedded": 0, "remaining": 0}

    def test_happy_path_writes_updates_and_embeds(self, monkeypatch):
        ent = _ent("e1", "Alice")
        graph = _FakeGraph(
            dirty=[ent],
            mentions_by_id={"e1": [
                _m("ch-1", description="short"),
                _m("ch-2", description="longer description here", aliases=["Al"]),
            ]},
            names_by_id={"e1": "Alice"},
        )
        llm = _FakeLLM([{
            "description": "Canonical Alice — refined.",
            "aliases": ["Al", "Ally"],
        }])
        from crucible.extraction import synthesize as _syn
        embed_calls: list = []

        def _fake_embed(texts, config, batch_size=20):
            embed_calls.append(list(texts))
            return [[0.0] * config.embed_dim for _ in texts]

        monkeypatch.setattr(_syn, "embed_batch", _fake_embed)

        synth = EntitySynthesizer(_config(), graph, llm_client=llm)
        stats = synth.synthesize_dirty("corp")

        assert stats["dirty_total"] == 1
        assert stats["synthesized"] == 1
        assert stats["embedded"] == 1
        assert llm.calls == 1
        eid, desc, aliases, synced = graph.updates[0]
        assert eid == "e1"
        assert desc == "Canonical Alice — refined."
        assert aliases == ["Al", "Ally"]
        # Synced timestamp parses as ISO8601
        datetime.fromisoformat(synced.replace("Z", "+00:00"))
        # Embedded text is "name: description"
        assert embed_calls == [["Alice: Canonical Alice — refined."]]

    def test_falls_back_to_bootstrap_on_empty_llm_response(self, monkeypatch):
        ent = _ent("e1", "Bob")
        mentions = [
            _m("ch-1", description="short", aliases=["B"]),
            _m("ch-2", description="this is the longest description here", aliases=["Bobby"]),
        ]
        graph = _FakeGraph(
            dirty=[ent], mentions_by_id={"e1": mentions},
            names_by_id={"e1": "Bob"},
        )
        from crucible.extraction import synthesize as _syn
        monkeypatch.setattr(
            _syn, "embed_batch",
            lambda texts, config, batch_size=20: [[0.0] for _ in texts],
        )

        synth = EntitySynthesizer(_config(), graph, llm_client=_FakeLLM([None]))
        stats = synth.synthesize_dirty("corp")
        assert stats["synthesized"] == 1
        _, desc, aliases, _ = graph.updates[0]
        # Longest mention's description wins as bootstrap.
        assert desc == "this is the longest description here"
        # Aliases unioned across mentions, dedup'd.
        assert sorted(aliases) == ["B", "Bobby"]

    def test_budget_cap_short_circuits(self, monkeypatch):
        dirty = [_ent(f"e{i}", f"E{i}") for i in range(5)]
        mentions = {ent["id"]: [_m("ch-1", description=f"d{ent['id']}")]
                    for ent in dirty}
        graph = _FakeGraph(
            dirty=dirty, mentions_by_id=mentions,
            names_by_id={ent["id"]: ent["name"] for ent in dirty},
        )
        # LLM will only be called twice — budget enforces the cap.
        responses = [
            {"description": "syn1", "aliases": []},
            {"description": "syn2", "aliases": []},
            {"description": "syn3", "aliases": []},  # never reached
        ]
        from crucible.extraction import synthesize as _syn
        monkeypatch.setattr(
            _syn, "embed_batch",
            lambda texts, config, batch_size=20: [[0.0] for _ in texts],
        )

        llm = _FakeLLM(responses)
        synth = EntitySynthesizer(_config(), graph, llm_client=llm)
        stats = synth.synthesize_dirty("corp", max_calls=2)

        assert stats["dirty_total"] == 5
        assert stats["synthesized"] == 2
        assert stats["remaining"] == 3
        assert llm.calls == 2

    def test_per_entity_llm_error_counts_but_continues(self, monkeypatch):
        dirty = [_ent("e1"), _ent("e2", "Bob")]
        graph = _FakeGraph(
            dirty=dirty,
            mentions_by_id={
                "e1": [_m("ch-1", description="a")],
                "e2": [_m("ch-1", description="b")],
            },
            names_by_id={"e1": "Alice", "e2": "Bob"},
        )

        class _Boom(_FakeLLM):
            def chat_json(self_inner, *args, **kwargs):
                self_inner.calls += 1
                if self_inner.calls == 1:
                    raise RuntimeError("network blip")
                return {"description": "ok", "aliases": []}

        from crucible.extraction import synthesize as _syn
        monkeypatch.setattr(
            _syn, "embed_batch",
            lambda texts, config, batch_size=20: [[0.0] for _ in texts],
        )

        synth = EntitySynthesizer(_config(), graph, llm_client=_Boom([]))
        stats = synth.synthesize_dirty("corp")
        assert stats["errors"] == 1
        assert stats["synthesized"] == 1  # second entity still processed
        assert len(graph.updates) == 1

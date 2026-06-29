"""Tests for the episodic-memory subscriber (MemoryExtractor).

Tier-0 (mechanical, always) and Tier-1 (Judge, opt-in) write paths, and the
judge-failure → Tier-0 fallback.  The KB is faked so these stay unit tests with
no embedding model.
"""

import numpy as np

from dsagt.memory import SESSION_MEMORY_COLLECTION, MemoryExtractor
from dsagt.traces import Trace


class FakeKB:
    def __init__(self):
        self.calls = []

    def add_entries(self, *, texts, collection, metadatas, return_embeddings=False):
        self.calls.append(
            {"texts": texts, "collection": collection, "metadatas": metadatas}
        )
        if return_embeddings:
            return {"embeddings": np.zeros((len(texts), 3), dtype=np.float32)}
        return {}


def _one_turn_trace():
    """A trace whose ``to_exchanges`` yields one exchange."""
    trace = Trace("t", "proj:s", "claude", "proj")
    trace.add_agent_root("r1", "conv", start_time=1.0, prompt="filter the reads")
    trace.add_llm_span(
        "r1-0",
        parent_id="r1",
        start_time=1.0,
        end_time=None,
        request=[
            {"role": "user", "content": [{"type": "text", "text": "filter the reads"}]}
        ],
        response=[{"type": "text", "text": "kept 90 percent after QC"}],
    )
    return trace


def test_tier0_indexes_turns_when_no_judge(tmp_path):
    kb = FakeKB()
    ext = MemoryExtractor(kb, runtime_dir=tmp_path, session_id="proj:s")
    ext.write(_one_turn_trace())

    assert len(kb.calls) == 1
    call = kb.calls[0]
    assert call["collection"] == SESSION_MEMORY_COLLECTION
    assert call["metadatas"][0]["source_type"] == "turn"
    assert call["metadatas"][0]["tier"] == "0"
    assert isinstance(call["metadatas"][0]["ts_epoch"], float)  # for recency
    assert "filter the reads" in call["texts"][0]


def test_tier1_stores_distilled_facts(tmp_path):
    class StubJudge:
        backend = "local"

        def distill(self, exchanges, tags):
            return [{"text": "QC pass rate was 90%", "tag": "quality_control"}]

    kb = FakeKB()
    ext = MemoryExtractor(
        kb, runtime_dir=tmp_path, session_id="proj:s", judge=StubJudge()
    )
    ext.write(_one_turn_trace())

    assert len(kb.calls) == 1
    call = kb.calls[0]
    assert call["texts"] == ["QC pass rate was 90%"]
    assert call["metadatas"][0]["source_type"] == "fact"
    assert call["metadatas"][0]["category"] == "quality_control"
    assert call["metadatas"][0]["tier"] == "1"
    assert isinstance(call["metadatas"][0]["ts_epoch"], float)  # for recency


def test_judge_failure_falls_back_to_tier0(tmp_path):
    class BoomJudge:
        backend = "local"

        def distill(self, exchanges, tags):
            raise RuntimeError("model OOM")

    kb = FakeKB()
    ext = MemoryExtractor(
        kb, runtime_dir=tmp_path, session_id="proj:s", judge=BoomJudge()
    )
    ext.write(_one_turn_trace())

    # Degraded to Tier-0 — data preserved, never blocked.
    assert len(kb.calls) == 1
    assert kb.calls[0]["metadatas"][0]["tier"] == "0"


def test_empty_trace_writes_nothing(tmp_path):
    kb = FakeKB()
    ext = MemoryExtractor(kb, runtime_dir=tmp_path, session_id="proj:s")
    ext.write(Trace("t", "proj:s", "claude", "proj"))
    assert kb.calls == []

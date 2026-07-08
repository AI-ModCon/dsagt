"""Tests for the episodic-memory subscriber (MemoryExtractor).

The mechanical write path: per-block chunking with producer/tool/turn_id
metadata.  The KB is faked so these stay unit tests with no embedding model.
"""

from dsagt.memory import SESSION_MEMORY_COLLECTION, MemoryExtractor
from dsagt.traces import Trace


class FakeKB:
    def __init__(self):
        self.calls = []

    def add_entries(self, *, texts, collection, metadatas):
        self.calls.append(
            {"texts": texts, "collection": collection, "metadatas": metadatas}
        )
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


def _tool_trace():
    """A two-turn trace: the assistant calls a tool, the result returns next turn."""
    trace = Trace("t", "proj:s", "claude", "proj")
    trace.add_agent_root("r1", "conv", start_time=1.0, prompt="profile it")
    trace.add_llm_span(
        "r1-0",
        parent_id="r1",
        start_time=1.0,
        end_time=None,
        request=[
            {"role": "user", "content": [{"type": "text", "text": "profile sales.csv"}]}
        ],
        response=[{"type": "tool_use", "id": "t1", "name": "profile", "input": {}}],
    )
    trace.add_llm_span(
        "r1-1",
        parent_id="r1",
        start_time=2.0,
        end_time=None,
        request=[
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "1000 rows"}
                ],
            }
        ],
        response=[{"type": "text", "text": "The file has 1000 rows."}],
    )
    return trace


def test_each_block_is_its_own_chunk_with_producer(tmp_path):
    kb = FakeKB()
    ext = MemoryExtractor(kb, runtime_dir=tmp_path, session_id="proj:s")
    ext.write(_one_turn_trace())

    assert len(kb.calls) == 1
    call = kb.calls[0]
    assert call["collection"] == SESSION_MEMORY_COLLECTION
    # The user question and the assistant answer are separate chunks/embeddings.
    assert call["texts"] == ["filter the reads", "kept 90 percent after QC"]
    metas = call["metadatas"]
    assert [m["producer"] for m in metas] == ["user", "llm"]
    assert all(m["turn_id"] == "r1-0" for m in metas)
    assert all(m["source_type"] == "turn" for m in metas)
    assert all(isinstance(m["ts_epoch"], float) for m in metas)  # for recency


def test_tool_result_chunk_carries_producer_and_resolved_tool(tmp_path):
    kb = FakeKB()
    ext = MemoryExtractor(kb, runtime_dir=tmp_path, session_id="proj:s")
    ext.write(_tool_trace())

    texts = kb.calls[0]["texts"]
    metas = kb.calls[0]["metadatas"]
    # tool_use is not embedded as prose...
    assert "profile" not in texts
    # ...but the tool_result is a chunk tagged producer=tool with the name
    # resolved across turns via tool_use_id.
    i = texts.index("1000 rows")
    assert metas[i]["producer"] == "tool"
    assert metas[i]["tool_name"] == "profile"
    assert metas[i]["turn_id"] == "r1-1"


def test_empty_trace_writes_nothing(tmp_path):
    kb = FakeKB()
    ext = MemoryExtractor(kb, runtime_dir=tmp_path, session_id="proj:s")
    ext.write(Trace("t", "proj:s", "claude", "proj"))
    assert kb.calls == []


def test_extraction_span_tagged_episodic_not_memory(tmp_path):
    """The per-turn embedding runs off the heartbeat, so it must carry
    dsagt.source=episodic — filtering apart from the user-facing memory tools
    (kb_remember / kb_get_memories) that carry dsagt.source=memory."""
    from unittest.mock import patch

    opened = {}

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_open_span(name, span_type=None, source=None):
        opened["source"] = source
        return _Span()

    kb = FakeKB()
    ext = MemoryExtractor(kb, runtime_dir=tmp_path, session_id="proj:s")
    with patch("dsagt.observability.open_span", _fake_open_span):
        ext.write(_one_turn_trace())

    assert opened["source"] == "episodic"

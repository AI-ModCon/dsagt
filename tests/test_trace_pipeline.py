"""Parity test: our pipeline vs MLflow's own claude_code autolog.

The same Claude transcript is logged two ways into one serverless sqlite store:
  oracle: mlflow.claude_code.tracing.process_transcript(path)
  ours:   ClaudeTranslator → Trace → observability.MLflowSink

then the two resulting MLflow traces are compared structurally.  This is the
"how does our product match up" check — it pins our foreign-trace output to the
autolog look (AGENT root + llm/tool children, anthropic message format, token
usage) that makes those traces navigable.

Real mlflow logging to a local sqlite file — no network.  Skips cleanly if the
maintained autolog parser isn't importable.
"""

import json

import pytest

from dsagt.observability import MLflowSink
from dsagt.traces import ClaudeTranslator

process_transcript = pytest.importorskip(
    "mlflow.claude_code.tracing"
).process_transcript


def _asst(ts, *blocks, model="claude-sonnet-4-5", usage=None):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "model": model,
            "content": list(blocks),
            "usage": usage or {"input_tokens": 100, "output_tokens": 20},
        },
    }


def _user(ts, content, tool_use_result=False):
    rec = {
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": content},
    }
    if tool_use_result:
        rec["toolUseResult"] = {"stdout": "a.db\nb.db"}
    return rec


TRANSCRIPT = [
    _user("2026-06-19T15:49:29.000Z", "list the db files"),
    _asst("2026-06-19T15:49:30.000Z", {"type": "thinking", "thinking": "let me think"}),
    _asst(
        "2026-06-19T15:49:31.000Z",
        {"type": "text", "text": "Let me look at the files."},
    ),
    _asst(
        "2026-06-19T15:49:32.000Z",
        {
            "type": "tool_use",
            "id": "tu1",
            "name": "Bash",
            "input": {"command": "ls *.db"},
        },
    ),
    _user(
        "2026-06-19T15:49:33.000Z",
        [{"type": "tool_result", "tool_use_id": "tu1", "content": "a.db\nb.db"}],
        tool_use_result=True,
    ),
    _asst(
        "2026-06-19T15:49:35.000Z",
        {"type": "text", "text": "Found two databases: a.db and b.db."},
    ),
]


def _span_shape(trace):
    """(span_type, name) for every span, root first then children in start order."""
    spans = sorted(trace.data.spans, key=lambda s: s.start_time_ns)
    return [(str(s.span_type), s.name) for s in spans]


def _llm_spans(trace):
    return [s for s in trace.data.spans if str(s.span_type) == "LLM"]


def _durations_ns(trace):
    """Per-span (end - start) in ns, ordered by start time."""
    spans = sorted(trace.data.spans, key=lambda s: s.start_time_ns)
    return [s.end_time_ns - s.start_time_ns for s in spans]


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    """Point MLflow at an isolated sqlite store under tmp; chdir so the autolog
    parser's ``.claude/mlflow`` log dir also lands in tmp."""
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("parity")
    return uri


def test_our_pipeline_matches_autolog_span_layout(tmp_path, mlflow_sqlite):
    import mlflow

    # --- oracle: mlflow's maintained parser ---
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in TRANSCRIPT))
    oracle = process_transcript(str(path), session_id="proj:sess1")
    assert oracle is not None, "autolog parser produced no trace"

    # --- ours: translator → sink ---
    trace = ClaudeTranslator().translate(
        TRANSCRIPT, trace_id="tr1", session_id="proj:sess1", project="parity"
    )
    (ours_id,) = MLflowSink(mlflow_sqlite, "parity").write(
        trace
    )  # one turn → one trace
    ours = mlflow.get_trace(ours_id)

    # Same span tree: one AGENT root, two llm turns, one tool span — thinking
    # turn produces no span in either.
    assert _span_shape(ours) == _span_shape(oracle)
    assert _span_shape(ours) == [
        ("AGENT", "claude_code_conversation"),
        ("LLM", "llm"),
        ("TOOL", "tool_Bash"),
        ("LLM", "llm"),
    ]

    # Span durations match the autolog model (ported #1): both derive each
    # span's end from the next entry's timestamp, so the per-span durations line
    # up — and crucially none is the absurd "now minus backdated start".
    ours_durs = _durations_ns(ours)
    oracle_durs = _durations_ns(oracle)
    assert all(d > 0 for d in ours_durs)
    for d_ours, d_oracle in zip(ours_durs, oracle_durs):
        assert abs(d_ours - d_oracle) <= 2_000_000  # within 2ms (int-ns rounding)


def test_our_llm_spans_carry_the_navigable_attributes(mlflow_sqlite):
    import mlflow
    from mlflow.tracing.constant import SpanAttributeKey

    trace = ClaudeTranslator().translate(
        TRANSCRIPT, trace_id="tr2", session_id="proj:sess1", project="parity"
    )
    (tid,) = MLflowSink(mlflow_sqlite, "parity").write(trace)
    ours = mlflow.get_trace(tid)

    llm = _llm_spans(ours)
    assert llm, "expected llm spans"
    for span in llm:
        attrs = span.attributes
        # the anthropic message-format flag is what triggers Chat-UI rendering
        assert attrs.get(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"
        # outputs are an anthropic assistant message
        assert span.outputs["role"] == "assistant"
        assert span.outputs["type"] == "message"
        # token usage present
        assert attrs.get(SpanAttributeKey.CHAT_USAGE)["input_tokens"] == 100


def test_multi_turn_each_subtree_matches_autolog_on_its_slice(tmp_path, mlflow_sqlite):
    """A 2-turn session → 2 of our MLflow traces; each must match what autolog
    produces from that turn's slice in isolation."""
    import mlflow

    turn1 = [
        _user("2026-06-19T16:00:00.000Z", "first question"),
        _asst("2026-06-19T16:00:01.000Z", {"type": "text", "text": "first answer"}),
    ]
    turn2 = [
        _user("2026-06-19T16:00:02.000Z", "second question"),
        _asst(
            "2026-06-19T16:00:03.000Z",
            {"type": "tool_use", "id": "tu9", "name": "Bash", "input": {"c": "ls"}},
        ),
        _user(
            "2026-06-19T16:00:04.000Z",
            [{"type": "tool_result", "tool_use_id": "tu9", "content": "ok"}],
            tool_use_result=True,
        ),
        _asst("2026-06-19T16:00:05.000Z", {"type": "text", "text": "second answer"}),
    ]
    session = turn1 + turn2

    # ours: one translate of the whole session → two traces, in turn order
    trace = ClaudeTranslator().translate(
        session, trace_id="trM", session_id="proj:sessM", project="parity"
    )
    our_ids = MLflowSink(mlflow_sqlite, "parity").write(trace)
    assert len(our_ids) == 2

    # oracle: autolog on each turn's slice independently
    for k, slice_records in enumerate([turn1, turn2]):
        path = tmp_path / f"turn{k}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in slice_records))
        oracle = process_transcript(str(path), session_id="proj:sessM")
        assert oracle is not None
        ours = mlflow.get_trace(our_ids[k])
        assert _span_shape(ours) == _span_shape(oracle)


def test_session_tag_and_canonical_id_on_trace(mlflow_sqlite):
    import mlflow
    from mlflow.tracing.constant import TraceMetadataKey

    trace = ClaudeTranslator().translate(
        TRANSCRIPT, trace_id="tr3", session_id="proj:sess1", project="parity"
    )
    (tid,) = MLflowSink(mlflow_sqlite, "parity").write(trace)
    ours = mlflow.get_trace(tid)
    md = ours.info.trace_metadata
    assert md.get(TraceMetadataKey.TRACE_SESSION) == "proj:sess1"
    # Per-turn idempotency key: <session trace_id>:<root span id>.
    assert md.get("dsagt.trace_id").startswith("tr3:")
    assert md.get("dsagt.agent") == "claude"

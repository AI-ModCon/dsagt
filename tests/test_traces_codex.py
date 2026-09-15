"""Fixture tests for the Codex rollout translator.

Synthetic but grammar-faithful records (validated against a real rollout in
discovery): developer/AGENTS context, the real prompt, reasoning, assistant
text, function_call + custom_tool_call with their outputs, across two turns.
"""

import json

import pytest

from dsagt.observability import MLflowSink
from dsagt.traces import CodexReader, CodexTranslator


def _rec(ts, rtype, payload):
    return {"timestamp": ts, "type": rtype, "payload": payload}


def _msg(ts, role, *texts):
    bt = "output_text" if role == "assistant" else "input_text"
    return _rec(
        ts,
        "response_item",
        {
            "type": "message",
            "role": role,
            "content": [{"type": bt, "text": t} for t in texts],
        },
    )


def _fcall(ts, name, args, cid):
    return _rec(
        ts,
        "response_item",
        {
            "type": "function_call",
            "name": name,
            "arguments": json.dumps(args),
            "call_id": cid,
        },
    )


def _fout(ts, cid, out):
    return _rec(
        ts,
        "response_item",
        {"type": "function_call_output", "call_id": cid, "output": out},
    )


def _custom(ts, name, raw_input, cid):
    return _rec(
        ts,
        "response_item",
        {"type": "custom_tool_call", "name": name, "input": raw_input, "call_id": cid},
    )


def _custom_out(ts, cid, out):
    return _rec(
        ts,
        "response_item",
        {"type": "custom_tool_call_output", "call_id": cid, "output": out},
    )


def _reasoning(ts):
    return _rec(
        ts,
        "response_item",
        {"type": "reasoning", "summary": [], "encrypted_content": "x"},
    )


def _session():
    T = "2026-06-18T21:30:%02d.000Z"
    return [
        _rec(T % 0, "session_meta", {"id": "s", "cwd": "/p"}),
        _msg(T % 1, "developer", "system instructions"),
        _msg(T % 2, "user", "# AGENTS.md instructions for /p"),  # injected context
        _msg(T % 3, "user", "fix the hang"),  # the real prompt (turn 1)
        _reasoning(T % 4),  # skipped
        _msg(T % 5, "assistant", "Let me look."),  # LLM
        _fcall(T % 6, "exec_command", {"cmd": "ls"}, "c1"),  # TOOL
        _fout(T % 7, "c1", "a.py\nb.py"),
        _custom(T % 8, "apply_patch", "*** Begin Patch ...", "c2"),  # TOOL (raw input)
        _custom_out(T % 9, "c2", "Success."),
        _msg(T % 10, "assistant", "Fixed it."),  # LLM (final of turn 1)
        _msg(T % 11, "user", "now run the tests"),  # turn 2 prompt
        _msg(T % 12, "assistant", "Tests pass."),  # LLM
    ]


def _translate(records=None):
    return CodexTranslator().translate(
        records if records is not None else _session(),
        trace_id="cx",
        session_id="proj:s",
        project="codexproj",
    )


def test_agents_md_injection_is_not_the_prompt():
    roots = [s for s in _translate().spans if s["parent_id"] is None]
    assert len(roots) == 2
    assert roots[0]["attributes"]["prompt"] == "fix the hang"  # not the AGENTS.md msg
    assert roots[1]["attributes"]["prompt"] == "now run the tests"


def test_turn1_span_layout_skips_reasoning():
    trace = _translate()
    roots = [s for s in trace.spans if s["parent_id"] is None]
    t1 = [
        (s["kind"], s["name"])
        for s in trace.spans
        if s["parent_id"] == roots[0]["span_id"]
    ]
    # reasoning → no span; 2 assistant texts → 2 LLM; 2 tool calls → 2 TOOL.
    assert t1 == [
        ("LLM", "llm"),
        ("TOOL", "tool_exec_command"),
        ("TOOL", "tool_apply_patch"),
        ("LLM", "llm"),
    ]
    assert roots[0]["attributes"]["response"] == "Fixed it."


def test_tool_input_parsed_and_result_matched_by_call_id():
    trace = _translate()
    by_name = {s["name"]: s for s in trace.spans if s["kind"] == "TOOL"}
    exec_span = by_name["tool_exec_command"]
    assert exec_span["attributes"]["input"] == {"cmd": "ls"}  # JSON args parsed
    assert exec_span["attributes"]["result"] == "a.py\nb.py"
    patch_span = by_name["tool_apply_patch"]
    assert (
        patch_span["attributes"]["input"] == "*** Begin Patch ..."
    )  # raw custom input
    assert patch_span["attributes"]["result"] == "Success."


def test_durations_are_bounded_per_turn():
    trace = _translate()
    for s in trace.spans:
        if s["parent_id"] is not None:
            assert s["end_time"] is not None and s["end_time"] > s["start_time"]
    # turn 1's final LLM is bounded by turn 2's prompt? No — turns are
    # independent, so its last span falls back to 1s, not the gap to turn 2.
    roots = [s for s in trace.spans if s["parent_id"] is None]
    t1_last_llm = [
        s
        for s in trace.spans
        if s["parent_id"] == roots[0]["span_id"] and s["kind"] == "LLM"
    ][-1]
    assert t1_last_llm["end_time"] - t1_last_llm["start_time"] == 1.0


def test_empty_records_is_none():
    assert _translate([]) is None
    assert _translate([{"type": "event_msg", "payload": {}}]) is None


def test_reader_picks_the_rollout_matching_the_project_cwd(tmp_path):
    root = tmp_path / "sessions" / "2026" / "06" / "18"
    root.mkdir(parents=True)
    # Two rollouts; only one's session_meta.cwd matches the project dir.
    mine = root / "rollout-mine.jsonl"
    mine.write_text("\n".join(json.dumps(r) for r in _session()))
    other = root / "rollout-other.jsonl"
    other.write_text(
        json.dumps(_rec("t", "session_meta", {"id": "o", "cwd": "/elsewhere"})) + "\n"
    )
    reader = CodexReader("/p", sessions_root=tmp_path / "sessions")
    # _session()'s session_meta cwd is "/p".
    assert reader.active_file() == mine
    records = reader.read()
    assert any(r.get("type") == "response_item" for r in records)


def test_reader_no_match_is_empty(tmp_path):
    (tmp_path / "sessions").mkdir()
    records = CodexReader("/p", sessions_root=tmp_path / "sessions").read()
    assert records == []


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("codexproj")
    return uri


def test_end_to_end_through_the_sink(mlflow_sqlite):
    import mlflow

    trace = _translate()
    ids = MLflowSink(mlflow_sqlite, "codexproj").write(trace)
    assert len(ids) == 2  # one MLflow trace per turn
    exp = mlflow.get_experiment_by_name("codexproj")
    traces = mlflow.search_traces(locations=[exp.experiment_id], return_type="list")
    assert len(traces) == 2
    # the tool-bearing turn rendered llm + tool spans under its agent root
    shapes = {len(t.data.spans) for t in traces}
    # tool-bearing turn: root + 2 llm + 2 tool = 5; second turn: root + 1 llm = 2.
    # Pin both — `5 in shapes` alone would pass even if turn 2 collapsed.
    assert shapes == {5, 2}


def _token_count(ts, *, input_tokens, output_tokens, cached=0):
    return _rec(
        ts,
        "event_msg",
        {
            "type": "token_count",
            "info": {
                "last_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached,
                    "cache_write_input_tokens": 0,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            },
        },
    )


def test_token_usage_and_model_are_carried_once_per_llm_call():
    """A call's ``token_count`` lands on its assistant message, or on its first
    tool call when the call produced no text — so a tool-only call still
    counts.  OpenAI's cached count is a subset of input and passes through."""
    records = [
        _rec("2026-06-19T15:00:00.000Z", "session_meta", {"cwd": "/p"}),
        _rec("2026-06-19T15:00:00.100Z", "turn_context", {"model": "gpt-x"}),
        _msg("2026-06-19T15:00:01.000Z", "user", "do it"),
        # call 1: tool only, no text
        _fcall("2026-06-19T15:00:02.000Z", "shell", {"cmd": "ls"}, "c1"),
        _fout("2026-06-19T15:00:03.000Z", "c1", "ok"),
        _token_count(
            "2026-06-19T15:00:03.500Z", input_tokens=1000, output_tokens=10, cached=800
        ),
        # call 2: text
        _msg("2026-06-19T15:00:04.000Z", "assistant", "done"),
        _token_count("2026-06-19T15:00:04.500Z", input_tokens=1200, output_tokens=5),
    ]
    trace = _translate(records)
    with_usage = [s for s in trace.spans if s.get("usage")]
    assert [s["kind"] for s in with_usage] == ["TOOL", "LLM"]
    assert with_usage[0]["usage"]["input_tokens"] == 1000  # not 1000 + 800
    assert with_usage[0]["usage"]["cache_read_input_tokens"] == 800
    assert with_usage[1]["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
    }
    assert with_usage[1]["model"] == "gpt-x"
    assert sum(s["usage"]["input_tokens"] for s in with_usage) == 2200

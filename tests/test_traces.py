"""Fixture tests for the canonical trace waist + Claude translator (Phase 2).

Pure, no disk / no network.  Two things under test:
- ``to_exchanges`` projects LLM spans onto the conversational shape the episodic
  indexer chunks + embeds (carrying ``turn_id`` + content) — the seam.
- ``ClaudeTranslator`` turns raw transcript records into a ``Trace``
  with the autolog-style AGENT/LLM/TOOL span layout.
"""

import json

from dsagt.traces import (
    ClaudeReader,
    ClaudeTranslator,
    Trace,
    _transcript_dir,
)

# ---------------------------------------------------------------------------
# to_exchanges projection (windowed model)
# ---------------------------------------------------------------------------


def _windowed_trace() -> Trace:
    """Two LLM spans whose ``request`` already holds the per-turn window."""
    trace = Trace(
        trace_id="tr1", session_id="proj:sess1", agent="claude", project="proj"
    )
    trace.add_llm_span(
        "s1",
        parent_id="r1",
        start_time=100.0,
        end_time=None,
        request=[
            {"role": "user", "content": [{"type": "text", "text": "Profile sales.csv"}]}
        ],
        response=[
            {
                "type": "tool_use",
                "id": None,
                "name": "profile",
                "input": {"path": "sales.csv"},
            }
        ],
    )
    trace.add_tool_span(
        "s3",
        parent_id="r1",
        start_time=101.0,
        end_time=None,
        name="profile",
        tool_input={"path": "sales.csv"},
        result="1000 rows",
    )
    trace.add_llm_span(
        "s2",
        parent_id="r1",
        start_time=102.0,
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


def test_to_exchanges_one_per_llm_span_skipping_tool_spans():
    exchanges = _windowed_trace().to_exchanges()
    assert len(exchanges) == 2
    assert [e["timestamp"] for e in exchanges] == [100.0, 102.0]


def test_to_exchanges_uses_request_window_directly():
    ex1, ex2 = _windowed_trace().to_exchanges()
    assert [m["role"] for m in ex1["new_messages"]] == ["user"]
    # The tool_result block round-trips in Anthropic shape memory can read.
    block = ex2["new_messages"][0]["content"][0]
    assert block == {"type": "tool_result", "tool_use_id": "t1", "content": "1000 rows"}
    assert ex2["response"] == [{"type": "text", "text": "The file has 1000 rows."}]


def test_projection_carries_turn_id_and_content():
    """The waist's whole justification: to_exchanges carries ``turn_id`` (groups
    a turn's chunks) and the real content the episodic indexer embeds."""
    exchanges = _windowed_trace().to_exchanges()
    assert all(ex["turn_id"] for ex in exchanges)  # span id, for chunk grouping
    flat = json.dumps(exchanges)
    assert "Profile sales.csv" in flat
    assert "profile" in flat  # tool_use name (in the assistant response)
    assert "1000 rows" in flat  # tool_result content
    assert "The file has 1000 rows." in flat  # final answer


def test_none_tolerant_usage_and_timing():
    trace = Trace(trace_id="t", session_id="s", agent="goose", project="p")
    span = trace.add_llm_span(
        "s1", parent_id="r1", start_time=None, end_time=None, request=[], response=[]
    )
    assert span["usage"] is None and span["start_time"] is None
    assert trace.to_exchanges() == [
        {"turn_id": "s1", "timestamp": None, "new_messages": [], "response": []}
    ]


# ---------------------------------------------------------------------------
# ClaudeTranslator
# ---------------------------------------------------------------------------


def _asst(ts, *blocks, model="claude-x", usage=None):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "model": model,
            "content": list(blocks),
            "usage": usage or {"input_tokens": 10, "output_tokens": 3},
        },
    }


def _user(ts, content, tool_use_result=False):
    rec = {
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": content},
    }
    if tool_use_result:
        rec["toolUseResult"] = {"stdout": "..."}
    return rec


def _single_turn_records():
    """user → thinking → text → tool_use → tool_result → text (final)."""
    return [
        _user("2026-06-19T15:49:29.000Z", "list the db files"),
        _asst("2026-06-19T15:49:30.000Z", {"type": "thinking", "thinking": "hmm"}),
        _asst("2026-06-19T15:49:31.000Z", {"type": "text", "text": "Let me look."}),
        _asst(
            "2026-06-19T15:49:32.000Z",
            {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "ls"}},
        ),
        _user(
            "2026-06-19T15:49:33.000Z",
            [{"type": "tool_result", "tool_use_id": "tu1", "content": "a.db\nb.db"}],
            tool_use_result=True,
        ),
        _asst(
            "2026-06-19T15:49:34.000Z", {"type": "text", "text": "Found a.db and b.db."}
        ),
    ]


def _translate(records):
    return ClaudeTranslator().translate(
        records, trace_id="tr", session_id="proj:s", project="proj"
    )


def test_translate_builds_agent_root_with_prompt_and_response():
    trace = _translate(_single_turn_records())
    root = trace.spans[0]
    assert root["kind"] == "AGENT"
    assert root["name"] == "claude_code_conversation"
    assert root["attributes"]["prompt"] == "list the db files"
    assert root["attributes"]["response"] == "Found a.db and b.db."
    assert trace.started_at is not None and trace.ended_at is not None


def test_translate_span_layout_matches_autolog_shape():
    trace = _translate(_single_turn_records())
    kinds = [(s["kind"], s["name"]) for s in trace.spans]
    # thinking turn produces NO span; two text turns → 2 LLM; one tool_use → 1 TOOL.
    assert kinds == [
        ("AGENT", "claude_code_conversation"),
        ("LLM", "llm"),
        ("TOOL", "tool_Bash"),
        ("LLM", "llm"),
    ]


def test_translate_llm_span_carries_model_usage_and_window():
    trace = _translate(_single_turn_records())
    llms = [s for s in trace.spans if s["kind"] == "LLM"]
    first, second = llms
    assert first["model"] == "claude-x"
    assert first["usage"]["input_tokens"] == 10 and first["usage"]["output_tokens"] == 3
    # The final LLM turn's window is the tool_use + tool_result run since the
    # previous text turn (the "Let me look." boundary is excluded).
    roles = [m["role"] for m in second["request"]]
    assert "user" in roles  # the tool_result message
    assert second["response"][0]["text"] == "Found a.db and b.db."


def test_translate_tool_span_has_input_and_result():
    trace = _translate(_single_turn_records())
    tool = next(s for s in trace.spans if s["kind"] == "TOOL")
    assert tool["attributes"]["tool_name"] == "Bash"
    assert tool["attributes"]["input"] == {"cmd": "ls"}
    assert tool["attributes"]["result"] == "a.db\nb.db"
    assert tool["parent_id"] == trace.spans[0]["span_id"]


def test_translate_empty_transcript_is_none():
    assert _translate([]) is None


# ---------------------------------------------------------------------------
# Ported edge cases (from mlflow's claude_code parser)
# ---------------------------------------------------------------------------


def test_spans_get_real_durations_not_a_now_end():
    """#1 — every span ends via the next-timestamp model, never an open end_time
    (which the sink would otherwise stamp as wall-clock now → absurd duration)."""
    trace = _translate(_single_turn_records())
    for span in trace.spans:
        assert span["start_time"] is not None and span["end_time"] is not None
        assert span["end_time"] > span["start_time"]
    # Final LLM turn has no following entry → 1s fallback, not a huge span.
    final_llm = [s for s in trace.spans if s["kind"] == "LLM"][-1]
    assert final_llm["end_time"] - final_llm["start_time"] == 1.0


def test_skill_injection_user_message_is_not_taken_as_the_prompt():
    """#2 — a user entry following a Skill tool result (commandName) is an
    injection, not the human prompt; the real prompt is selected instead."""
    records = [
        _user("2026-06-19T15:00:00.000Z", "real prompt"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "answer"}),
        {
            "type": "user",
            "timestamp": "2026-06-19T15:00:02.000Z",
            "toolUseResult": {"commandName": "some-skill"},
            "message": {"role": "user", "content": "skill tool result"},
        },
        _user("2026-06-19T15:00:03.000Z", "injected skill content"),
    ]
    trace = _translate(records)
    assert trace.spans[0]["attributes"]["prompt"] == "real prompt"


def test_local_command_stdout_is_not_taken_as_the_prompt():
    """#2 — slash-command stdout echoes are skipped when finding the prompt."""
    records = [
        _user("2026-06-19T15:00:00.000Z", "real prompt"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "answer"}),
        _user(
            "2026-06-19T15:00:02.000Z",
            "<local-command-stdout>ran /foo</local-command-stdout>",
        ),
    ]
    trace = _translate(records)
    assert trace.spans[0]["attributes"]["prompt"] == "real prompt"


def test_steer_message_folds_into_the_llm_window():
    """#3 — a queue-operation/enqueue steer message becomes a user message in
    the following text turn's input window."""
    records = [
        _user("2026-06-19T15:00:00.000Z", "start a long task"),
        _asst(
            "2026-06-19T15:00:01.000Z",
            {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"c": "sleep"}},
        ),
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "2026-06-19T15:00:02.000Z",
            "content": "actually, stop and summarize",
        },
        _user(
            "2026-06-19T15:00:03.000Z",
            [{"type": "tool_result", "tool_use_id": "tu1", "content": "done"}],
            tool_use_result=True,
        ),
        _asst("2026-06-19T15:00:04.000Z", {"type": "text", "text": "Summary."}),
    ]
    trace = _translate(records)
    final_llm = [s for s in trace.spans if s["kind"] == "LLM"][-1]
    texts = [
        b["text"]
        for m in final_llm["request"]
        for b in m["content"]
        if b["type"] == "text"
    ]
    assert "actually, stop and summarize" in texts


# ---------------------------------------------------------------------------
# Multi-turn segmentation
# ---------------------------------------------------------------------------


def _two_turn_records():
    return [
        _user("2026-06-19T15:00:00.000Z", "first question"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "first answer"}),
        _user("2026-06-19T15:00:02.000Z", "second question"),
        _asst(
            "2026-06-19T15:00:03.000Z",
            {"type": "tool_use", "id": "t", "name": "Bash", "input": {"c": "ls"}},
        ),
        _user(
            "2026-06-19T15:00:04.000Z",
            [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
            tool_use_result=True,
        ),
        _asst("2026-06-19T15:00:05.000Z", {"type": "text", "text": "second answer"}),
    ]


def test_translate_segments_one_agent_subtree_per_turn():
    trace = _translate(_two_turn_records())
    roots = [s for s in trace.spans if s["parent_id"] is None]
    assert len(roots) == 2
    assert roots[0]["attributes"]["prompt"] == "first question"
    assert roots[1]["attributes"]["prompt"] == "second question"
    # Turn 2's tool + final-answer LLM hang off turn 2's root, not turn 1's.
    t2 = [s for s in trace.spans if s["parent_id"] == roots[1]["span_id"]]
    assert {s["kind"] for s in t2} == {"LLM", "TOOL"}
    assert roots[1]["attributes"]["response"] == "second answer"


def test_each_turn_is_self_contained_no_cross_turn_duration_borrow():
    trace = _translate(_two_turn_records())
    roots = [s for s in trace.spans if s["parent_id"] is None]
    # Turn 1's only LLM span is its last → 1s fallback, NOT the 2s gap to turn 2.
    t1_llm = next(
        s
        for s in trace.spans
        if s["parent_id"] == roots[0]["span_id"] and s["kind"] == "LLM"
    )
    assert t1_llm["end_time"] - t1_llm["start_time"] == 1.0


def test_translate_skips_a_leading_partial_turn():
    """Records before the first prompt in the batch belong to an already-processed
    turn (incremental read) — they produce no subtree, which is what keeps
    cursor-driven reads idempotent at turn granularity."""
    records = [
        _asst(
            "2026-06-19T15:00:00.000Z", {"type": "text", "text": "tail of prev turn"}
        ),
        _user("2026-06-19T15:00:01.000Z", "new prompt"),
        _asst("2026-06-19T15:00:02.000Z", {"type": "text", "text": "answer"}),
    ]
    trace = _translate(records)
    roots = [s for s in trace.spans if s["parent_id"] is None]
    assert len(roots) == 1
    assert roots[0]["attributes"]["prompt"] == "new prompt"


# ---------------------------------------------------------------------------
# ClaudeReader — whole-file read
# ---------------------------------------------------------------------------


def _write_lines(path, objs, mode="w"):
    with open(path, mode) as fh:
        for o in objs:
            fh.write(json.dumps(o) + "\n")


def test_reader_reads_the_whole_active_file(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    root = tmp_path / "projects"
    tdir = _transcript_dir(proj, root)
    tdir.mkdir(parents=True)
    f = tdir / "sess.jsonl"
    _write_lines(f, [{"a": 1}, {"a": 2}])

    reader = ClaudeReader(proj, projects_root=root)
    recs = reader.read()
    assert [r["a"] for r in recs] == [1, 2]

    _write_lines(f, [{"a": 3}], mode="a")
    recs2 = reader.read()
    assert [r["a"] for r in recs2] == [1, 2, 3]  # re-reads the whole file


def test_reader_leaves_a_partial_trailing_line(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    root = tmp_path / "projects"
    tdir = _transcript_dir(proj, root)
    tdir.mkdir(parents=True)
    f = tdir / "sess.jsonl"
    _write_lines(f, [{"a": 1}])

    reader = ClaudeReader(proj, projects_root=root)
    assert [r["a"] for r in reader.read()] == [1]

    # A half-written append (no newline) must not be parsed.
    with open(f, "a") as fh:
        fh.write('{"a": 2')
    assert [r["a"] for r in reader.read()] == [1]

    # Once the line completes, it's read.
    with open(f, "a") as fh:
        fh.write("}\n")
    assert [r["a"] for r in reader.read()] == [1, 2]


def test_reader_picks_the_most_recent_session(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    root = tmp_path / "projects"
    tdir = _transcript_dir(proj, root)
    tdir.mkdir(parents=True)
    old = tdir / "old.jsonl"
    _write_lines(old, [{"a": 1}, {"a": 2}])

    # A newer session file becomes active; the reader follows the newest mtime.
    import os

    new = tdir / "new.jsonl"
    _write_lines(new, [{"b": 1}])
    newer = old.stat().st_mtime + 10
    os.utime(new, (newer, newer))  # ensure new is the most-recent file
    reader = ClaudeReader(proj, projects_root=root)
    assert [r.get("b") for r in reader.read()] == [1]


def test_reader_no_transcripts_is_empty(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    reader = ClaudeReader(proj, projects_root=tmp_path / "projects")
    assert reader.read() == []

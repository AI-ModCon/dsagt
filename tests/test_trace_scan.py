"""Tests for the in-session trace heartbeat (TraceCollector).

Exercise the collect logic directly (no event loop): the completeness watermark
(defer the open last turn) and ack-set idempotency, end-to-end through the real
ClaudeReader + ClaudeTranslator + MLflowSink into a tmp sqlite store.
"""

import json

import pytest

from dsagt.traces import (
    Trace,
    TraceCollector,
    _transcript_dir,
    make_trace_collector,
)


def _asst(ts, *blocks):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "model": "claude-x",
            "content": list(blocks),
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    }


def _user(ts, content, uuid, tool_use_result=False):
    rec = {
        "type": "user",
        "timestamp": ts,
        "uuid": uuid,
        "message": {"role": "user", "content": content},
    }
    if tool_use_result:
        rec["toolUseResult"] = {"stdout": "ok"}
    return rec


def _append(path, *records):
    with open(path, "a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "proj"
    (proj / ".dsagt").mkdir(parents=True)
    proot = tmp_path / "projects"
    tdir = _transcript_dir(proj, proot)
    tdir.mkdir(parents=True)
    f = tdir / "sess.jsonl"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    collector = make_trace_collector(
        "claude", proj, "proj", "proj:s", uri, projects_root=proot
    )
    return collector, f, proj


def test_make_trace_collector_registered_agents(tmp_path):
    # The heartbeat isn't Claude-special: it runs for any agent with a pipeline.
    for agent in ("claude", "codex", "goose", "opencode", "cline"):
        assert (
            make_trace_collector(agent, tmp_path, "p", "p:s", "sqlite:///x.db")
            is not None
        )
    # An agent with no pipeline registered simply gets no heartbeat.
    assert (
        make_trace_collector("nonesuch", tmp_path, "p", "p:s", "sqlite:///x.db") is None
    )


def test_subset_keeps_only_named_roots_and_their_children():
    trace = Trace("t", "s", "claude", "p")
    trace.add_agent_root("r1", "claude_code_conversation", start_time=None, prompt="")
    trace.add_llm_span(
        "r1-0", parent_id="r1", start_time=None, end_time=None, request=[], response=[]
    )
    trace.add_agent_root("r2", "claude_code_conversation", start_time=None, prompt="")
    trace.add_llm_span(
        "r2-0", parent_id="r2", start_time=None, end_time=None, request=[], response=[]
    )
    sub = trace.subset({"r1"})
    assert [s["span_id"] for s in sub.spans] == ["r1", "r1-0"]


def test_periodic_collect_emits_complete_turns_and_defers_the_open_last(scan_env):
    collector, f, _ = scan_env
    # Two complete turns, then an open turn (prompt only).
    _append(
        f,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
        _user("2026-06-19T15:00:02.000Z", "q2", "u2"),
        _asst("2026-06-19T15:00:03.000Z", {"type": "text", "text": "a2"}),
        _user("2026-06-19T15:00:04.000Z", "q3 (still working)", "u3"),
    )
    # roots = [u1, u2, u3]; periodic emits roots[:-1] = u1, u2; u3 deferred.
    assert collector.collect() == 2
    assert collector._load_acks("mlflow") == {"proj:s:u1", "proj:s:u2"}


def test_collect_is_idempotent(scan_env):
    collector, f, _ = scan_env
    _append(
        f,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
        _user("2026-06-19T15:00:02.000Z", "q2", "u2"),
    )
    assert collector.collect() == 1  # u1 emitted, u2 (last) deferred
    assert collector.collect() == 0  # nothing new
    assert collector.collect() == 0


def test_deferred_turn_emits_once_a_later_prompt_bounds_it(scan_env):
    collector, f, _ = scan_env
    _append(
        f,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
        _user("2026-06-19T15:00:02.000Z", "q2", "u2"),
    )
    assert collector.collect() == 1  # u1 only
    # A new prompt arrives → u2 is no longer the open turn.
    _append(
        f,
        _asst("2026-06-19T15:00:03.000Z", {"type": "text", "text": "a2"}),
        _user("2026-06-19T15:00:04.000Z", "q3", "u3"),
    )
    assert collector.collect() == 1  # u2 now emitted; u3 deferred
    assert collector._load_acks("mlflow") == {"proj:s:u1", "proj:s:u2"}


def test_final_flush_emits_the_deferred_last_turn(scan_env):
    collector, f, _ = scan_env
    _append(
        f,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
        _user("2026-06-19T15:00:02.000Z", "q2", "u2"),
        _asst("2026-06-19T15:00:03.000Z", {"type": "text", "text": "a2"}),
    )
    assert collector.collect() == 1  # u1; u2 deferred
    assert collector.collect(include_last=True) == 1  # end-of-session flush emits u2
    assert collector.collect(include_last=True) == 0  # idempotent
    assert collector._load_acks("mlflow") == {"proj:s:u1", "proj:s:u2"}


def test_collect_on_empty_transcript_is_zero(scan_env):
    collector, f, _ = scan_env
    f.write_text("")
    assert collector.collect() == 0
    assert collector.collect(include_last=True) == 0


class _Recorder:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.seen = []

    def write(self, trace):
        if self.fail:
            raise RuntimeError("consumer down")
        self.seen.append({s["span_id"] for s in trace.spans if s["parent_id"] is None})


class _FakeReader:
    def read(self):
        return [{"x": 1}]  # non-empty so collect proceeds; translator ignores


class _FakeTranslator:
    def __init__(self, trace):
        self._trace = trace

    def translate(self, records, **kw):
        return self._trace


def test_consumers_ack_independently(tmp_path):
    # Two AGENT roots → two complete turns (include_last emits both).
    trace = Trace("t", "p:s", "claude", "p")
    trace.add_agent_root("r1", "c", start_time=None, prompt="")
    trace.add_agent_root("r2", "c", start_time=None, prompt="")
    good = _Recorder("good")
    bad = _Recorder("bad", fail=True)
    collector = TraceCollector(
        _FakeReader(),
        _FakeTranslator(trace),
        project="p",
        session_id="p:s",
        project_dir=tmp_path,
        consumers=[good, bad],
    )

    # One turn advanced for at least one consumer → returns 2 (both turns new
    # to the good consumer); the failing one logs and holds its mark back.
    assert collector.collect(include_last=True) == 2
    assert good.seen == [{"r1", "r2"}]
    # Acks are session-qualified (<session_id>:<span_id>).
    assert collector._load_acks("good") == {"p:s:r1", "p:s:r2"}
    assert collector._load_acks("bad") == set()  # wedged consumer didn't advance

    # Good is fully caught up; bad retries (and fails again) — good doesn't redo.
    assert collector.collect(include_last=True) == 0
    assert good.seen == [{"r1", "r2"}]  # not re-delivered
    assert collector._load_acks("bad") == set()


def test_acks_are_session_qualified_no_cross_session_collision(tmp_path):
    """Two sessions in one project share the ack file, but each turn is keyed by
    ``<session_id>:<span_id>`` — so session B's turns (the same per-transcript
    ``turn-N`` indices) are not suppressed by session A's acks.  Regression for
    the cross-session collision that silently dropped every session after the
    first.
    """

    def _trace(session_id):
        # Both sessions produce the SAME index-based span ids ("r1"/"r2").
        t = Trace("t", session_id, "claude", "p")
        t.add_agent_root("r1", "c", start_time=None, prompt="")
        t.add_agent_root("r2", "c", start_time=None, prompt="")
        return t

    rec_a = _Recorder("mlflow")
    collector_a = TraceCollector(
        _FakeReader(),
        _FakeTranslator(_trace("p:1")),
        project="p",
        session_id="p:1",
        project_dir=tmp_path,
        consumers=[rec_a],
    )
    assert collector_a.collect(include_last=True) == 2

    # Session B: new collector, SAME project_dir (shared ack file), new session
    # id, identical span ids.  Must still emit both turns.
    rec_b = _Recorder("mlflow")
    collector_b = TraceCollector(
        _FakeReader(),
        _FakeTranslator(_trace("p:2")),
        project="p",
        session_id="p:2",
        project_dir=tmp_path,
        consumers=[rec_b],
    )
    assert collector_b.collect(include_last=True) == 2
    assert rec_b.seen == [{"r1", "r2"}]
    # Both sessions' qualified keys coexist in the one shared ack file.
    assert collector_b._load_acks("mlflow") == {
        "p:1:r1",
        "p:1:r2",
        "p:2:r1",
        "p:2:r2",
    }


def test_make_trace_collector_pins_source(tmp_path):
    """A pinned source is read regardless of the newest-file logic, and the
    collector reports it via active_source()."""
    transcript = tmp_path / "pinned.jsonl"
    _append(
        transcript,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
    )
    collector = make_trace_collector(
        "claude",
        tmp_path / "proj",
        "p",
        "p:1",
        f"sqlite:///{tmp_path / 'm.db'}",
        source=str(transcript),
    )
    assert collector.active_source() == str(transcript)
    assert collector.collect(include_last=True) == 1  # the one complete turn


def test_record_trace_source_updates_current_session(tmp_path):
    from dsagt.session import read_state, record_trace_source, write_state

    proj = tmp_path / "proj"
    (proj / ".dsagt").mkdir(parents=True)
    write_state(proj, {"sessions": [{"id": 1}], "memory_cursor": {}})

    record_trace_source(proj, "/path/to/t.jsonl")
    assert read_state(proj)["sessions"][-1]["trace_source"] == "/path/to/t.jsonl"
    # A non-str token (e.g. a SQLite session id) round-trips through YAML.
    record_trace_source(proj, 42)
    assert read_state(proj)["sessions"][-1]["trace_source"] == 42
    # No session minted yet → safe no-op.
    write_state(tmp_path / "empty", {"sessions": [], "memory_cursor": {}})
    record_trace_source(tmp_path / "empty", "/x")  # must not raise


def test_catch_up_traces_emits_previous_session_dangling_and_is_idempotent(tmp_path):
    """The startup catch-up re-collects the previous session's pinned transcript,
    emits the turns the live pass missed, and a second pass is a no-op (the
    session-qualified ack files dedupe)."""
    from unittest.mock import MagicMock

    from dsagt.session import _catch_up_traces, write_state

    proj = tmp_path / "proj"
    (proj / ".dsagt").mkdir(parents=True)
    transcript = tmp_path / "prev.jsonl"  # pinned, so location is irrelevant
    _append(
        transcript,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
        _user("2026-06-19T15:00:02.000Z", "q2", "u2"),
        _asst("2026-06-19T15:00:03.000Z", {"type": "text", "text": "a2"}),
    )
    # Previous session (id=1) recorded its trace source; current session is id=2.
    write_state(
        proj,
        {
            "sessions": [{"id": 1, "trace_source": str(transcript)}, {"id": 2}],
            "memory_cursor": {},
        },
    )
    config = {"project": "proj", "agent": "claude", "project_dir": str(proj)}

    n = _catch_up_traces(proj, config, MagicMock())
    assert n == 2  # both completed turns flushed via include_last
    # Idempotent — acks (keyed proj-1:<uuid>) suppress a re-collect.
    assert _catch_up_traces(proj, config, MagicMock()) == 0


def test_catch_up_traces_noop_without_previous_or_transcript(tmp_path):
    from unittest.mock import MagicMock

    from dsagt.session import _catch_up_traces, write_state

    proj = tmp_path / "proj"
    (proj / ".dsagt").mkdir(parents=True)
    config = {"project": "proj", "agent": "claude", "project_dir": str(proj)}

    # Only one session → nothing to catch up.
    write_state(proj, {"sessions": [{"id": 1}], "memory_cursor": {}})
    assert _catch_up_traces(proj, config, MagicMock()) == 0

    # Previous session exists but recorded no transcript (SQLite agent / too
    # short) → skip rather than risk reading the new session's transcript.
    write_state(
        proj, {"sessions": [{"id": 1}, {"id": 2}], "memory_cursor": {}}
    )
    assert _catch_up_traces(proj, config, MagicMock()) == 0


def test_sqlite_reader_pins_to_a_specific_session(tmp_path):
    """SQLite agents pin uniformly: a pinned GooseReader reads the *specified*
    session, not the latest — so the catch-up backstops goose/opencode/cline
    too, not just the JSONL agents.  (opencode/cline mirror this shape.)
    """
    import os
    import sqlite3

    from dsagt.traces import GooseReader

    db = tmp_path / "sessions.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE sessions (id TEXT, working_dir TEXT, updated_at INTEGER);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,"
        " content_json TEXT, created_timestamp INTEGER);"
    )
    wd = os.path.abspath(tmp_path / "proj")
    con.execute("INSERT INTO sessions VALUES ('A', ?, 100)", (wd,))  # older
    con.execute("INSERT INTO sessions VALUES ('B', ?, 200)", (wd,))  # newer
    con.execute("INSERT INTO messages VALUES (1, 'A', 'user', '[\"qA\"]', 1)")
    con.execute("INSERT INTO messages VALUES (2, 'B', 'user', '[\"qB\"]', 2)")
    con.commit()
    con.close()

    reader = GooseReader(tmp_path / "proj", db_path=db)
    # Unpinned → newest session (B), and reports its id.
    assert reader.active_source() == "B"
    assert reader.read()[0]["content"] == ["qB"]
    # Pinned to the older session A → reads exactly A (the catch-up's job).
    reader.pin("A")
    assert reader.active_source() == "A"
    assert reader.read()[0]["content"] == ["qA"]


def test_emitted_traces_land_in_the_store(scan_env):
    import mlflow

    collector, f, _ = scan_env
    _append(
        f,
        _user("2026-06-19T15:00:00.000Z", "q1", "u1"),
        _asst("2026-06-19T15:00:01.000Z", {"type": "text", "text": "a1"}),
        _user("2026-06-19T15:00:02.000Z", "q2", "u2"),
        _asst("2026-06-19T15:00:03.000Z", {"type": "text", "text": "a2"}),
    )
    collector.collect(include_last=True)  # emit both turns
    exp = mlflow.get_experiment_by_name("proj")
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id], return_type="list"
    )
    sessions = {t.info.trace_metadata.get("mlflow.trace.session") for t in traces}
    assert len(traces) == 2
    assert sessions == {"proj:s"}

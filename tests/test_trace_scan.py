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
    assert collector._load_acks("mlflow") == {"u1", "u2"}


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
    assert collector._load_acks("mlflow") == {"u1", "u2"}


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
    assert collector._load_acks("mlflow") == {"u1", "u2"}


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
    assert collector._load_acks("good") == {"r1", "r2"}
    assert collector._load_acks("bad") == set()  # wedged consumer didn't advance

    # Good is fully caught up; bad retries (and fails again) — good doesn't redo.
    assert collector.collect(include_last=True) == 0
    assert good.seen == [{"r1", "r2"}]  # not re-delivered
    assert collector._load_acks("bad") == set()


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

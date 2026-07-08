"""Fixture tests for the Cline CLI translator + reader.

The text path mirrors real ``~/.cline`` sessions (validated in discovery); the
tool path uses standard Anthropic blocks (what Cline's SDK emits) since the
captured sessions were text-only.
"""

import json

import pytest

from dsagt.observability import MLflowSink
from dsagt.traces import ClineReader, ClineTranslator


def _user(ts, text):
    return {
        "id": f"u{ts}",
        "role": "user",
        "ts": ts,
        "content": [{"type": "text", "text": text}],
    }


def _asst(ts, text=None, tools=(), model=None):
    content = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for name, inp, tid in tools:
        content.append({"type": "tool_use", "id": tid, "name": name, "input": inp})
    return {
        "id": f"a{ts}",
        "role": "assistant",
        "ts": ts,
        "content": content,
        "model": model,
    }


def _toolresult(ts, tid, result):
    return {
        "id": f"t{ts}",
        "role": "user",
        "ts": ts,
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": result}],
    }


def _session():
    return [
        _user(1_000_000, '<user_input mode="act">do the thing</user_input>'),  # turn 1
        _asst(
            1_001_000,
            "Let me run a command.",
            tools=[("shell", {"cmd": "ls"}, "t1")],
            model="m",
        ),
        _toolresult(1_002_000, "t1", "a.txt"),  # tool result (user msg, not a prompt)
        _asst(1_003_000, "Done.", model="m"),
        _user(1_004_000, "<user_input>next</user_input>"),  # turn 2
        _asst(1_005_000, "OK.", model="m"),
    ]


def _translate(records=None):
    return ClineTranslator().translate(
        records if records is not None else _session(),
        trace_id="cl",
        session_id="proj:s",
        project="clineproj",
    )


def test_user_input_wrapper_stripped_and_turns_segmented():
    roots = [s for s in _translate().spans if s["parent_id"] is None]
    assert len(roots) == 2  # the tool_result user message is not a turn
    assert roots[0]["attributes"]["prompt"] == "do the thing"  # <user_input> unwrapped
    assert roots[1]["attributes"]["prompt"] == "next"


def test_assistant_text_and_tool_use_in_one_message():
    trace = _translate()
    roots = [s for s in trace.spans if s["parent_id"] is None]
    t1 = [
        (s["kind"], s["name"])
        for s in trace.spans
        if s["parent_id"] == roots[0]["span_id"]
    ]
    # message 1 has text + tool_use → an LLM span then a TOOL span; then "Done." LLM.
    assert t1 == [
        ("LLM", "llm"),
        ("TOOL", "tool_shell"),
        ("LLM", "llm"),
    ]
    tool = next(s for s in trace.spans if s["kind"] == "TOOL")
    assert tool["attributes"]["input"] == {"cmd": "ls"}
    assert tool["attributes"]["result"] == "a.txt"
    assert roots[0]["attributes"]["response"] == "Done."
    assert trace.spans[1]["model"] == "m"  # first LLM carries the session model


def test_tool_result_as_text_block_list():
    records = [
        _user(1, "<user_input>go</user_input>"),
        _asst(2, "calling", tools=[("t", {}, "x1")]),
        _toolresult(3, "x1", [{"type": "text", "text": "block result"}]),
        _asst(4, "ok"),
    ]
    tool = next(s for s in _translate(records).spans if s["kind"] == "TOOL")
    assert tool["attributes"]["result"] == "block result"


def test_empty_is_none():
    assert _translate([]) is None
    assert _translate([_asst(1, "no user prompt")]) is None


# --- reader over a tiny ~/.cline/data/sessions tree ---


def _make_session(root, sid, cwd, messages, model="m"):
    d = root / sid
    d.mkdir(parents=True)
    (d / f"{sid}.json").write_text(json.dumps({"cwd": cwd, "model": model}))
    (d / f"{sid}.messages.json").write_text(json.dumps({"messages": messages}))


def test_reader_picks_newest_session_for_the_cwd(tmp_path):
    root = tmp_path / "sessions"
    _make_session(root, "200_p1", "/proj", [_user(1, "old")])
    _make_session(root, "300_p2", "/proj", [_user(2, "<user_input>new</user_input>")])
    _make_session(
        root, "400_x", "/elsewhere", [_user(3, "other")]
    )  # newest overall, wrong cwd
    reader = ClineReader("/proj", sessions_root=root)
    assert reader._active_dir().name == "300_p2"
    records = reader.read()
    assert records[0]["content"][0]["text"] == "<user_input>new</user_input>"
    assert records[0]["model"] == "m"  # attached from metadata


def test_reader_no_match_is_empty(tmp_path):
    root = tmp_path / "sessions"
    _make_session(root, "100_a", "/elsewhere", [_user(1, "x")])
    records = ClineReader("/proj", sessions_root=root).read()
    assert records == []


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("clineproj")
    return uri


def test_end_to_end_through_the_sink(mlflow_sqlite):
    import mlflow

    ids = MLflowSink(mlflow_sqlite, "clineproj").write(_translate())
    assert len(ids) == 2
    exp = mlflow.get_experiment_by_name("clineproj")
    traces = mlflow.search_traces(locations=[exp.experiment_id], return_type="list")
    assert len(traces) == 2

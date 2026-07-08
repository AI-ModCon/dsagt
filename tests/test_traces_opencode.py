"""Fixture tests for the opencode SQLite translator + reader."""

import json
import sqlite3

import pytest

from dsagt.observability import MLflowSink
from dsagt.traces import OpenCodeReader, OpenCodeTranslator


def _part(role, ts, data, model=None):
    return {"role": role, "ts": ts, "model": model, "data": data}


def _session():
    return [
        _part("user", 1000.0, {"type": "text", "text": "do the thing"}),  # turn 1
        _part("assistant", 1001.0, {"type": "step-start"}),  # skip
        _part("assistant", 1002.0, {"type": "text", "text": "Working."}, "m"),  # LLM
        _part(
            "assistant",
            1003.0,
            {
                "type": "tool",
                "tool": "shell",
                "callID": "c1",
                "state": {"input": {"cmd": "ls"}, "output": "a.txt"},
            },
        ),  # TOOL (call + result in one part)
        _part("assistant", 1004.0, {"type": "step-finish", "tokens": {}}),  # skip
        _part("assistant", 1005.0, {"type": "text", "text": "Done."}, "m"),  # LLM
        _part("user", 1006.0, {"type": "text", "text": "next"}),  # turn 2
        _part("assistant", 1007.0, {"type": "text", "text": "OK."}, "m"),  # LLM
    ]


def _translate(records=None):
    return OpenCodeTranslator().translate(
        records if records is not None else _session(),
        trace_id="oc",
        session_id="proj:s",
        project="ocproj",
    )


def test_two_turns_with_step_parts_skipped():
    trace = _translate()
    roots = [s for s in trace.spans if s["parent_id"] is None]
    assert len(roots) == 2
    assert roots[0]["attributes"]["prompt"] == "do the thing"
    assert roots[1]["attributes"]["prompt"] == "next"
    t1 = [
        (s["kind"], s["name"])
        for s in trace.spans
        if s["parent_id"] == roots[0]["span_id"]
    ]
    assert t1 == [
        ("LLM", "llm"),
        ("TOOL", "tool_shell"),
        ("LLM", "llm"),
    ]
    assert roots[0]["attributes"]["response"] == "Done."


def test_tool_call_and_result_from_one_part():
    tool = next(s for s in _translate().spans if s["kind"] == "TOOL")
    assert tool["attributes"]["input"] == {"cmd": "ls"}
    assert tool["attributes"]["result"] == "a.txt"


def test_model_carried_onto_llm_spans():
    llm = next(s for s in _translate().spans if s["kind"] == "LLM")
    assert llm["model"] == "m"


def test_empty_is_none():
    assert _translate([]) is None
    assert (
        _translate([_part("assistant", 1.0, {"type": "text", "text": "no prompt"})])
        is None
    )


# --- reader over a tiny opencode.db ---


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE session(id TEXT, directory TEXT, time_updated INTEGER)")
    con.execute("CREATE TABLE message(id TEXT, session_id TEXT, data TEXT)")
    con.execute(
        "CREATE TABLE part(id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, "
        "time_created INTEGER, data TEXT)"
    )
    con.execute("INSERT INTO session VALUES('s_new','/proj',200)")
    con.execute("INSERT INTO session VALUES('s_old','/proj',100)")
    con.execute("INSERT INTO session VALUES('s_other','/elsewhere',300)")
    con.executemany(
        "INSERT INTO message VALUES(?,?,?)",
        [
            ("m1", "s_new", json.dumps({"role": "user", "model": {"modelID": "x"}})),
            (
                "m2",
                "s_new",
                json.dumps({"role": "assistant", "model": {"modelID": "x"}}),
            ),
            ("mo", "s_old", json.dumps({"role": "user"})),
        ],
    )
    con.executemany(
        "INSERT INTO part VALUES(?,?,?,?,?)",
        [
            (
                "p1",
                "m1",
                "s_new",
                1777819990000,
                json.dumps({"type": "text", "text": "hi"}),
            ),
            (
                "p2",
                "m2",
                "s_new",
                1777819991000,
                json.dumps({"type": "text", "text": "hello"}),
            ),
            ("po", "mo", "s_old", 1, json.dumps({"type": "text", "text": "old"})),
        ],
    )
    con.commit()
    con.close()


def test_reader_picks_newest_session_and_converts_ms(tmp_path):
    db = tmp_path / "opencode.db"
    _make_db(db)
    parts = OpenCodeReader("/proj", db_path=db).read()
    assert [p["role"] for p in parts] == ["user", "assistant"]  # s_new only
    assert [p["data"]["text"] for p in parts] == ["hi", "hello"]
    assert parts[0]["ts"] == 1777819990.0  # ms → s
    assert parts[0]["model"] == "x"


def test_reader_missing_db_is_empty(tmp_path):
    parts = OpenCodeReader("/proj", db_path=tmp_path / "nope.db").read()
    assert parts == []


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ocproj")
    return uri


def test_end_to_end_through_the_sink(mlflow_sqlite):
    import mlflow

    ids = MLflowSink(mlflow_sqlite, "ocproj").write(_translate())
    assert len(ids) == 2
    exp = mlflow.get_experiment_by_name("ocproj")
    traces = mlflow.search_traces(
        locations=[exp.experiment_id], return_type="list"
    )
    assert len(traces) == 2

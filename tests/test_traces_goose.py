"""Fixture tests for the Goose SQLite translator + reader.

Translator records mirror what ``GooseReader`` yields (``{role, content, ts}``
with parsed content_json blocks); the reader test builds a tiny sessions.db.
"""

import sqlite3

import pytest

from dsagt.observability import MLflowSink
from dsagt.traces import GooseReader, GooseTranslator


def _utext(ts, text):
    return {"role": "user", "ts": ts, "content": [{"type": "text", "text": text}]}


def _atext(ts, text):
    return {"role": "assistant", "ts": ts, "content": [{"type": "text", "text": text}]}


def _atool(ts, name, args, tid):
    return {
        "role": "assistant",
        "ts": ts,
        "content": [
            {
                "type": "toolRequest",
                "id": tid,
                "toolCall": {"value": {"name": name, "arguments": args}},
            }
        ],
    }


def _uresp(ts, tid, text):
    return {
        "role": "user",
        "ts": ts,
        "content": [
            {
                "type": "toolResponse",
                "id": tid,
                "toolResult": {"value": {"content": [{"type": "text", "text": text}]}},
            }
        ],
    }


def _session():
    return [
        _utext(1000, "do the thing"),  # turn 1 prompt
        _atext(1001, "Let me check."),  # LLM
        _atool(1002, "shell", {"command": "ls"}, "t1"),  # TOOL
        _uresp(1003, "t1", "a.txt\nb.txt"),  # result (user msg, not a prompt)
        _atext(1004, "Done."),  # LLM
        _utext(1005, "now another"),  # turn 2 prompt
        _atext(1006, "OK."),  # LLM
    ]


def _translate(records=None):
    return GooseTranslator().translate(
        records if records is not None else _session(),
        trace_id="g",
        session_id="proj:s",
        project="gooseproj",
    )


def test_turns_segment_on_user_text_not_tool_responses():
    roots = [s for s in _translate().spans if s["parent_id"] is None]
    assert len(roots) == 2  # the toolResponse user message is NOT a turn
    assert roots[0]["attributes"]["prompt"] == "do the thing"
    assert roots[1]["attributes"]["prompt"] == "now another"


def test_span_layout_and_tool_extraction():
    trace = _translate()
    roots = [s for s in trace.spans if s["parent_id"] is None]
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
    tool = next(s for s in trace.spans if s["kind"] == "TOOL")
    assert tool["attributes"]["input"] == {"command": "ls"}
    assert (
        tool["attributes"]["result"] == "a.txt\nb.txt"
    )  # toolResult.value.content text
    assert roots[0]["attributes"]["response"] == "Done."


def test_empty_is_none():
    assert _translate([]) is None
    assert _translate([_atext(1, "no user prompt")]) is None


# --- reader over a tiny sessions.db ---


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sessions(id TEXT, working_dir TEXT, updated_at INTEGER)")
    con.execute(
        "CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content_json TEXT, created_timestamp INTEGER)"
    )
    con.execute("INSERT INTO sessions VALUES('s_old', '/proj', 100)")
    con.execute(
        "INSERT INTO sessions VALUES('s_new', '/proj', 200)"
    )  # newest for /proj
    con.execute("INSERT INTO sessions VALUES('s_other', '/elsewhere', 300)")
    rows = [
        ("s_new", "user", '[{"type":"text","text":"hi"}]', 1000),
        ("s_new", "assistant", '[{"type":"text","text":"hello"}]', 1001),
        ("s_old", "user", '[{"type":"text","text":"old session"}]', 50),
        ("s_other", "user", '[{"type":"text","text":"other dir"}]', 5),
    ]
    con.executemany(
        "INSERT INTO messages(session_id, role, content_json, created_timestamp) "
        "VALUES(?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def test_reader_picks_newest_session_for_the_project_dir(tmp_path):
    db = tmp_path / "sessions.db"
    _make_db(db)
    records = GooseReader("/proj", db_path=db).read()
    assert [r["content"][0]["text"] for r in records] == ["hi", "hello"]  # s_new only


def test_reader_missing_db_is_empty(tmp_path):
    records = GooseReader("/proj", db_path=tmp_path / "nope.db").read()
    assert records == []


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("gooseproj")
    return uri


def test_end_to_end_through_the_sink(mlflow_sqlite):
    import mlflow

    ids = MLflowSink(mlflow_sqlite, "gooseproj").write(_translate())
    assert len(ids) == 2
    exp = mlflow.get_experiment_by_name("gooseproj")
    traces = mlflow.search_traces(
        locations=[exp.experiment_id], return_type="list"
    )
    assert len(traces) == 2

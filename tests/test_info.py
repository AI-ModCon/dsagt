"""
Unit tests for ``dsagt info`` reporting logic.

Feeds a synthetic traces DataFrame into ``_report()`` so we can assert the
grouping, token sums, and error surfacing without spinning up MLflow.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dsagt.commands.info import _report, _tokens, _fmt_count, _is_error


def _metadata(*, session: str, source: str | None, agent: str | None,
              in_t: int, out_t: int) -> dict:
    """Build a trace_metadata dict in MLflow's on-disk shape."""
    md = {"mlflow.trace.session": session}
    if source is not None:
        md["dsagt.source"] = source
    if agent is not None:
        md["dsagt.agent"] = agent
    if in_t or out_t:
        md["mlflow.trace.tokenUsage"] = json.dumps({
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_tokens": in_t + out_t,
        })
    return md


def _traces_df(rows: list[dict]) -> pd.DataFrame:
    """Assemble a DataFrame with the MLflow search_traces column shape.

    We only populate the columns ``_report`` actually reads; the rest stays
    absent so an accidental new dependency on another column shows up as a
    KeyError in a test instead of silently reading null production data.
    """
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _tokens / _fmt_count / _is_error primitives
# ---------------------------------------------------------------------------

def test_tokens_missing_returns_zeros():
    assert _tokens({}) == (0, 0)


def test_tokens_malformed_json_returns_zeros():
    assert _tokens({"mlflow.trace.tokenUsage": "not-json"}) == (0, 0)


def test_tokens_extracts_input_output():
    md = {"mlflow.trace.tokenUsage": '{"input_tokens": 123, "output_tokens": 45}'}
    assert _tokens(md) == (123, 45)


@pytest.mark.parametrize("n,expected", [
    (0, "0"), (999, "999"), (1000, "1.0k"),
    (12345, "12.3k"), (1_500_000, "1.5M"),
])
def test_fmt_count(n, expected):
    assert _fmt_count(n) == expected


@pytest.mark.parametrize("state,expected", [
    ("OK", False), ("ERROR", True),
    ("TraceState.OK", False), ("TraceState.ERROR", True),
])
def test_is_error_handles_enum_reprs(state, expected):
    assert _is_error(state) is expected


# ---------------------------------------------------------------------------
# _report end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "agent": "claude-code",
        "llm": {"model": "claude-haiku-test"},
    }


def test_report_empty_traces(config):
    r = _report("proj", config, None)
    assert r["total_traces"] == 0
    assert r["by_source"] == []
    assert r["by_session"] == []
    assert r["errors"] == []
    assert r["agent"] == "claude-code"
    assert r["model"] == "claude-haiku-test"


def test_report_aggregates_by_source_and_session(config):
    df = _traces_df([
        {
            "trace_id": "t1",
            "state": "OK",
            "request_time": 100,
            "trace_metadata": _metadata(
                session="sess-A", source="agent", agent="goose",
                in_t=1000, out_t=100,
            ),
        },
        {
            "trace_id": "t2",
            "state": "OK",
            "request_time": 200,
            "trace_metadata": _metadata(
                session="sess-A", source="agent", agent="goose",
                in_t=500, out_t=50,
            ),
        },
        {
            "trace_id": "t3",
            "state": "OK",
            "request_time": 150,
            "trace_metadata": _metadata(
                session="sess-A", source="embedding", agent="goose",
                in_t=200, out_t=0,
            ),
        },
        {
            "trace_id": "t4",
            "state": "ERROR",
            "request_time": 300,
            "trace_metadata": _metadata(
                session="sess-B", source="agent", agent="claude-code",
                in_t=800, out_t=20,
            ),
        },
    ])

    r = _report("proj", config, df)

    assert r["total_traces"] == 4
    assert r["total_errors"] == 1
    assert r["input_tokens"] == 2500
    assert r["output_tokens"] == 170

    # By source: agent (3 traces) and embedding (1 trace).
    sources = {row["source"]: row for row in r["by_source"]}
    assert sources["agent"]["traces"] == 3
    assert sources["agent"]["input_tokens"] == 2300
    assert sources["agent"]["output_tokens"] == 170
    assert sources["agent"]["errors"] == 1
    assert sources["embedding"]["traces"] == 1
    assert sources["embedding"]["errors"] == 0

    # By session: sess-B has a later request_time, so it lands first.
    assert [s["session"] for s in r["by_session"]] == ["sess-B", "sess-A"]
    sess_a = next(s for s in r["by_session"] if s["session"] == "sess-A")
    assert sess_a["traces"] == 3
    assert sess_a["input_tokens"] == 1700
    assert sess_a["agent"] == "goose"

    # Errors surface the triage-relevant fields.
    assert len(r["errors"]) == 1
    err = r["errors"][0]
    assert err["session"] == "sess-B"
    assert err["source"] == "agent"
    assert err["trace_id"] == "t4"


def test_report_missing_source_falls_back_to_unknown(config):
    df = _traces_df([
        {
            "trace_id": "t1",
            "state": "OK",
            "request_time": 100,
            "trace_metadata": _metadata(
                session="sess-A", source=None, agent=None,
                in_t=0, out_t=0,
            ),
        },
    ])
    r = _report("proj", config, df)
    assert r["by_source"][0]["source"] == "unknown"
    assert r["by_session"][0]["agent"] == "-"

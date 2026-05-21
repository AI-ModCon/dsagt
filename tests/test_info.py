"""
Unit tests for ``dsagt info`` reporting logic.

Feeds a synthetic traces DataFrame into ``_report()`` so we can assert the
grouping, token sums, and error surfacing without spinning up MLflow.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dsagt.commands.info import (
    _config_sources,
    _fmt_count,
    _is_error,
    _mask_secret,
    _report,
    _tokens,
)


def _metadata(*, session: str, agent: str | None,
              in_t: int, out_t: int) -> dict:
    """Build a trace_metadata dict in MLflow's on-disk shape."""
    md = {"mlflow.trace.session": session}
    if agent is not None:
        md["dsagt.agent"] = agent
    if in_t or out_t:
        md["mlflow.trace.tokenUsage"] = json.dumps({
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_tokens": in_t + out_t,
        })
    return md


def _spans_for(service_name: str | None) -> list:
    """Build the ``spans`` column entry: one root span carrying service.name.

    ``_source_from_spans`` reads ``service.name`` off the root span's
    attributes (MLflow's OTLP receiver flows the OTel resource attribute
    through to span attributes), so a single SimpleNamespace is enough
    for the report-side test.  ``None`` → empty spans list, source falls
    through to ``"unknown"``.
    """
    from types import SimpleNamespace
    if service_name is None:
        return []
    return [SimpleNamespace(attributes={"service.name": service_name})]


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
        "agent": "claude",
        "llm": {"model": "claude-haiku-test"},
    }


def test_report_empty_traces(config):
    r = _report("proj", config, None)
    assert r["total_traces"] == 0
    assert r["by_source"] == []
    assert r["by_session"] == []
    assert r["errors"] == []
    assert r["agent"] == "claude"
    assert r["model"] == "claude-haiku-test"


def test_report_aggregates_by_source_and_session(config):
    """Source bucketing comes from the root span's ``service.name``.

    Three claude-code agent traces (one of which errored) + one
    knowledge-server trace, across two sessions.  Sums + bucket counts
    match expected totals.
    """
    df = _traces_df([
        {
            "trace_id": "t1",
            "state": "OK",
            "request_time": 100,
            "trace_metadata": _metadata(
                session="sess-A", agent="claude", in_t=1000, out_t=100,
            ),
            "spans": _spans_for("claude-code"),
        },
        {
            "trace_id": "t2",
            "state": "OK",
            "request_time": 200,
            "trace_metadata": _metadata(
                session="sess-A", agent="claude", in_t=500, out_t=50,
            ),
            "spans": _spans_for("claude-code"),
        },
        {
            "trace_id": "t3",
            "state": "OK",
            "request_time": 150,
            "trace_metadata": _metadata(
                session="sess-A", agent="claude", in_t=200, out_t=0,
            ),
            "spans": _spans_for("dsagt-knowledge-server"),
        },
        {
            "trace_id": "t4",
            "state": "ERROR",
            "request_time": 300,
            "trace_metadata": _metadata(
                session="sess-B", agent="claude", in_t=800, out_t=20,
            ),
            "spans": _spans_for("claude-code"),
        },
    ])

    r = _report("proj", config, df)

    assert r["total_traces"] == 4
    assert r["total_errors"] == 1
    assert r["input_tokens"] == 2500
    assert r["output_tokens"] == 170

    sources = {row["source"]: row for row in r["by_source"]}
    assert sources["claude-code"]["traces"] == 3
    assert sources["claude-code"]["input_tokens"] == 2300
    assert sources["claude-code"]["output_tokens"] == 170
    assert sources["claude-code"]["errors"] == 1
    assert sources["dsagt-knowledge-server"]["traces"] == 1
    assert sources["dsagt-knowledge-server"]["errors"] == 0

    assert [s["session"] for s in r["by_session"]] == ["sess-B", "sess-A"]
    sess_a = next(s for s in r["by_session"] if s["session"] == "sess-A")
    assert sess_a["traces"] == 3
    assert sess_a["input_tokens"] == 1700
    assert sess_a["agent"] == "claude"

    assert len(r["errors"]) == 1
    err = r["errors"][0]
    assert err["session"] == "sess-B"
    assert err["source"] == "claude-code"
    assert err["trace_id"] == "t4"


def test_report_missing_source_falls_back_to_unknown(config):
    """No spans column / no service.name → bucket is "unknown"."""
    df = _traces_df([
        {
            "trace_id": "t1",
            "state": "OK",
            "request_time": 100,
            "trace_metadata": _metadata(
                session="sess-A", agent=None, in_t=0, out_t=0,
            ),
            "spans": _spans_for(None),
        },
    ])
    r = _report("proj", config, df)
    assert r["by_source"][0]["source"] == "unknown"
    assert r["by_session"][0]["agent"] == "-"


# ---------------------------------------------------------------------------
# Config source tracking (config / shell / unresolved)
# ---------------------------------------------------------------------------

def test_mask_secret_short_value():
    assert _mask_secret("short") == "***"
    assert _mask_secret("exactlytwelv") == "***"


def test_mask_secret_long_value():
    assert _mask_secret("sk-abcdefghijklmnop") == "sk-a...mnop"


def _write_project(tmp_path, monkeypatch, raw_yaml: str):
    """Register a temp project with the given dsagt_config.yaml content."""
    import yaml as _yaml
    from dsagt.session import register_project

    pdir = tmp_path / "proj"
    pdir.mkdir()
    (pdir / "dsagt_config.yaml").write_text(raw_yaml)

    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    monkeypatch.setattr("dsagt.session.REGISTRY_DIR", registry_dir)
    monkeypatch.setattr("dsagt.session.REGISTRY_FILE", registry_dir / "projects.yaml")
    register_project("proj", pdir)
    return pdir


def test_config_sources_classifies_shell(tmp_path, monkeypatch):
    """${VAR} resolves from os.environ — that's the only source for
    user-provided values now (no .env file is consulted)."""
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  provider: ${LLM_PROVIDER}\n  model: ${LLM_MODEL}\n")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")

    rows = {r["path"]: r for r in _config_sources("proj")}
    assert rows["llm.provider"] == {"path": "llm.provider", "value": "openai", "source": "shell"}
    assert rows["llm.model"]["value"] == "gpt-4"
    assert rows["llm.model"]["source"] == "shell"


def test_config_sources_classifies_unresolved(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  provider: ${LLM_PROVIDER}\n")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    rows = {r["path"]: r for r in _config_sources("proj")}
    assert rows["llm.provider"]["source"] == "unresolved"
    assert rows["llm.provider"]["value"] == "${LLM_PROVIDER}"


def test_config_sources_classifies_literal(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nproxy:\n  port: 4000\n")

    rows = {r["path"]: r for r in _config_sources("proj")}
    assert rows["proxy.port"] == {"path": "proxy.port", "value": "4000", "source": "config"}


def test_config_sources_masks_api_key(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  api_key: ${LLM_API_KEY}\n")
    monkeypatch.setenv("LLM_API_KEY", "sk-1234567890abcdef")

    rows = {r["path"]: r for r in _config_sources("proj")}
    assert rows["llm.api_key"]["value"] == "sk-1...cdef"
    assert rows["llm.api_key"]["source"] == "shell"


def test_config_sources_skips_internal_sections(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nknowledge:\n  chunk_size: 1024\nextraction:\n  threshold: 0\ncategories:\n  qc: stuff\n")

    paths = {r["path"] for r in _config_sources("proj")}
    assert "knowledge.chunk_size" not in paths
    assert "extraction.threshold" not in paths
    assert "categories.qc" not in paths
    assert "project" in paths


# ---------------------------------------------------------------------------
# _kb_collections — read chunks.jsonl, count entries, break down by source
# ---------------------------------------------------------------------------

def _write_chunk(path, text="x", source=None):
    """Append one chunks.jsonl line with optional metadata.source."""
    md = {"chunk_index": 0}
    if source is not None:
        md["source"] = source
    with path.open("a") as f:
        f.write(json.dumps({"text": text, "metadata": md}) + "\n")


def test_kb_collections_empty_when_no_kb_index(tmp_path):
    from dsagt.commands.info import _kb_collections
    assert _kb_collections(tmp_path) == []


def test_kb_collections_counts_chunks_and_breaks_down_sources(tmp_path):
    from dsagt.commands.info import _kb_collections

    kb = tmp_path / "kb_index"
    (kb / "tools").mkdir(parents=True)
    (kb / "skills").mkdir(parents=True)
    (kb / "research").mkdir(parents=True)

    # Tools: 2 bundled, 1 project
    for _ in range(2):
        _write_chunk(kb / "tools" / "chunks.jsonl", source="bundled")
    _write_chunk(kb / "tools" / "chunks.jsonl", source="project")

    # Skills: 1 bundled
    _write_chunk(kb / "skills" / "chunks.jsonl", source="bundled")

    # Research: 3 chunks, no source field
    for _ in range(3):
        _write_chunk(kb / "research" / "chunks.jsonl")

    rows = {r["collection"]: r for r in _kb_collections(tmp_path)}
    assert rows["tools"] == {
        "collection": "tools",
        "chunks": 3,
        "by_source": {"bundled": 2, "project": 1},
    }
    assert rows["skills"]["chunks"] == 1
    assert rows["skills"]["by_source"] == {"bundled": 1}
    assert rows["research"]["chunks"] == 3
    assert rows["research"]["by_source"] == {}


def test_kb_collections_skips_dirs_without_chunks_jsonl(tmp_path):
    from dsagt.commands.info import _kb_collections

    kb = tmp_path / "kb_index"
    (kb / "no_chunks").mkdir(parents=True)
    (kb / "no_chunks" / "route.json").write_text("{}")
    assert _kb_collections(tmp_path) == []


# ---------------------------------------------------------------------------
# _kb_retrieval — pull kb.search spans out of a synthetic traces frame
# ---------------------------------------------------------------------------

def _trace_with_kb_spans(session: str, hits_per_search: list[int]) -> dict:
    """Build a single trace row carrying one kb.search span per hits entry."""
    from types import SimpleNamespace
    spans = [
        SimpleNamespace(name="kb.search", attributes={"hits": h})
        for h in hits_per_search
    ]
    # Stick a non-kb span in there too so the filter is exercised.
    spans.append(SimpleNamespace(name="kb.embed", attributes={}))
    return {
        "trace_id": f"t-{session}",
        "state": "OK",
        "request_time": 1,
        "trace_metadata": {"mlflow.trace.session": session},
        "spans": spans,
    }


def test_kb_retrieval_empty_when_no_traces():
    from dsagt.commands.info import _kb_retrieval
    assert _kb_retrieval(None) == []
    assert _kb_retrieval(pd.DataFrame()) == []


def test_kb_retrieval_aggregates_searches_and_hits_by_session():
    from dsagt.commands.info import _kb_retrieval

    df = pd.DataFrame([
        _trace_with_kb_spans("sess-A", hits_per_search=[3, 5]),
        _trace_with_kb_spans("sess-A", hits_per_search=[2]),
        _trace_with_kb_spans("sess-B", hits_per_search=[7]),
    ])
    rows = {r["session"]: r for r in _kb_retrieval(df)}
    assert rows["sess-A"] == {"session": "sess-A", "searches": 3, "hits": 10}
    assert rows["sess-B"] == {"session": "sess-B", "searches": 1, "hits": 7}


def test_kb_retrieval_handles_missing_session():
    from dsagt.commands.info import _kb_retrieval
    from types import SimpleNamespace

    df = pd.DataFrame([{
        "trace_id": "t1",
        "state": "OK",
        "request_time": 1,
        "trace_metadata": {},  # no mlflow.trace.session
        "spans": [SimpleNamespace(name="kb.search", attributes={"hits": 4})],
    }])
    rows = _kb_retrieval(df)
    assert len(rows) == 1
    assert rows[0]["session"] == "(no-session)"
    assert rows[0]["hits"] == 4


def test_kb_retrieval_ignores_non_kb_spans():
    from dsagt.commands.info import _kb_retrieval
    from types import SimpleNamespace

    df = pd.DataFrame([{
        "trace_id": "t1",
        "state": "OK",
        "request_time": 1,
        "trace_metadata": {"mlflow.trace.session": "s"},
        "spans": [
            SimpleNamespace(name="kb.embed", attributes={}),
            SimpleNamespace(name="registry.save_tool_spec", attributes={}),
        ],
    }])
    assert _kb_retrieval(df) == []


# ---------------------------------------------------------------------------
# _project_created — best-effort project-start date
# ---------------------------------------------------------------------------

def test_project_created_returns_iso_date_for_existing_dir(tmp_path):
    from dsagt.commands.info import _project_created

    pdir = tmp_path / "proj"
    pdir.mkdir()
    out = _project_created(pdir)
    assert out is not None
    # YYYY-MM-DD format.
    assert len(out) == 10 and out[4] == "-" and out[7] == "-"


def test_project_created_returns_none_when_dir_missing(tmp_path):
    from dsagt.commands.info import _project_created
    assert _project_created(tmp_path / "does-not-exist") is None

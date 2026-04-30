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
    _read_env_file,
    _report,
    _tokens,
)


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
                session="sess-B", source="agent", agent="claude",
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


# ---------------------------------------------------------------------------
# Config source tracking (.env / environment / literal)
# ---------------------------------------------------------------------------

def test_read_env_file_skips_comments_and_blanks(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# header\n\nA=1\n  B = 2 \n# C=ignored\n")
    assert _read_env_file(p) == {"A": "1", "B": "2"}


def test_read_env_file_missing_returns_empty(tmp_path):
    assert _read_env_file(tmp_path / "missing.env") == {}


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


def test_config_sources_classifies_env_file(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  provider: ${LLM_PROVIDER}\n  model: ${LLM_MODEL}\n")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openai\nLLM_MODEL=gpt-4\n")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    rows = {r["path"]: r for r in _config_sources("proj", env_file)}
    assert rows["llm.provider"] == {"path": "llm.provider", "value": "openai", "source": ".env"}
    assert rows["llm.model"]["value"] == "gpt-4"
    assert rows["llm.model"]["source"] == ".env"


def test_config_sources_classifies_environment(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  provider: ${LLM_PROVIDER}\n")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    rows = {r["path"]: r for r in _config_sources("proj", tmp_path / "missing.env")}
    assert rows["llm.provider"]["value"] == "anthropic"
    assert rows["llm.provider"]["source"] == "environment"


def test_config_sources_classifies_unresolved(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  provider: ${LLM_PROVIDER}\n")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    rows = {r["path"]: r for r in _config_sources("proj", tmp_path / "missing.env")}
    assert rows["llm.provider"]["source"] == "unresolved"
    assert rows["llm.provider"]["value"] == "${LLM_PROVIDER}"


def test_config_sources_classifies_literal(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nproxy:\n  port: 4000\n")

    rows = {r["path"]: r for r in _config_sources("proj", tmp_path / "missing.env")}
    assert rows["proxy.port"] == {"path": "proxy.port", "value": "4000", "source": "config"}


def test_config_sources_masks_api_key(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nllm:\n  api_key: ${LLM_API_KEY}\n")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=sk-1234567890abcdef\n")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    rows = {r["path"]: r for r in _config_sources("proj", env_file)}
    assert rows["llm.api_key"]["value"] == "sk-1...cdef"
    assert rows["llm.api_key"]["source"] == ".env"


def test_config_sources_skips_internal_sections(tmp_path, monkeypatch):
    _write_project(tmp_path, monkeypatch,
        "project: proj\nagent: goose\nknowledge:\n  chunk_size: 1024\nextraction:\n  threshold: 0\ncategories:\n  qc: stuff\n")

    paths = {r["path"] for r in _config_sources("proj", tmp_path / "missing.env")}
    assert "knowledge.chunk_size" not in paths
    assert "extraction.threshold" not in paths
    assert "categories.qc" not in paths
    assert "project" in paths

"""
Tests for sidechannel-model detection and reporting.

Two layers:

- ``_record_sidechannel`` (provenance.py) — writes a JSONL entry when a
  proxy request's model doesn't match ``DSAGT_PRIMARY_MODEL``.
- ``_print_sidechannel_warning`` (cli.py) — reads the JSONL at session
  teardown, dedups by model within the current session, and prints a
  yellow terminal warning pointing at the README.
"""

from __future__ import annotations

import json

import pytest

from dsagt.sidechannel import (
    LOG_FILENAME as _SIDECHANNEL_LOG,
    PRIMARY_MODEL_ENV,
    print_warning as _print_sidechannel_warning,
    record as _record_sidechannel,
)


# ---------------------------------------------------------------------------
# _record_sidechannel
# ---------------------------------------------------------------------------

@pytest.fixture
def records_dir(tmp_path):
    d = tmp_path / "trace_archive"
    d.mkdir()
    return d


def _log_path(records_dir):
    return records_dir.parent / _SIDECHANNEL_LOG


def _kwargs(*, client_model: str | None = None, routed_model: str | None = None) -> dict:
    """Mimic the kwargs shape LiteLLM passes to a success callback.

    - ``standard_logging_object.model_group``: what the client sent
    - ``model``: what LiteLLM routed to (wildcard's litellm_params.model
      for sidechannel hits, or the configured primary for exact matches)
    """
    kw: dict = {}
    if routed_model is not None:
        kw["model"] = routed_model
    if client_model is not None:
        kw["standard_logging_object"] = {"model_group": client_model}
    return kw


def test_record_logs_client_name_not_routing_target(records_dir, monkeypatch):
    """The warning must show the model the AGENT asked for (sidechannel
    name), not the proxy's internal catchall route target.

    This is the regression test for the bug where ``sidechannel.jsonl``
    showed ``dsagt-sidechannel-catchall`` — useless for the user — instead
    of ``gpt-4o-mini``/``claude-haiku-4-5-20251001``.
    """
    monkeypatch.setenv(PRIMARY_MODEL_ENV, "haiku-v1-project")
    monkeypatch.setenv("DSAGT_AGENT", "claude-code")
    monkeypatch.setenv("DSAGT_SESSION_ID", "sess-1")

    _record_sidechannel(records_dir, _kwargs(
        client_model="claude-haiku-4-5-20251001",
        routed_model="dsagt-sidechannel-catchall",
    ))

    entries = [json.loads(l) for l in _log_path(records_dir).read_text().splitlines()]
    assert len(entries) == 1
    e = entries[0]
    assert e["model"] == "claude-haiku-4-5-20251001", (
        "must record the agent-requested name, not the proxy's catchall target"
    )
    assert e["agent"] == "claude-code"
    assert e["session"] == "sess-1"
    assert e["timestamp"].endswith("Z")


def test_record_strips_provider_prefix(records_dir, monkeypatch):
    monkeypatch.setenv(PRIMARY_MODEL_ENV, "haiku-v1-project")
    _record_sidechannel(records_dir, _kwargs(
        client_model="openai/some-sidechannel",
        routed_model="dsagt-sidechannel-catchall",
    ))
    entries = [json.loads(l) for l in _log_path(records_dir).read_text().splitlines()]
    assert entries[0]["model"] == "some-sidechannel"


def test_record_match_does_not_write(records_dir, monkeypatch):
    """Primary model matches → no sidechannel entry."""
    monkeypatch.setenv(PRIMARY_MODEL_ENV, "haiku-v1-project")
    _record_sidechannel(records_dir, _kwargs(
        client_model="haiku-v1-project",
        routed_model="haiku-v1-project",
    ))
    assert not _log_path(records_dir).exists()


def test_record_without_primary_skipped(records_dir, monkeypatch):
    """No DSAGT_PRIMARY_MODEL (e.g. direct-callback tests) → skip silently."""
    monkeypatch.delenv(PRIMARY_MODEL_ENV, raising=False)
    _record_sidechannel(records_dir, _kwargs(client_model="anything"))
    assert not _log_path(records_dir).exists()


def test_record_falls_back_to_kwargs_model(records_dir, monkeypatch):
    """When standard_logging_object isn't populated, fall back to kwargs.model.

    Direct-callback use from tests / synthetic paths that don't go through
    the full LiteLLM routing won't have slo.model_group set — the function
    should still log something useful rather than silently skip.
    """
    monkeypatch.setenv(PRIMARY_MODEL_ENV, "haiku-v1-project")
    _record_sidechannel(records_dir, {"model": "some-other-model"})
    entries = [json.loads(l) for l in _log_path(records_dir).read_text().splitlines()]
    assert entries[0]["model"] == "some-other-model"


def test_record_empty_model_skipped(records_dir, monkeypatch):
    monkeypatch.setenv(PRIMARY_MODEL_ENV, "haiku-v1-project")
    _record_sidechannel(records_dir, {"model": ""})
    _record_sidechannel(records_dir, {})
    assert not _log_path(records_dir).exists()


# ---------------------------------------------------------------------------
# _print_sidechannel_warning
# ---------------------------------------------------------------------------

def _write_log(pdir, entries):
    pdir.mkdir(exist_ok=True)
    (pdir / _SIDECHANNEL_LOG).write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_print_noop_without_log(tmp_path, capsys):
    _print_sidechannel_warning(tmp_path, "sess-1")
    assert capsys.readouterr().out == ""


def test_print_dedups_and_counts(tmp_path, capsys):
    _write_log(tmp_path, [
        {"model": "haiku-X", "session": "sess-1", "agent": "claude-code", "timestamp": "t1"},
        {"model": "haiku-X", "session": "sess-1", "agent": "claude-code", "timestamp": "t2"},
        {"model": "gpt-mini", "session": "sess-1", "agent": "goose", "timestamp": "t3"},
    ])
    _print_sidechannel_warning(tmp_path, "sess-1")
    out = capsys.readouterr().out
    assert "haiku-X" in out
    assert "(2 calls)" in out
    assert "gpt-mini" in out
    assert "(1 call)" in out  # singular
    assert "README" in out


def test_print_filters_to_current_session(tmp_path, capsys):
    """Entries from previous sessions must not leak into this session's warning.

    The log is append-only across runs; dedup happens per-session so the
    user only sees models that fired in the just-finished session.
    """
    _write_log(tmp_path, [
        {"model": "old-model", "session": "sess-OLD", "agent": "x", "timestamp": "t0"},
        {"model": "new-model", "session": "sess-NEW", "agent": "x", "timestamp": "t1"},
    ])
    _print_sidechannel_warning(tmp_path, "sess-NEW")
    out = capsys.readouterr().out
    assert "new-model" in out
    assert "old-model" not in out


def test_print_handles_malformed_line(tmp_path, capsys):
    """A corrupt JSON line shouldn't crash session teardown."""
    (tmp_path / _SIDECHANNEL_LOG).write_text("not-json\n")
    # Must not raise; may or may not print anything.
    _print_sidechannel_warning(tmp_path, "sess-1")

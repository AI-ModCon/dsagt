"""
Tests for the Phase 2 proxy callback (cache breakpoints + sidechannel
detection) in ``observability.py``.  The callback is the only piece
LiteLLM autolog can't cover — request mutation (cache markers) and
canned-response detection (sidechannel) need the proxy hot path.

LiteLLM's standard ``"otel"`` callback handles trace transport;
``DSAGTCallback`` only intercepts for these two intercept-time concerns.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _inject_cache_breakpoints
# ---------------------------------------------------------------------------


class TestInjectCacheBreakpoints:
    """Anthropic prompt caching keys on the prefix UP TO each marked
    block.  We mark the last system text block + last tool definition
    so subsequent turns within the 5-min TTL pay 10% on the cached
    prefix.  Providers without caching ignore the marker as a no-op."""

    def test_marks_last_tool_definition(self):
        from dsagt.observability import _inject_cache_breakpoints
        messages: list = []
        kwargs = {
            "tools": [
                {"name": "tool_a", "description": "a"},
                {"name": "tool_b", "description": "b"},
            ],
        }
        _inject_cache_breakpoints(messages, kwargs)
        assert "cache_control" not in kwargs["tools"][0]
        assert kwargs["tools"][1]["cache_control"] == {"type": "ephemeral"}

    def test_promotes_system_string_to_block_with_marker(self):
        from dsagt.observability import _inject_cache_breakpoints
        messages = [{"role": "system", "content": "you are helpful"}]
        kwargs = {}
        _inject_cache_breakpoints(messages, kwargs)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "you are helpful"
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_marks_last_system_block(self):
        from dsagt.observability import _inject_cache_breakpoints
        messages = [{"role": "system", "content": [
            {"type": "text", "text": "rule 1"},
            {"type": "text", "text": "rule 2"},
        ]}]
        kwargs = {}
        _inject_cache_breakpoints(messages, kwargs)
        blocks = messages[0]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_no_tools_no_system_is_noop(self):
        from dsagt.observability import _inject_cache_breakpoints
        messages = [{"role": "user", "content": "hi"}]
        kwargs = {}
        _inject_cache_breakpoints(messages, kwargs)
        assert "tools" not in kwargs
        assert messages[0]["content"] == "hi"

    def test_only_first_system_message_marked(self):
        """If multiple system messages somehow exist, only the first
        gets stamped (loop breaks after first match)."""
        from dsagt.observability import _inject_cache_breakpoints
        messages = [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
        ]
        _inject_cache_breakpoints(messages, {})
        assert isinstance(messages[0]["content"], list)
        assert messages[1]["content"] == "second"  # untouched


# ---------------------------------------------------------------------------
# record_sidechannel_call / print_sidechannel_warning
# ---------------------------------------------------------------------------


class TestSidechannelDetection:
    """``record_sidechannel_call`` writes to ``sidechannel.jsonl`` only
    when ``kwargs["model"]`` matches ``SIDECHANNEL_CATCHALL_MODEL`` —
    that's the wildcard's post-routing target, the only reliable
    discriminator from primary / alias hits."""

    def _kwargs(self, routed_to: str, requested: str | None = "gpt-4o-mini"):
        slo = {}
        if requested:
            slo["model_group"] = f"openai/{requested}"
        return {"model": routed_to, "standard_logging_object": slo}

    def test_records_when_routed_to_catchall(self, tmp_path, monkeypatch):
        from dsagt.observability import (
            record_sidechannel_call, SIDECHANNEL_CATCHALL_MODEL,
            SIDECHANNEL_LOG_FILENAME,
        )
        monkeypatch.setenv("DSAGT_PRIMARY_MODEL", "real-model")
        records_dir = tmp_path / "trace_archive"
        records_dir.mkdir()
        record_sidechannel_call(records_dir, self._kwargs(SIDECHANNEL_CATCHALL_MODEL))
        log_path = tmp_path / SIDECHANNEL_LOG_FILENAME
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["model"] == "gpt-4o-mini"

    def test_skipped_when_routed_to_primary(self, tmp_path, monkeypatch):
        """Real upstream calls (primary entry or alias) shouldn't log —
        only true mock hits do."""
        from dsagt.observability import (
            record_sidechannel_call, SIDECHANNEL_LOG_FILENAME,
        )
        monkeypatch.setenv("DSAGT_PRIMARY_MODEL", "real-model")
        records_dir = tmp_path / "trace_archive"
        records_dir.mkdir()
        record_sidechannel_call(records_dir, self._kwargs("openai/real-model"))
        log_path = tmp_path / SIDECHANNEL_LOG_FILENAME
        assert not log_path.exists()

    def test_no_op_when_primary_env_unset(self, tmp_path, monkeypatch):
        """If the proxy never set DSAGT_PRIMARY_MODEL we don't know what
        to compare against — refuse to log rather than misclassify."""
        from dsagt.observability import (
            record_sidechannel_call, SIDECHANNEL_CATCHALL_MODEL,
            SIDECHANNEL_LOG_FILENAME,
        )
        monkeypatch.delenv("DSAGT_PRIMARY_MODEL", raising=False)
        records_dir = tmp_path / "trace_archive"
        records_dir.mkdir()
        record_sidechannel_call(records_dir, self._kwargs(SIDECHANNEL_CATCHALL_MODEL))
        assert not (tmp_path / SIDECHANNEL_LOG_FILENAME).exists()


class TestPrintSidechannelWarning:
    """End-of-session warning: dedups by model name within the current
    session, only emits ANSI on a TTY, says nothing when no entries
    matched (most common case)."""

    def _write_log(self, project_dir: Path, entries: list[dict]):
        from dsagt.observability import SIDECHANNEL_LOG_FILENAME
        path = project_dir / SIDECHANNEL_LOG_FILENAME
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    def test_silent_when_no_log(self, tmp_path, capsys):
        from dsagt.observability import print_sidechannel_warning
        print_sidechannel_warning(tmp_path, "any-session")
        assert capsys.readouterr().out == ""

    def test_silent_when_session_doesnt_match(self, tmp_path, capsys):
        from dsagt.observability import print_sidechannel_warning
        self._write_log(tmp_path, [
            {"model": "gpt-4o-mini", "session": "OTHER", "agent": "goose"},
        ])
        print_sidechannel_warning(tmp_path, "MINE")
        assert capsys.readouterr().out == ""

    def test_lists_unique_models_with_counts(self, tmp_path, capsys):
        from dsagt.observability import print_sidechannel_warning
        self._write_log(tmp_path, [
            {"model": "gpt-4o-mini", "session": "S", "agent": "goose"},
            {"model": "gpt-4o-mini", "session": "S", "agent": "goose"},
            {"model": "claude-haiku-4-5", "session": "S", "agent": "claude"},
        ])
        print_sidechannel_warning(tmp_path, "S")
        out = capsys.readouterr().out
        assert "Sidechannel model calls intercepted" in out
        assert "gpt-4o-mini" in out
        assert "(2 calls)" in out
        assert "claude-haiku-4-5" in out
        assert "(1 call)" in out


# ---------------------------------------------------------------------------
# DSAGTCallback (the LiteLLM CustomLogger)
# ---------------------------------------------------------------------------


class TestDSAGTCallback:
    """Minimal callback that hooks two LiteLLM events:
    log_pre_api_call → cache injection
    log_success_event → sidechannel detection.
    Trace transport is via ``litellm.callbacks = ["otel"]``, not here.
    """

    def test_pre_api_call_injects_cache(self, tmp_path):
        from dsagt.observability import _make_dsagt_callback
        # litellm.integrations.custom_logger may not be installed in some
        # CI envs; if so, skip — the callback is purely a litellm wrapper.
        pytest.importorskip("litellm")
        cb = _make_dsagt_callback(records_dir=tmp_path / "trace_archive")
        messages = [{"role": "system", "content": "hi"}]
        kwargs = {}
        cb.log_pre_api_call(model="m", messages=messages, kwargs=kwargs)
        assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_success_event_records_sidechannel(self, tmp_path, monkeypatch):
        from dsagt.observability import (
            _make_dsagt_callback, SIDECHANNEL_CATCHALL_MODEL,
            SIDECHANNEL_LOG_FILENAME,
        )
        pytest.importorskip("litellm")
        monkeypatch.setenv("DSAGT_PRIMARY_MODEL", "real-model")
        records_dir = tmp_path / "trace_archive"
        records_dir.mkdir()
        cb = _make_dsagt_callback(records_dir)
        kwargs = {
            "model": SIDECHANNEL_CATCHALL_MODEL,
            "standard_logging_object": {"model_group": "openai/gpt-4o-mini"},
        }
        cb.log_success_event(kwargs=kwargs, response_obj=None,
                             start_time=0, end_time=0)
        log_path = tmp_path / SIDECHANNEL_LOG_FILENAME
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["model"] == "gpt-4o-mini"

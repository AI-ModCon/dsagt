"""
Tests for dsagt proxy callback.

Covers tool_use extraction (OpenAI and Anthropic formats), tool_result matching,
execution record creation, and the success handler wiring.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dsagt.provenance import (
    ToolRecordStore,
    _extract_tool_result_text,
    _handle_success,
    _parse_arguments,
    _response_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ToolRecordStore(records_dir=tmp_path, session_id="test-session")


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ToolRecordStore: tool_use tracking — OpenAI format
# ---------------------------------------------------------------------------

class TestTrackToolUsesOpenAI:

    def _openai_response(self, tool_calls):
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            }]
        }

    def test_tracks_single_tool_call(self, store):
        response = self._openai_response([{
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
        }])

        tracked = store.track_tool_uses(response)

        assert len(tracked) == 1
        assert tracked[0]["tool_name"] == "bash"
        assert tracked[0]["parameters"] == {"command": "ls -la"}
        assert store.pending_count == 1

    def test_tracks_multiple_tool_calls(self, store):
        response = self._openai_response([
            {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x.py"}'}},
        ])

        tracked = store.track_tool_uses(response)

        assert len(tracked) == 2
        assert store.pending_count == 2

    def test_no_tool_calls(self, store):
        response = {"choices": [{"message": {"content": "Hello!", "tool_calls": None}}]}
        tracked = store.track_tool_uses(response)
        assert tracked == []


# ---------------------------------------------------------------------------
# ToolRecordStore: tool_use tracking — Anthropic format
# ---------------------------------------------------------------------------

class TestTrackToolUsesAnthropic:

    def test_tracks_anthropic_tool_use(self, store):
        response = {
            "content": [
                {"type": "text", "text": "Let me run that."},
                {"type": "tool_use", "id": "toolu_abc", "name": "bash", "input": {"command": "echo hi"}},
            ]
        }

        tracked = store.track_tool_uses(response)

        assert len(tracked) == 1
        assert tracked[0]["tool_name"] == "bash"
        assert tracked[0]["tool_use_id"] == "toolu_abc"
        assert tracked[0]["parameters"] == {"command": "echo hi"}

    def test_ignores_text_blocks(self, store):
        response = {"content": [{"type": "text", "text": "Just text."}]}
        tracked = store.track_tool_uses(response)
        assert tracked == []


# ---------------------------------------------------------------------------
# ToolRecordStore: tool_result matching — OpenAI format
# ---------------------------------------------------------------------------

class TestMatchToolResultsOpenAI:

    def test_matches_tool_result(self, store):
        # First, track a tool_use
        store._pending["call_abc"] = {
            "tool_use_id": "call_abc",
            "tool_name": "bash",
            "parameters": {"command": "ls"},
            "timestamp_requested": _now().isoformat(),
        }

        messages = [{"role": "tool", "tool_call_id": "call_abc", "content": "file1.txt\nfile2.txt"}]
        paths = store.match_tool_results(messages)

        assert len(paths) == 1
        assert store.pending_count == 0

        data = json.loads(paths[0].read_text())
        assert data["tool_name"] == "bash"
        assert data["intent"]["command"] == "bash"
        assert data["report"]["agent_output"] == "file1.txt\nfile2.txt"
        assert data["execution"] is None

    def test_ignores_unmatched_tool_result(self, store):
        messages = [{"role": "tool", "tool_call_id": "call_unknown", "content": "output"}]
        paths = store.match_tool_results(messages)
        assert paths == []


# ---------------------------------------------------------------------------
# ToolRecordStore: tool_result matching — Anthropic format
# ---------------------------------------------------------------------------

class TestMatchToolResultsAnthropic:

    def test_matches_anthropic_tool_result(self, store):
        store._pending["toolu_xyz"] = {
            "tool_use_id": "toolu_xyz",
            "tool_name": "fastp",
            "parameters": {"input": "reads.fq"},
            "timestamp_requested": _now().isoformat(),
        }

        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "Filtering complete."},
            ],
        }]
        paths = store.match_tool_results(messages)

        assert len(paths) == 1
        data = json.loads(paths[0].read_text())
        assert data["tool_name"] == "fastp"
        assert data["report"]["agent_output"] == "Filtering complete."

    def test_handles_list_content_in_tool_result(self, store):
        """Anthropic tool_result content can be a list of text blocks."""
        store._pending["toolu_list"] = {
            "tool_use_id": "toolu_list",
            "tool_name": "megahit",
            "parameters": {},
            "timestamp_requested": _now().isoformat(),
        }

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_list",
                    "content": [
                        {"type": "text", "text": "Assembly complete."},
                        {"type": "text", "text": "42 contigs produced."},
                    ],
                },
            ],
        }]
        paths = store.match_tool_results(messages)

        data = json.loads(paths[0].read_text())
        assert "Assembly complete." in data["report"]["agent_output"]
        assert "42 contigs" in data["report"]["agent_output"]


# ---------------------------------------------------------------------------
# ToolRecordStore: end-to-end tool lifecycle
# ---------------------------------------------------------------------------

class TestToolLifecycle:

    def test_track_then_match(self, store):
        """Full flow: response has tool_use → next request has tool_result."""
        # Step 1: LLM responds with a tool_use
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_lifecycle",
                        "type": "function",
                        "function": {"name": "fastp", "arguments": '{"quality": 20}'},
                    }]
                }
            }]
        }
        store.track_tool_uses(response)
        assert store.pending_count == 1

        # Step 2: Agent runs the tool, next request includes the result
        messages = [{"role": "tool", "tool_call_id": "call_lifecycle", "content": "98% reads passed"}]
        paths = store.match_tool_results(messages)

        assert len(paths) == 1
        assert store.pending_count == 0

        record = json.loads(paths[0].read_text())
        assert record["intent"]["parameters"] == {"quality": 20}
        assert record["report"]["agent_output"] == "98% reads passed"
        assert record["execution"] is None
        assert record["report"]["wrapper_used"] is False


# ---------------------------------------------------------------------------
# Execution record format
# ---------------------------------------------------------------------------

class TestExecutionRecordFormat:

    def test_record_has_all_layers(self, store):
        store._pending["id_fmt"] = {
            "tool_use_id": "id_fmt",
            "tool_name": "test_tool",
            "parameters": {"x": 1},
            "timestamp_requested": "2025-01-01T00:00:00+00:00",
        }

        paths = store.match_tool_results([{"role": "tool", "tool_call_id": "id_fmt", "content": "done"}])
        data = json.loads(paths[0].read_text())

        assert data["record_id"] == "id_fmt"
        assert data["tool_name"] == "test_tool"
        assert data["session_id"] == "test-session"

        assert "command" in data["intent"]
        assert "parameters" in data["intent"]
        assert "timestamp_requested" in data["intent"]

        assert data["execution"] is None

        assert "agent_output" in data["report"]
        assert "timestamp_reported" in data["report"]
        assert "wrapper_used" in data["report"]


# ---------------------------------------------------------------------------
# _handle_success (the callback wiring)
# ---------------------------------------------------------------------------

class TestHandleSuccess:

    def test_tracks_tool_uses_from_response(self, store):
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "run ls"}],
        }
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_from_handler",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }]
                }
            }],
        }

        _handle_success(store, kwargs, response, _now(), _now())

        assert store.pending_count == 1
        assert "call_from_handler" in store._pending

    def test_matches_tool_results_from_messages(self, store):
        """When messages contain tool_results, they're matched."""
        store._pending["call_prev"] = {
            "tool_use_id": "call_prev",
            "tool_name": "fastp",
            "parameters": {},
            "timestamp_requested": _now().isoformat(),
        }

        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "tool", "tool_call_id": "call_prev", "content": "done"},
                {"role": "user", "content": "what happened?"},
            ],
        }
        response = {"id": "msg_02"}

        _handle_success(store, kwargs, response, _now(), _now())

        assert store.pending_count == 0
        records = [f for f in store.records_dir.glob("*.json") if "fastp" in f.name]
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestParseArguments:

    def test_json_string(self):
        assert _parse_arguments('{"a": 1}') == {"a": 1}

    def test_already_dict(self):
        assert _parse_arguments({"a": 1}) == {"a": 1}

    def test_invalid_json(self):
        result = _parse_arguments("not json")
        assert "_raw" in result


class TestExtractToolResultText:

    def test_string_content(self):
        assert _extract_tool_result_text({"content": "hello"}) == "hello"

    def test_list_content(self):
        block = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        assert _extract_tool_result_text(block) == "a\nb"

    def test_empty(self):
        assert _extract_tool_result_text({}) == ""


class TestResponseToDict:

    def test_dict_passthrough(self):
        assert _response_to_dict({"a": 1}) == {"a": 1}

    def test_object_with_model_dump(self):
        class FakeResponse:
            def model_dump(self):
                return {"id": "msg_01"}
        assert _response_to_dict(FakeResponse()) == {"id": "msg_01"}

    def test_fallback_to_repr(self):
        result = _response_to_dict(42)
        assert "_repr" in result

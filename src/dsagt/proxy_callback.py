"""
DSAGT proxy callback for LiteLLM.

Creates tool execution records from tool_use/tool_result blocks in LLM
conversations. Runs alongside LiteLLM's built-in OTel callback which handles
standard trace capture (spans, metrics → MLflow/Jaeger).

This callback writes tool execution records to the records directory:
  <tool>_<ts>_<id>.json  — intent + report layers

These records pair with execution-layer records from dsagt-run.
Together they form the three-layer tool execution record from the design doc.

Responsibility split:
  OTel callback (LiteLLM built-in) → all API traces, spans, token/cost metrics
  DSAGT callback (this module)     → tool execution records only
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool execution record storage
# ---------------------------------------------------------------------------

class ToolRecordStore:
    """Manages tool execution records on disk.

    Tracks tool_use blocks from LLM responses, matches them against
    tool_result blocks in subsequent requests, and writes the combined
    intent + report record. dsagt-run optionally fills in the execution layer.
    """

    def __init__(self, records_dir: str | Path, session_id: str | None = None):
        self.records_dir = Path(records_dir)
        self.session_id = session_id or os.environ.get("DSAGT_SESSION_ID")

        # Pending tool_use blocks awaiting their tool_result.
        # Keyed by tool_use id (from the LLM response).
        self._pending: dict[str, dict] = {}

        self.records_dir.mkdir(parents=True, exist_ok=True)

    # -- Tool use/result tracking -------------------------------------------

    def track_tool_uses(self, response_data: dict) -> list[dict]:
        """Extract tool_use/tool_calls from a response and track them.

        Handles both Anthropic format (content blocks with type=tool_use)
        and OpenAI format (choices[].message.tool_calls).

        Returns the list of tracked tool uses for testing.
        """
        tracked = []
        now = datetime.now(timezone.utc).isoformat()

        # OpenAI format: choices[].message.tool_calls
        for choice in response_data.get("choices", []):
            message = choice.get("message") or {}
            for tc in message.get("tool_calls") or []:
                func = tc.get("function", {})
                entry = {
                    "tool_use_id": tc.get("id"),
                    "tool_name": func.get("name"),
                    "parameters": _parse_arguments(func.get("arguments", "{}")),
                    "timestamp_requested": now,
                }
                self._pending[entry["tool_use_id"]] = entry
                tracked.append(entry)

        # Anthropic format: content[] with type=tool_use
        for block in response_data.get("content", []):
            if block.get("type") == "tool_use":
                entry = {
                    "tool_use_id": block["id"],
                    "tool_name": block["name"],
                    "parameters": block.get("input", {}),
                    "timestamp_requested": now,
                }
                self._pending[entry["tool_use_id"]] = entry
                tracked.append(entry)

        for t in tracked:
            logger.info("Tracking tool_use: %s (%s)", t["tool_name"], t["tool_use_id"][:12])

        return tracked

    def match_tool_results(self, messages: list[dict]) -> list[Path]:
        """Find tool_result/tool-role messages and write execution records.

        Handles both Anthropic format (user messages with tool_result content
        blocks) and OpenAI format (messages with role=tool).

        Returns paths to any records written.
        """
        paths = []

        for msg in messages:
            # OpenAI format: role=tool, tool_call_id, content
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id in self._pending:
                    intent = self._pending.pop(tool_call_id)
                    path = self._write_execution_record(intent, msg.get("content", ""))
                    paths.append(path)
                continue

            # Anthropic format: role=user with content blocks of type=tool_result
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if tool_use_id and tool_use_id in self._pending:
                    intent = self._pending.pop(tool_use_id)
                    result_text = _extract_tool_result_text(block)
                    path = self._write_execution_record(intent, result_text)
                    paths.append(path)

        return paths

    def _write_execution_record(self, intent: dict, agent_output: str) -> Path:
        """Write a tool execution record with intent + report layers."""
        record = {
            "record_id": intent["tool_use_id"],
            "tool_name": intent["tool_name"],
            "session_id": self.session_id,
            "intent": {
                "command": intent["tool_name"],
                "parameters": intent["parameters"],
                "timestamp_requested": intent["timestamp_requested"],
                "session_id": self.session_id,
            },
            "execution": None,
            "report": {
                "agent_output": agent_output,
                "timestamp_reported": datetime.now(timezone.utc).isoformat(),
                "wrapper_used": False,
            },
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid = intent["tool_use_id"][:12]
        filename = f"{intent['tool_name']}_{ts}_{rid}.json"
        path = self.records_dir / filename
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        logger.info("Execution record: %s", filename)
        return path

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_arguments(raw: str | dict) -> dict:
    """Parse tool call arguments — may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def _extract_tool_result_text(block: dict) -> str:
    """Extract text content from an Anthropic tool_result block."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(texts)
    return str(content)


# ---------------------------------------------------------------------------
# Response serialization
# ---------------------------------------------------------------------------

def _response_to_dict(response_obj) -> dict:
    """Convert a LiteLLM ModelResponse (or dict) to a plain dict."""
    if isinstance(response_obj, dict):
        return response_obj
    # ModelResponse and similar objects support model_dump() or dict()
    for method in ("model_dump", "dict", "to_dict"):
        fn = getattr(response_obj, method, None)
        if fn:
            return fn()
    # Last resort
    return {"_repr": str(response_obj)}


# ---------------------------------------------------------------------------
# LiteLLM callback (requires litellm to be installed)
# ---------------------------------------------------------------------------

def create_callback(records_dir: str | Path, session_id: str | None = None):
    """Create a DSAGT callback instance for LiteLLM.

    Imports litellm lazily so the rest of the module is testable without it.
    """
    from litellm.integrations.custom_logger import CustomLogger

    store = ToolRecordStore(records_dir, session_id)

    class DSAGTCallback(CustomLogger):
        """LiteLLM callback that creates tool execution records.

        Fires on every LLM call but only writes to disk when tool_use
        or tool_result blocks are present. Standard trace capture is
        handled by LiteLLM's OTel callback.
        """

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _handle_success(store, kwargs, response_obj, start_time, end_time)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            _handle_success(store, kwargs, response_obj, start_time, end_time)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            logger.warning("LLM call failed: model=%s", kwargs.get("model"))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            logger.warning("LLM call failed: model=%s", kwargs.get("model"))

    return DSAGTCallback()


def _handle_success(store: ToolRecordStore, kwargs, response_obj, start_time, end_time):
    """Shared logic for sync and async success handlers."""
    messages = kwargs.get("messages", [])
    response_data = _response_to_dict(response_obj)

    # Match any tool_results from the current request against pending tool_uses
    store.match_tool_results(messages)

    # Track any new tool_uses from the response
    store.track_tool_uses(response_data)

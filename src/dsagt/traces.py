"""
Trace pipeline — read each agent's on-disk session, normalize it to one common
``Trace``, and hand that to whatever consumes traces (the MLflow logger in
:mod:`dsagt.observability`, the episodic-memory indexer in :mod:`dsagt.memory`).

While a session runs, the MCP server wakes on a timer and, for the running agent:
a **Reader** finds and reads the platform's session files on disk into raw
records; the matching **Translator** maps those records into one **Trace**; and
the **TraceCollector** hands the result to its consumers, skipping the turns it
already handled.

 A session ``Trace`` carries one AGENT subtree per turn
 (an AGENT root with ``llm`` / ``tool_<name>``
children), matching the per-prompt granularity MLflow's own claude autolog
produces.  Fidelity is capped by what the transcript persisted: every timestamp
and token count is ``None``-tolerant.

The Claude grammar below is ported from MLflow's
``claude_code/tracing.py`` — © Databricks, Inc., Apache-2.0 — specifically its
turn-windowing, skill/command skips, and next-timestamp span durations.  See
NOTICE.

Class map — ``▷`` inherits · ``◆`` owns · ``◇`` holds  (``*`` = many)::

    Trace                       one session: id fields + spans (list of dicts)
                                + compose / query / to_exchanges methods

    Reader  «abstract»          locate + read a platform's session → raw records
    ├─▷ JsonlReader «abstract»    shared whole-file line framing; active_file() hook
    │     ├─▷ ClaudeReader        ~/.claude/projects/<cwd>/*.jsonl (newest)
    │     └─▷ CodexReader         ~/.codex/sessions/**/rollout-*.jsonl (by cwd)
    ├─▷ GooseReader              goose sessions.db        (sqlite, read-only)
    ├─▷ OpenCodeReader           opencode.db              (sqlite, read-only)
    └─▷ ClineReader              ~/.cline/.../<id>.messages.json (whole file)

    Translator  «abstract»      raw records → Trace (pure); shared turn template
    ├─▷ ClaudeTranslator         overrides translate() — bespoke grammar
    ├─▷ CodexTranslator          fills parse hooks (+ normalize / prompt-index)
    ├─▷ GooseTranslator          fills parse hooks
    ├─▷ OpenCodeTranslator       fills parse hooks
    └─▷ ClineTranslator          fills parse hooks

    TraceCollector              the driver: read → translate → hand to consumers
                                (MLflow logger, memory indexer); a per-consumer
                                ack set makes repeated passes idempotent
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SPAN_SECONDS = (
    1.0  # fallback turn-span duration when no next timestamp bounds it
)
_ROLE_ASSISTANT = "assistant"
_ROLE_USER = "user"
_TYPE_QUEUE_OP = "queue-operation"


def _parse_ts(ts: object) -> float | None:
    """ISO-8601 string (or epoch number) → epoch seconds; ``None`` on failure."""
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# The block / message / usage shapes (the one place they're constructed)
# ---------------------------------------------------------------------------


def _text_block(text: str | None) -> dict:
    return {"type": "text", "text": text or ""}


def _tool_use_block(name, tool_input, tool_call_id=None) -> dict:
    return {
        "type": "tool_use",
        "id": tool_call_id,
        "name": name,
        "input": tool_input or {},
    }


def _tool_result_block(output, tool_call_id=None) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_call_id, "content": output}


def _message(role: str, blocks: list[dict]) -> dict:
    return {"role": role, "content": blocks}


def _usage(raw: dict | None) -> dict | None:
    """Token counts from a transcript ``usage`` dict; ``None`` when absent."""
    if not raw:
        return None
    return {
        "input_tokens": raw.get("input_tokens"),
        "output_tokens": raw.get("output_tokens"),
        "cache_read_input_tokens": raw.get("cache_read_input_tokens"),
        "cache_write_input_tokens": raw.get("cache_creation_input_tokens"),
    }


# ===========================================================================
# Trace — the canonical form (nested data + composition / query methods)
# ===========================================================================


class Trace:
    """One finished agent session: id fields plus a list of span dicts.

    The span/message/block shapes are documented in the module docstring and
    built *only* by the ``add_*`` methods here, so the schema has one home.
    Consumers read ``spans`` (and ``to_exchanges()``) directly — the data is
    already in the dict shape both the MLflow logger and memory want.
    """

    def __init__(self, trace_id: str, session_id: str, agent: str, project: str):
        self.trace_id = trace_id
        self.session_id = session_id
        self.agent = agent
        self.project = project
        self.spans: list[dict] = []

    @property
    def started_at(self) -> float | None:
        return self.spans[0]["start_time"] if self.spans else None

    @property
    def ended_at(self) -> float | None:
        return self.spans[-1]["end_time"] if self.spans else None

    # -- composition (the only place a span dict is constructed) -------------

    def add_agent_root(self, span_id, name, *, start_time, prompt) -> dict:
        span = {
            "span_id": span_id,
            "name": name,
            "kind": "AGENT",
            "parent_id": None,
            "start_time": start_time,
            "end_time": None,
            "status": "ok",
            "request": [],
            "response": [],
            "model": None,
            "usage": None,
            "attributes": {"prompt": prompt},
        }
        self.spans.append(span)
        return span

    def add_llm_span(
        self,
        span_id,
        *,
        parent_id,
        start_time,
        end_time,
        request,
        response,
        model=None,
        usage=None,
    ) -> dict:
        span = {
            "span_id": span_id,
            "name": "llm",
            "kind": "LLM",
            "parent_id": parent_id,
            "start_time": start_time,
            "end_time": end_time,
            "status": "ok",
            "request": request,
            "response": response,
            "model": model,
            "usage": usage,
            "attributes": {},
        }
        self.spans.append(span)
        return span

    def add_tool_span(
        self,
        span_id,
        *,
        parent_id,
        start_time,
        end_time,
        name,
        tool_input,
        result,
        tool_id="",
    ) -> dict:
        span = {
            "span_id": span_id,
            "name": f"tool_{name}",
            "kind": "TOOL",
            "parent_id": parent_id,
            "start_time": start_time,
            "end_time": end_time,
            "status": "ok",
            "request": [],
            "response": [],
            "model": None,
            "usage": None,
            "attributes": {
                "tool_name": name,
                "tool_id": tool_id,
                "input": tool_input,
                "result": result,
            },
        }
        self.spans.append(span)
        return span

    def add_turn(self, *, root_id, root_name, prompt, root_ts, events, last_ts) -> None:
        """Append one AGENT subtree from a turn's ordered ``events``.

        Each event is a tuple — ``("llm", ts, text, model, usage)`` or
        ``("tool", ts, name, input, result)`` — in transcript order.  This is the
        shared builder the four template translators use: it derives each span's
        duration from the next event's timestamp (1s fallback for the last), and
        threads the request "window" (the prompt, then each tool call+result)
        into the following ``llm`` span's ``request`` — which is what memory's
        ``to_exchanges`` reads.
        """
        root = self.add_agent_root(
            root_id, root_name, start_time=root_ts, prompt=prompt
        )
        pending = [_message(_ROLE_USER, [_text_block(prompt)])]
        ts_list = [e[1] for e in events]
        final_response: str | None = None
        for i, ev in enumerate(events):
            ts = ev[1]
            nxt = next((t for t in ts_list[i + 1 :] if t is not None), last_ts)
            dur = (
                (nxt - ts)
                if (ts is not None and nxt is not None and nxt > ts)
                else _DEFAULT_SPAN_SECONDS
            )
            end = (ts + dur) if ts is not None else None
            if ev[0] == "llm":
                _, _, text, model, usage = ev
                final_response = text
                self.add_llm_span(
                    f"{root_id}-{i}",
                    parent_id=root_id,
                    start_time=ts,
                    end_time=end,
                    request=list(pending),
                    response=[_text_block(text)],
                    model=model,
                    usage=usage,
                )
                pending = []
            else:  # "tool"
                _, _, name, tin, result = ev
                self.add_tool_span(
                    f"{root_id}-{i}",
                    parent_id=root_id,
                    start_time=ts,
                    end_time=end,
                    name=name,
                    tool_input=tin,
                    result=result,
                )
                tool_input = tin if isinstance(tin, dict) else {"raw": tin}
                pending.append(
                    _message(_ROLE_ASSISTANT, [_tool_use_block(name, tool_input)])
                )
                pending.append(_message(_ROLE_USER, [_tool_result_block(result)]))
        root["end_time"] = last_ts if last_ts is not None else root_ts
        if final_response is not None:
            root["attributes"]["response"] = final_response

    # -- query / projection -------------------------------------------------

    def roots(self) -> list[dict]:
        return [s for s in self.spans if s["parent_id"] is None]

    def children(self, root_id) -> list[dict]:
        return [s for s in self.spans if s["parent_id"] == root_id]

    def subset(self, root_ids: set[str]) -> "Trace":
        """A copy carrying only the given AGENT roots and their direct children."""
        keep = [
            s
            for s in self.spans
            if (s["parent_id"] is None and s["span_id"] in root_ids)
            or s["parent_id"] in root_ids
        ]
        out = Trace(self.trace_id, self.session_id, self.agent, self.project)
        out.spans = keep
        return out

    def to_exchanges(self) -> list[dict]:
        """Project the ``llm`` spans onto memory's conversational shape.

        One ``llm`` span → one ``{turn_id, timestamp, new_messages, response}``
        exchange; ``turn_id`` is the span id (groups a turn's chunks back
        together downstream), ``request`` is already the windowed message list,
        ``response`` the output blocks — so this is a straight projection.
        """
        return [
            {
                "turn_id": s["span_id"],
                "timestamp": s["start_time"],
                "new_messages": s["request"],
                "response": s["response"],
            }
            for s in self.spans
            if s["kind"] == "LLM"
        ]


# ===========================================================================
# Readers — locate + read a platform's session record into raw records
# ===========================================================================


class Reader(ABC):
    """Find this project's active session for an agent and read its records.

    Each reader resolves *the latest session for this project* and reads it.  A
    reader can also be pinned (:meth:`pin`) to a specific session — the startup
    catch-up does this to re-read the *previous* session rather than whatever is
    newest now.  :meth:`active_source` returns an opaque, agent-shaped token for
    the session being read — a transcript path (claude/codex), a DB session id
    (goose/opencode), or a session-dir name (cline) — which the server records
    in ``state.yaml`` so the next session's catch-up can pin it back.  The token
    round-trips through YAML, so its native type (str/int) is preserved.
    """

    agent: str
    _pinned = None

    def pin(self, source) -> None:
        """Pin to a specific session token (as returned by :meth:`active_source`)."""
        self._pinned = source

    def active_source(self):
        """Token identifying the session being read, or ``None`` if none.

        Subclasses override to resolve the latest session when unpinned; the
        base returns the pinned token (``None`` when neither pinned nor
        overridden).
        """
        return self._pinned

    @abstractmethod
    def read(self) -> list[dict]:
        """The whole active session's raw records (re-read each pass; the
        collector's ack set dedupes, so reads need not be incremental)."""


class JsonlReader(Reader):
    """Read the whole active ``*.jsonl`` file, framing only complete lines.

    A trailing half-written line is dropped (picked up next pass).  Subclasses
    supply :meth:`active_file`; the framing is identical for claude and codex.
    A pinned reader reads that exact file instead of the newest.
    """

    @abstractmethod
    def active_file(self) -> Path | None: ...

    def _file(self) -> Path | None:
        if self._pinned is not None:
            p = Path(self._pinned)
            return p if p.is_file() else None
        return self.active_file()

    def active_source(self) -> str | None:
        f = self._file()
        return str(f) if f else None

    def read(self) -> list[dict]:
        f = self._file()
        if f is None:
            return []
        with open(f, "rb") as fh:
            chunk = fh.read()
        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            return []
        return [
            json.loads(line)
            for line in chunk[: last_nl + 1].splitlines()
            if line.strip()
        ]


def _transcript_dir(project_dir: str | Path, projects_root: Path | None = None) -> Path:
    """``~/.claude/projects/<mangled-cwd>`` for a project directory.

    Claude derives the dir name by replacing every non-alphanumeric character of
    the launch cwd with ``-``.  The MCP server launches as ``cd <project_dir> &&
    claude``, so cwd == project_dir.
    """
    root = projects_root or (Path.home() / ".claude" / "projects")
    mangled = re.sub(r"[^a-zA-Z0-9]", "-", os.path.abspath(project_dir))
    return root / mangled


class ClaudeReader(JsonlReader):
    """The most-recently-modified transcript in the project's ``~/.claude`` dir."""

    agent = "claude"

    def __init__(self, project_dir, *, projects_root: Path | None = None):
        self._dir = _transcript_dir(project_dir, projects_root)

    def active_file(self) -> Path | None:
        if not self._dir.is_dir():
            return None
        files = list(self._dir.glob("*.jsonl"))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None


_CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"


class CodexReader(JsonlReader):
    """The newest ``rollout-*.jsonl`` whose ``session_meta.cwd`` is this project.

    Codex rollouts live globally under ``~/.codex/sessions/YYYY/MM/DD/``; each
    opens with a ``session_meta`` record carrying the launch ``cwd``.
    """

    agent = "codex"

    def __init__(self, project_dir, *, sessions_root: Path | None = None):
        self._project_dir = os.path.abspath(project_dir)
        self._root = Path(sessions_root) if sessions_root else _CODEX_SESSIONS_ROOT

    def _rollout_cwd(self, path: Path) -> str | None:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        if not first.strip():
            return None
        rec = json.loads(first)
        if rec.get("type") != "session_meta":
            return None
        return (rec.get("payload") or {}).get("cwd")

    def active_file(self) -> Path | None:
        if not self._root.is_dir():
            return None
        files = sorted(
            self._root.glob("**/rollout-*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files:
            if self._rollout_cwd(f) == self._project_dir:
                return f
        return None


_GOOSE_DB = Path.home() / ".local" / "share" / "goose" / "sessions" / "sessions.db"


class GooseReader(Reader):
    """Read the project's active Goose session from its SQLite store (read-only)."""

    agent = "goose"

    def __init__(self, project_dir, *, db_path: Path | None = None):
        self._project_dir = os.path.abspath(project_dir)
        self._db = Path(db_path) if db_path else _GOOSE_DB

    def _latest_session(self, con):
        row = con.execute(
            "SELECT id FROM sessions WHERE working_dir = ? "
            "ORDER BY updated_at DESC LIMIT 1",
            (self._project_dir,),
        ).fetchone()
        return row[0] if row else None

    def active_source(self):
        if self._pinned is not None:
            return self._pinned
        if not self._db.exists():
            return None
        con = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
        try:
            return self._latest_session(con)
        finally:
            con.close()

    def read(self) -> list[dict]:
        if not self._db.exists():
            return []
        con = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
        try:
            sid = (
                self._pinned if self._pinned is not None else self._latest_session(con)
            )
            if sid is None:
                return []
            return [
                {"role": role, "content": _loads_list(cj), "ts": ts}
                for role, cj, ts in con.execute(
                    "SELECT role, content_json, created_timestamp FROM messages "
                    "WHERE session_id = ? ORDER BY id",
                    (sid,),
                )
            ]
        finally:
            con.close()


_OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"


class OpenCodeReader(Reader):
    """Read + flatten the project's active opencode session (sqlite, read-only).

    opencode splits a session into ``message`` rows (role/model) and ``part``
    rows (the text/tool payloads); this joins them into one time-ordered list of
    parts, each tagged with its message's role + model and a timestamp in
    seconds (opencode stores milliseconds).
    """

    agent = "opencode"

    def __init__(self, project_dir, *, db_path: Path | None = None):
        self._project_dir = os.path.abspath(project_dir)
        self._db = Path(db_path) if db_path else _OPENCODE_DB

    def _latest_session(self, con):
        row = con.execute(
            "SELECT id FROM session WHERE directory = ? "
            "ORDER BY time_updated DESC LIMIT 1",
            (self._project_dir,),
        ).fetchone()
        return row[0] if row else None

    def active_source(self):
        if self._pinned is not None:
            return self._pinned
        if not self._db.exists():
            return None
        con = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
        try:
            return self._latest_session(con)
        finally:
            con.close()

    def read(self) -> list[dict]:
        if not self._db.exists():
            return []
        con = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
        try:
            sid = (
                self._pinned if self._pinned is not None else self._latest_session(con)
            )
            if sid is None:
                return []
            messages = {
                mid: _loads_dict(d)
                for mid, d in con.execute(
                    "SELECT id, data FROM message WHERE session_id = ?", (sid,)
                )
            }
            parts = []
            for mid, t_created, data in con.execute(
                "SELECT message_id, time_created, data FROM part "
                "WHERE session_id = ? ORDER BY time_created",
                (sid,),
            ):
                m = messages.get(mid, {})
                parts.append(
                    {
                        "role": m.get("role"),
                        "model": (m.get("model") or {}).get("modelID"),
                        "ts": t_created / 1000.0,
                        "data": _loads_dict(data),
                    }
                )
            return parts
        finally:
            con.close()


_CLINE_SESSIONS_ROOT = Path.home() / ".cline" / "data" / "sessions"


class ClineReader(Reader):
    """Locate the project's active Cline CLI session by its metadata ``cwd``.

    Each session is ``~/.cline/data/sessions/<id>/`` with ``<id>.json`` (metadata
    incl. ``cwd`` + ``model``) and ``<id>.messages.json`` (the whole message list).
    """

    agent = "cline"

    def __init__(self, project_dir, *, sessions_root: Path | None = None):
        self._project_dir = os.path.abspath(project_dir)
        self._root = Path(sessions_root) if sessions_root else _CLINE_SESSIONS_ROOT

    def _active_dir(self) -> Path | None:
        if not self._root.is_dir():
            return None
        for d in sorted(self._root.iterdir(), key=lambda p: p.name, reverse=True):
            meta = d / f"{d.name}.json"
            if not meta.exists():
                continue
            try:
                cwd = json.loads(meta.read_text()).get("cwd", "")
            except (json.JSONDecodeError, OSError):
                continue
            if os.path.abspath(cwd) == self._project_dir:
                return d
        return None

    def _dir(self) -> Path | None:
        if self._pinned is not None:
            d = self._root / str(self._pinned)
            return d if d.is_dir() else None
        return self._active_dir()

    def active_source(self) -> str | None:
        d = self._dir()
        return d.name if d else None

    def read(self) -> list[dict]:
        d = self._dir()
        if d is None:
            return []
        msgs_path = d / f"{d.name}.messages.json"
        if not msgs_path.exists():
            return []
        try:
            data = json.loads(msgs_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        model = self._session_model(d)
        messages = data.get("messages", []) if isinstance(data, dict) else data
        for m in messages:
            m["model"] = model
        return messages

    def _session_model(self, session_dir: Path) -> str | None:
        try:
            meta = json.loads((session_dir / f"{session_dir.name}.json").read_text())
            return meta.get("model")
        except (json.JSONDecodeError, OSError):
            return None


def _loads_list(raw: str) -> list:
    try:
        out = json.loads(raw)
        return out if isinstance(out, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _loads_dict(raw: str) -> dict:
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ===========================================================================
# Translators — raw records → Trace (pure, no I/O)
# ===========================================================================


class Translator(ABC):
    """Map one platform's records to a :class:`Trace`.

    The default ``translate`` is the shared template every platform but Claude
    uses: build a tool-result map, find the turn-start (prompt) indices, and for
    each turn lower its records into ordered ``("llm"|"tool", …)`` events that
    ``Trace.add_turn`` turns into a span subtree.  A subclass supplies the small
    parse hooks (``_is_prompt`` / ``_prompt_text`` / ``_ts`` / ``_events`` and,
    where needed, ``_tool_results`` / ``_normalize`` / ``_prompt_indices``).
    Claude overrides ``translate`` outright — its grammar exceeds this shape.
    """

    agent: str
    root_name: str

    def translate(self, records, *, trace_id, session_id, project) -> Trace | None:
        records = self._normalize(records)
        results = self._tool_results(records)
        prompts = self._prompt_indices(records)
        if not prompts:
            return None
        trace = Trace(trace_id, session_id, self.agent, project)
        bounds = prompts + [len(records)]
        for k, start in enumerate(prompts):
            self._build_turn(trace, records, start, bounds[k + 1], results)
        return trace if trace.spans else None

    def _build_turn(self, trace, records, start, end, results) -> None:
        root_ts = self._ts(records[start])
        last_ts = root_ts
        events = []
        for i in range(start + 1, end):
            rec = records[i]
            ts = self._ts(rec)
            if ts is not None:
                last_ts = ts
            events.extend(self._events(rec, ts, results))
        trace.add_turn(
            root_id=f"turn-{start}",
            root_name=self.root_name,
            prompt=self._prompt_text(records[start]),
            root_ts=root_ts,
            events=events,
            last_ts=last_ts,
        )

    # -- hooks (defaults; concrete translators fill what they need) ----------

    def _normalize(self, records):
        return records

    def _prompt_indices(self, records) -> list[int]:
        return [i for i, r in enumerate(records) if self._is_prompt(r)]

    def _tool_results(self, records) -> dict:
        return {}

    def _is_prompt(self, rec) -> bool:
        raise NotImplementedError

    def _prompt_text(self, rec) -> str:
        raise NotImplementedError

    def _ts(self, rec) -> float | None:
        raise NotImplementedError

    def _events(self, rec, ts, results) -> list:
        raise NotImplementedError


class GooseTranslator(Translator):
    """Goose message rows → Trace (content blocks: text / toolRequest / toolResponse)."""

    agent = "goose"
    root_name = "goose_conversation"

    @staticmethod
    def _blocks(msg) -> list[dict]:
        c = msg.get("content")
        return c if isinstance(c, list) else []

    def _is_prompt(self, rec) -> bool:
        return rec.get("role") == _ROLE_USER and any(
            b.get("type") == "text" for b in self._blocks(rec)
        )

    def _prompt_text(self, rec) -> str:
        return "".join(
            b.get("text", "") for b in self._blocks(rec) if b.get("type") == "text"
        )

    def _ts(self, rec) -> float | None:
        return _parse_ts(rec.get("ts"))

    def _tool_results(self, records) -> dict:
        results: dict = {}
        for msg in records:
            for b in self._blocks(msg):
                if b.get("type") == "toolResponse" and (tid := b.get("id")):
                    results[tid] = self._result_text(b)
        return results

    @staticmethod
    def _result_text(block) -> object:
        value = (block.get("toolResult") or {}).get("value") or {}
        content = value.get("content")
        if isinstance(content, list):
            return "".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        return content if content is not None else ""

    def _events(self, rec, ts, results) -> list:
        if rec.get("role") != _ROLE_ASSISTANT:
            return []
        out = []
        for b in self._blocks(rec):
            if b.get("type") == "text" and b.get("text", "").strip():
                out.append(("llm", ts, b["text"], None, None))
            elif b.get("type") == "toolRequest":
                call = (b.get("toolCall") or {}).get("value") or {}
                out.append(
                    (
                        "tool",
                        ts,
                        call.get("name", "unknown"),
                        call.get("arguments", {}),
                        results.get(b.get("id"), ""),
                    )
                )
        return out


class OpenCodeTranslator(Translator):
    """opencode flattened parts → Trace (call + result live in one tool part)."""

    agent = "opencode"
    root_name = "opencode_conversation"

    def _is_prompt(self, rec) -> bool:
        d = rec.get("data") or {}
        return (
            rec.get("role") == _ROLE_USER
            and d.get("type") == "text"
            and bool(d.get("text", "").strip())
        )

    def _prompt_text(self, rec) -> str:
        return (rec.get("data") or {}).get("text", "")

    def _ts(self, rec) -> float | None:
        return rec.get("ts")

    def _events(self, rec, ts, results) -> list:
        d = rec.get("data") or {}
        if (
            d.get("type") == "text"
            and rec.get("role") == _ROLE_ASSISTANT
            and d.get("text", "").strip()
        ):
            return [("llm", ts, d["text"], rec.get("model"), None)]
        if d.get("type") == "tool":
            state = d.get("state") or {}
            return [
                (
                    "tool",
                    ts,
                    d.get("tool", "unknown"),
                    state.get("input", {}),
                    state.get("output", ""),
                )
            ]
        return []


_CLINE_USER_INPUT_RE = re.compile(r"<user_input[^>]*>(.*?)</user_input>", re.DOTALL)


class ClineTranslator(Translator):
    """Cline message records → Trace (Anthropic block list; ms timestamps)."""

    agent = "cline"
    root_name = "cline_conversation"

    @staticmethod
    def _content(msg) -> list[dict]:
        c = msg.get("content")
        return c if isinstance(c, list) else []

    def _join_text(self, msg) -> str:
        return "".join(
            b.get("text", "") for b in self._content(msg) if b.get("type") == "text"
        )

    def _is_prompt(self, rec) -> bool:
        return rec.get("role") == _ROLE_USER and any(
            b.get("type") == "text" and b.get("text", "").strip()
            for b in self._content(rec)
        )

    def _prompt_text(self, rec) -> str:
        text = self._join_text(rec)
        found = _CLINE_USER_INPUT_RE.findall(text)
        return "".join(found).strip() if found else text.strip()

    def _ts(self, rec) -> float | None:
        ts = rec.get("ts")
        return ts / 1000.0 if isinstance(ts, (int, float)) else None

    def _tool_results(self, records) -> dict:
        results: dict = {}
        for msg in records:
            for b in self._content(msg):
                if b.get("type") == "tool_result" and (tid := b.get("tool_use_id")):
                    results[tid] = self._result_text(b)
        return results

    @staticmethod
    def _result_text(block) -> object:
        content = block.get("content")
        if isinstance(content, list):
            return "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        return content if content is not None else ""

    def _events(self, rec, ts, results) -> list:
        if rec.get("role") != _ROLE_ASSISTANT:
            return []
        out = []
        text = self._join_text(rec)
        if text.strip():
            out.append(("llm", ts, text, rec.get("model"), None))
        for b in self._content(rec):
            if b.get("type") == "tool_use":
                out.append(
                    (
                        "tool",
                        ts,
                        b.get("name", "unknown"),
                        b.get("input", {}),
                        results.get(b.get("id"), ""),
                    )
                )
        return out


class CodexTranslator(Translator):
    """Codex rollout records → Trace (OpenAI Responses format).

    Normalizes the rollout into ``{ts, p}`` conversation items, then fits the
    template — except the prompt is the *last* user message in a consecutive run
    (Codex injects an AGENTS.md context message earlier), so ``_prompt_indices``
    is overridden with that lookahead.
    """

    agent = "codex"
    root_name = "codex_conversation"

    def _normalize(self, records) -> list[dict]:
        return [
            {"ts": _parse_ts(r.get("timestamp")), "p": r["payload"]}
            for r in records
            if r.get("type") == "response_item" and isinstance(r.get("payload"), dict)
        ]

    def _ts(self, rec) -> float | None:
        return rec["ts"]

    def _tool_results(self, records) -> dict:
        results: dict = {}
        for rec in records:
            p = rec["p"]
            if p.get("type") in ("function_call_output", "custom_tool_call_output"):
                if cid := p.get("call_id"):
                    results[cid] = p.get("output", "")
        return results

    def _prompt_indices(self, records) -> list[int]:
        msg_idxs = [i for i, r in enumerate(records) if r["p"].get("type") == "message"]
        prompts = []
        for pos, i in enumerate(msg_idxs):
            if records[i]["p"].get("role") != _ROLE_USER:
                continue
            nxt = msg_idxs[pos + 1] if pos + 1 < len(msg_idxs) else None
            if nxt is None or records[nxt]["p"].get("role") == _ROLE_ASSISTANT:
                prompts.append(i)
        return prompts

    def _prompt_text(self, rec) -> str:
        return self._msg_text(rec["p"])

    @staticmethod
    def _msg_text(item) -> str:
        return "".join(
            b.get("text", "")
            for b in (item.get("content") or [])
            if isinstance(b, dict) and b.get("type") in ("input_text", "output_text")
        )

    def _events(self, rec, ts, results) -> list:
        p = rec["p"]
        ptype = p.get("type")
        if ptype == "message" and p.get("role") == _ROLE_ASSISTANT:
            text = self._msg_text(p)
            return [("llm", ts, text, None, None)] if text.strip() else []
        if ptype in ("function_call", "custom_tool_call"):
            name, tool_input = self._tool_call(p)
            return [
                ("tool", ts, name, tool_input, results.get(p.get("call_id", ""), ""))
            ]
        return []

    @staticmethod
    def _tool_call(item) -> tuple[str, object]:
        name = item.get("name", "unknown")
        if item.get("type") == "function_call":
            raw = item.get("arguments", "")
            try:
                return name, json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return name, raw
        return name, item.get("input", "")  # custom_tool_call: raw input string


class ClaudeTranslator(Translator):
    """Claude transcript → Trace — bespoke; overrides the template.

    Claude emits separate entries per thinking / text / tool_use, folds queued
    "steer" messages into the request window, splits a turn's duration across
    multiple tool calls, and carries token usage — richer than the event
    template, so it builds spans directly via ``Trace``'s ``add_*`` methods.
    """

    agent = "claude"

    def translate(self, records, *, trace_id, session_id, project) -> Trace | None:
        starts = self._prompt_indices(records)
        if not starts:
            return None
        trace = Trace(trace_id, session_id, self.agent, project)
        bounds = starts + [len(records)]
        for k, user_idx in enumerate(starts):
            self._build_turn(trace, records, user_idx, bounds[k + 1])
        return trace if trace.spans else None

    # _build_turn here takes no results arg (Claude looks results up per turn),
    # so it intentionally shadows the template's signature.
    def _build_turn(self, trace, records, user_idx, end_idx) -> None:  # type: ignore[override]
        user_rec = records[user_idx]
        user_msg = (user_rec.get("message") or {}).get("content", "")
        prompt = (
            user_msg if isinstance(user_msg, str) else self._text_and_tools(user_msg)[0]
        )
        root_id = user_rec.get("uuid") or f"turn-{user_idx}"
        root_ts = _parse_ts(user_rec.get("timestamp"))
        root = trace.add_agent_root(
            root_id, "claude_code_conversation", start_time=root_ts, prompt=prompt
        )

        counter = 0
        final_response: str | None = None
        last_ts = root_ts
        for i in range(user_idx + 1, end_idx):
            rec = records[i]
            if (t := _parse_ts(rec.get("timestamp"))) is not None:
                last_ts = t
            if rec.get("type") != _ROLE_ASSISTANT:
                continue
            msg = rec.get("message") or {}
            ts = _parse_ts(rec.get("timestamp"))
            text, tools = self._text_and_tools(msg.get("content"))
            nxt = self._next_timestamp(records, i, stop=end_idx)
            duration = (
                (nxt - ts)
                if (ts is not None and nxt is not None and nxt > ts)
                else _DEFAULT_SPAN_SECONDS
            )

            if text.strip() and not tools:
                final_response = text
                trace.add_llm_span(
                    f"{root_id}-{counter}",
                    parent_id=root_id,
                    start_time=ts,
                    end_time=(ts + duration) if ts is not None else None,
                    request=self._window_messages(records, i),
                    response=[_text_block(text)],
                    model=msg.get("model"),
                    usage=_usage(msg.get("usage")),
                )
                counter += 1

            if tools:
                results = self._tool_results_after(records, i)
                tool_duration = duration / len(tools)
                for idx_t, tu in enumerate(tools):
                    tid = tu.get("id", "")
                    t_start = (ts + idx_t * tool_duration) if ts is not None else None
                    trace.add_tool_span(
                        f"{root_id}-{counter}",
                        parent_id=root_id,
                        start_time=t_start,
                        end_time=(
                            (t_start + tool_duration) if t_start is not None else None
                        ),
                        name=tu.get("name", "unknown"),
                        tool_input=tu.get("input", {}),
                        result=results.get(tid, ""),
                        tool_id=tid,
                    )
                    counter += 1

        root["end_time"] = last_ts
        if final_response is not None:
            root["attributes"]["response"] = final_response

    # -- Claude grammar (ported from mlflow claude_code; see NOTICE) ---------

    def _prompt_indices(self, records) -> list[int]:
        return [i for i in range(len(records)) if self._is_user_prompt(records, i)]

    @staticmethod
    def _is_user_prompt(records, i) -> bool:
        rec = records[i]
        if rec.get("type") != _ROLE_USER or rec.get("toolUseResult"):
            return False
        prev = records[i - 1] if i > 0 else None
        prev_tur = prev.get("toolUseResult") if isinstance(prev, dict) else None
        if isinstance(prev_tur, dict) and prev_tur.get("commandName"):
            return False
        content = (rec.get("message") or {}).get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            if content[0].get("type") == "tool_result":
                return False
        if isinstance(content, str):
            if "<local-command-stdout>" in content or not content.strip():
                return False
        return bool(content)

    @staticmethod
    def _text_and_tools(content) -> tuple[str, list[dict]]:
        text, tools = "", []
        if isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    text += b.get("text", "")
                elif b.get("type") == "tool_use":
                    tools.append(b)
        elif isinstance(content, str):
            text = content
        return text, tools

    @staticmethod
    def _next_timestamp(records, idx, stop=None) -> float | None:
        end = len(records) if stop is None else stop
        for j in range(idx + 1, end):
            if (t := _parse_ts(records[j].get("timestamp"))) is not None:
                return t
        return None

    def _block_from_raw(self, raw) -> dict | None:
        btype = raw.get("type")
        if btype == "text":
            return _text_block(raw.get("text", ""))
        if btype == "tool_use":
            return _tool_use_block(
                raw.get("name"), raw.get("input") or {}, raw.get("id")
            )
        if btype == "tool_result":
            return _tool_result_block(raw.get("content"), raw.get("tool_use_id"))
        return None  # thinking / unknown — no conversational payload

    def _message_from_record(self, record) -> dict | None:
        msg = record.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")
        if not role or not content:
            return None
        if isinstance(content, str):
            return _message(role, [_text_block(content)])
        blocks = [b for raw in content if (b := self._block_from_raw(raw))]
        return _message(role, blocks) if blocks else None

    def _window_messages(self, records, current_idx) -> list[dict]:
        """The user/tool entries since the previous text-bearing assistant turn."""
        out: list[dict] = []
        for i in range(current_idx - 1, -1, -1):
            rec = records[i]
            if rec.get("type") == _ROLE_ASSISTANT:
                text, _ = self._text_and_tools(
                    (rec.get("message") or {}).get("content")
                )
                if text.strip():
                    break
            if (
                rec.get("type") == _TYPE_QUEUE_OP
                and rec.get("operation") == "enqueue"
                and (steer := rec.get("content"))
            ):
                txt = steer if isinstance(steer, str) else str(steer)
                out.append(_message(_ROLE_USER, [_text_block(txt)]))
                continue
            if (m := self._message_from_record(rec)) is not None:
                out.append(m)
        out.reverse()
        return out

    @staticmethod
    def _tool_results_after(records, start_idx) -> dict:
        results: dict = {}
        for i in range(start_idx + 1, len(records)):
            rec = records[i]
            if rec.get("type") != _ROLE_USER:
                continue
            content = (rec.get("message") or {}).get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        if tid := b.get("tool_use_id"):
                            results[tid] = b.get("content", "")
        return results


# ===========================================================================
# TraceCollector — the driver (read → translate → hand to consumers)
# ===========================================================================

# An agent appears here once both its reader and translator exist; the collector
# runs for any agent in the table and is simply absent for the rest.
_PIPELINES = {
    "claude": lambda pd, pr: (ClaudeReader(pd, projects_root=pr), ClaudeTranslator()),
    "codex": lambda pd, pr: (CodexReader(pd), CodexTranslator()),
    "goose": lambda pd, pr: (GooseReader(pd), GooseTranslator()),
    "opencode": lambda pd, pr: (OpenCodeReader(pd), OpenCodeTranslator()),
    "cline": lambda pd, pr: (ClineReader(pd), ClineTranslator()),
}


def make_trace_collector(
    agent,
    project_dir,
    project,
    session_id,
    tracking_uri,
    *,
    projects_root: Path | None = None,
    extra_consumers: list | None = None,
    source=None,
) -> "TraceCollector | None":
    """Build the collector for ``agent``, or ``None`` if no pipeline is registered.

    The MLflow logger is always the first consumer (observability is universal);
    ``extra_consumers`` (e.g. a :class:`~dsagt.memory.MemoryExtractor` when
    episodic memory is enabled) are appended, each acking independently.

    ``source`` pins the reader to a specific session (the startup catch-up passes
    the *previous* session's recorded :meth:`Reader.active_source` token), so it
    re-reads that exact session instead of whatever is newest now — uniformly
    across all agents (transcript path, DB session id, or session-dir name).
    """
    builder = _PIPELINES.get(agent)
    if builder is None:
        return None
    reader, translator = builder(project_dir, projects_root)
    if source is not None:
        reader.pin(source)
    # Imported here (not at module top) so traces stays a lean leaf — the MLflow
    # logger drags in mlflow, the heaviest thing in the pipeline.
    from dsagt.observability import MLflowSink

    consumers = [MLflowSink(tracking_uri, project), *(extra_consumers or [])]
    return TraceCollector(
        reader,
        translator,
        project=project,
        session_id=session_id,
        project_dir=project_dir,
        consumers=consumers,
    )


class TraceCollector:
    """Periodically read the session, translate it, and hand it to consumers.

    A *consumer* is anything with a ``name`` and a ``write(trace)`` — the MLflow
    logger and the memory indexer both qualify (no shared base needed).  Each
    consumer keeps its own ack set (``.dsagt/trace_acks_<name>.json``), keyed by
    session-qualified turn id (``<session_id>:<span_id>``) so the per-transcript
    ``turn-N`` indices can't collide across sessions in the shared file.  A
    re-pass or an N+1 catch-up can only waste work, never double-log or lose a
    turn, and a failing consumer holds back only its own mark.

    Completeness watermark: a periodic pass emits only *completed* turns (all but
    the still-open last one); the deferred final turn flushes when a later prompt
    bounds it or at end-of-session (``include_last=True``).  An OS file lock
    serializes overlapping passes against the shared ack files.
    """

    def __init__(
        self, reader, translator, *, project, session_id, project_dir, consumers
    ):
        self._reader = reader
        self._translator = translator
        self._project = project
        self._session_id = session_id
        self._project_dir = Path(project_dir)
        self._consumers = list(consumers)
        self._dsagt_dir = self._project_dir / ".dsagt"
        self._lock = threading.Lock()

    def active_source(self):
        """The reader's session token (see :meth:`Reader.active_source`), or
        ``None``.  Recorded in ``state.yaml`` so the next session's catch-up can
        pin this exact session — uniform across all agents.
        """
        try:
            return self._reader.active_source()
        except Exception:  # noqa: BLE001 — best-effort; never break the heartbeat
            return None

    def _acks_path(self, name: str) -> Path:
        return self._dsagt_dir / f"trace_acks_{name}.json"

    def _load_acks(self, name: str) -> set[str]:
        try:
            return set(json.loads(self._acks_path(name).read_text()))
        except FileNotFoundError:
            return set()

    def _save_acks(self, name: str, acks: set[str]) -> None:
        self._dsagt_dir.mkdir(parents=True, exist_ok=True)
        self._acks_path(name).write_text(json.dumps(sorted(acks)))

    @contextmanager
    def _lock_file(self):
        self._dsagt_dir.mkdir(parents=True, exist_ok=True)
        with open(self._dsagt_dir / "trace_acks.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def collect(self, *, include_last: bool = False) -> int:
        """Read → translate → hand completed turns to each consumer.

        Returns the number of turns newly delivered to at least one consumer.
        ``include_last=True`` (end-of-session flush / per-turn hook) also emits
        the otherwise-deferred final turn.  Blocking — call via
        ``asyncio.to_thread`` from the event loop.
        """
        with self._lock, self._lock_file():
            records = self._reader.read()
            if not records:
                return 0
            trace = self._translator.translate(
                records,
                trace_id=self._session_id,
                session_id=self._session_id,
                project=self._project,
            )
            if trace is None:
                return 0

            roots = trace.roots()
            candidates = roots if include_last else roots[:-1]
            # Ack keys are session-qualified.  span_id is a per-transcript record
            # index ("turn-N"), so the same ids recur in every session's
            # transcript; a bare span_id would collide across sessions in the
            # shared, never-reset ack file and suppress every turn after the
            # first session.  Qualifying by session id matches the key MLflowSink
            # already uses for its own idempotency (``{trace_id}:{span_id}``).
            key_by_span = {
                r["span_id"]: f"{self._session_id}:{r['span_id']}" for r in candidates
            }
            if not key_by_span:
                return 0

            emitted: set[str] = set()
            for consumer in self._consumers:
                acks = self._load_acks(consumer.name)
                emit_ids = {s for s, key in key_by_span.items() if key not in acks}
                if not emit_ids:
                    continue
                try:
                    consumer.write(trace.subset(emit_ids))
                except Exception as e:  # noqa: BLE001 — per-consumer isolation
                    logger.warning("Trace consumer %r failed: %s", consumer.name, e)
                    continue
                self._save_acks(
                    consumer.name, acks | {key_by_span[s] for s in emit_ids}
                )
                emitted |= emit_ids
            return len(emitted)

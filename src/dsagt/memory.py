"""
Explicit memory store for DSAGT projects.

Manages a YAML file of user-confirmed facts in the project runtime directory.
Designed to be loaded into agent context at session start — every active memory
is visible without the agent searching for it.

The file lives at ``<runtime_dir>/explicit_memories.yaml`` and contains only
active (non-superseded) entries.  Superseded entries are moved to a history
file for audit purposes.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id(text: str, timestamp: str) -> str:
    return hashlib.sha256(f"{text}:{timestamp}".encode()).hexdigest()[:16]


class ExplicitMemory:
    """
    File-backed store for explicit (user-confirmed) facts.

    Parameters
    ----------
    runtime_dir : Path
        Project runtime directory.  The memory file is created here
        on first write.
    """

    FILENAME = "explicit_memories.yaml"
    HISTORY_FILENAME = "explicit_memories_history.yaml"

    def __init__(self, runtime_dir: str | Path):
        self._dir = Path(runtime_dir)
        self._path = self._dir / self.FILENAME
        self._history_path = self._dir / self.HISTORY_FILENAME

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        text = self._path.read_text()
        if not text.strip():
            return []
        data = yaml.safe_load(text)
        return data if isinstance(data, list) else []

    def _save(self, entries: list[dict]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            yaml.dump(entries, default_flow_style=False, sort_keys=False)
            if entries else ""
        )

    def _append_history(self, entry: dict) -> None:
        """Append a superseded entry to the history file."""
        self._dir.mkdir(parents=True, exist_ok=True)
        history = []
        if self._history_path.exists():
            text = self._history_path.read_text()
            if text.strip():
                loaded = yaml.safe_load(text)
                if isinstance(loaded, list):
                    history = loaded
        history.append(entry)
        self._history_path.write_text(
            yaml.dump(history, default_flow_style=False, sort_keys=False)
        )

    def remember(
        self,
        text: str,
        category: str = "",
        session_id: str = "",
        supersedes: str | None = None,
    ) -> dict:
        """
        Store a fact.  Optionally supersede an existing entry.

        Parameters
        ----------
        text : str
            The fact (short declarative statement).
        category : str
            Classification tag.
        session_id : str
            Current session identifier.
        supersedes : str, optional
            ``id`` of an existing memory to replace.

        Returns
        -------
        dict
            ``{"stored": True, "entry_id": ..., "superseded_id": ...}``
        """
        entries = self._load()
        now = _now_iso()
        entry_id = _make_id(text, now)

        superseded_id = None
        if supersedes:
            remaining = []
            for e in entries:
                if e.get("id") == supersedes:
                    superseded_id = supersedes
                    e["superseded_by"] = entry_id
                    e["superseded_at"] = now
                    self._append_history(e)
                else:
                    remaining.append(e)
            if superseded_id is None:
                return {
                    "stored": False,
                    "error": f"Entry '{supersedes}' not found",
                }
            entries = remaining

        entry = {
            "id": entry_id,
            "text": text,
            "category": category,
            "session_id": session_id,
            "timestamp": now,
        }
        entries.append(entry)
        self._save(entries)

        logger.info("Stored explicit memory %s: %s", entry_id, text[:80])
        return {
            "stored": True,
            "entry_id": entry_id,
            "superseded_id": superseded_id,
        }

    def get_all(self) -> list[dict]:
        """Return all active memories."""
        return self._load()

    def get_by_id(self, entry_id: str) -> dict | None:
        """Return a single memory by id, or None."""
        for e in self._load():
            if e.get("id") == entry_id:
                return e
        return None

    def remove(self, entry_id: str) -> dict:
        """
        Remove a memory (move to history).

        Returns
        -------
        dict
            ``{"removed": True, "entry_id": ...}`` or error.
        """
        entries = self._load()
        remaining = []
        removed = None
        for e in entries:
            if e.get("id") == entry_id:
                removed = e
                e["removed_at"] = _now_iso()
                self._append_history(e)
            else:
                remaining.append(e)

        if removed is None:
            return {"removed": False, "error": f"Entry '{entry_id}' not found"}

        self._save(remaining)
        return {"removed": True, "entry_id": entry_id}

    def count(self) -> int:
        """Number of active memories."""
        return len(self._load())

    def render_context(self) -> str:
        """
        Render all active memories as a text block suitable for
        inclusion in agent context.
        """
        entries = self._load()
        if not entries:
            return ""

        lines = ["# Explicit Memories", ""]
        for e in entries:
            cat = f" [{e['category']}]" if e.get("category") else ""
            lines.append(f"- {e['text']}{cat}")
        return "\n".join(lines)

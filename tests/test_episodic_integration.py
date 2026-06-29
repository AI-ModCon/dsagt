"""End-to-end episodic-memory integration: heartbeat → judge → session_memory.

The full Phase-3 chain with nothing faked: a real Claude transcript on disk →
``TraceCollector`` (the heartbeat) → the ``MemoryExtractor`` consumer built by the
server's own ``_episodic_consumers`` wiring → the real ``LocalJudge`` (GGUF
inference) → distilled facts embedded into the ``session_memory`` collection of a
real ``KnowledgeBase`` → retrieved by ``kb.search``.

Marked ``integration``: loads the local embedder and downloads/loads the ~1GB
GGUF judge, so it's deselected from the fast suite (``-m 'not integration'``).
"""

import json

import pytest

from dsagt.knowledge import KnowledgeBase
from dsagt.memory import SESSION_MEMORY_COLLECTION
from dsagt.traces import _transcript_dir, make_trace_collector


def _asst(ts, text):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    }


def _user(ts, text, uuid):
    return {
        "type": "user",
        "timestamp": ts,
        "uuid": uuid,
        "message": {"role": "user", "content": text},
    }


@pytest.mark.integration
def test_heartbeat_distills_a_fact_into_session_memory(tmp_path):
    project_dir = tmp_path / "proj"
    (project_dir / ".dsagt").mkdir(parents=True)

    # A real KB with the local embedder (the project's session_memory lives here).
    kb = KnowledgeBase(index_dir=project_dir / "kb_index", default_embedder="local")

    # Episodic enabled with the Tier-1 local judge — exactly what `dsagt init
    # --episodic` writes — fed through the server's real subscriber builder.
    from dsagt.mcp.server import _episodic_consumers

    config = {
        "episodic": {
            "enabled": True,
            "judge": {"backend": "local", "model": ""},
            "domain_tags": {},
            "outlier_sensitivity": 0.0,
        }
    }
    subs = _episodic_consumers(config, kb, project_dir, "proj:s")
    assert subs and subs[0].name == "memory"

    # A real Claude transcript with one clearly fact-bearing completed turn,
    # then an open turn so the first is a "completed" candidate for the tick.
    proot = tmp_path / "projects"
    tdir = _transcript_dir(project_dir, proot)
    tdir.mkdir(parents=True)
    transcript = tdir / "sess.jsonl"
    records = [
        _user(
            "2026-06-28T15:00:00.000Z",
            "Run fastp on the reads at a Q30 quality threshold.",
            "u1",
        ),
        _asst(
            "2026-06-28T15:00:01.000Z",
            "Done — fastp filtered the reads at Q30; 92% of reads passed and were written to clean.fq.gz.",
        ),
        _user("2026-06-28T15:00:02.000Z", "what next?", "u2"),  # opens turn 2
    ]
    with open(transcript, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    collector = make_trace_collector(
        "claude",
        project_dir,
        "proj",
        "proj:s",
        tracking_uri,
        projects_root=proot,
        extra_consumers=subs,
    )

    # include_last=True flushes both turns; the memory consumer distills them.
    n = collector.collect(include_last=True)
    assert n >= 1

    # Both consumers advanced their own marks independently.
    assert collector._load_acks("mlflow")  # observability sink
    assert collector._load_acks("memory")  # episodic consumer

    # The distilled fact is retrievable from session_memory, tagged as a Tier-1
    # fact from this session.
    hits = kb.search(
        "fastp quality filtering Q30", collection=SESSION_MEMORY_COLLECTION, top_k=5
    )
    assert hits, "expected at least one distilled fact in session_memory"
    chunk = hits[0]["chunk"]
    text = chunk["text"].lower()
    assert "fastp" in text or "q30" in text or "qc" in text
    meta = chunk["metadata"]
    assert meta.get("source_type") == "fact"
    assert meta.get("tier") == "1"
    assert meta.get("session_id") == "proj:s"

    kb.close()

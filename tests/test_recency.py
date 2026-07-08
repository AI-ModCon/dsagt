"""Recency weighting for episodic (session_memory) retrieval.

``_apply_recency`` is a pure re-ranker, so these test the *behavior that matters*
without an embedder: a recent fact edges out a same-relevance stale one, while a
strongly-relevant old fact still beats an irrelevant recent one (the guard that
recency is a bounded boost, never a filter).
"""

from dsagt.knowledge import _apply_recency

DAY = 86400.0
NOW = 1_000_000_000.0


def _r(score, age_days, label):
    ts = NOW - age_days * DAY
    return {"chunk": {"text": label, "metadata": {"ts_epoch": ts}}, "score": score}


def _labels(results):
    return [r["chunk"]["text"] for r in results]


def test_recent_edges_out_same_relevance_stale_fact():
    # Equal base relevance → the newer one wins purely on recency.
    out = _apply_recency(
        [_r(1.0, age_days=30, label="old"), _r(1.0, age_days=0, label="new")],
        half_life_days=14,
        now=NOW,
    )
    assert _labels(out)[0] == "new"


def test_strong_old_fact_still_beats_irrelevant_recent_one():
    # The guard: recency is a bounded boost (≤ +50%), so a much-more-relevant old
    # fact is never buried by a barely-relevant recent one.
    out = _apply_recency(
        [
            _r(1.0, age_days=60, label="relevant_old"),
            _r(0.4, age_days=0, label="recent_junk"),
        ],
        half_life_days=14,
        now=NOW,
    )
    assert _labels(out)[0] == "relevant_old"


def test_half_life_controls_decay_strength():
    # At one half-life the boost halves: factor = 1 + 0.5 * 0.5 = 1.25.
    (one,) = _apply_recency(
        [_r(1.0, age_days=14, label="x")], half_life_days=14, now=NOW
    )
    assert abs(one["recency_factor"] - 1.25) < 1e-9
    # Brand-new → full boost 1.5; very old → ~1.0 (no penalty below relevance).
    (fresh,) = _apply_recency(
        [_r(1.0, age_days=0, label="x")], half_life_days=14, now=NOW
    )
    assert abs(fresh["recency_factor"] - 1.5) < 1e-9
    (ancient,) = _apply_recency(
        [_r(1.0, age_days=3650, label="x")], half_life_days=14, now=NOW
    )
    assert 1.0 <= ancient["recency_factor"] < 1.001


def test_missing_ts_epoch_is_unweighted_not_dropped():
    res = [{"chunk": {"text": "no_ts", "metadata": {}}, "score": 1.0}]
    (out,) = _apply_recency(res, half_life_days=14, now=NOW)
    assert out["recency_factor"] == 1.0
    assert out["chunk"]["text"] == "no_ts"

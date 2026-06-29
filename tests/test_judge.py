"""Tests for the episodic-memory Judge (Phase 3, Tier-1).

The backend ``distill`` is a no-op for now; what's contractual today is the
factory, the lean per-turn prompt, and the tolerant response parser.
"""

import pytest

from dsagt.judge import (
    APIJudge,
    Judge,
    LocalJudge,
    build_distill_prompt,
    parse_distill_response,
)

TAGS = {
    "performance": "Runtime, memory usage, throughput",
    "results": "Output summaries, key findings",
}


def test_create_selects_backends_and_defaults_local():
    assert isinstance(Judge.create("local"), LocalJudge)
    assert isinstance(Judge.create("api"), APIJudge)
    assert isinstance(Judge.create(""), LocalJudge)  # default
    with pytest.raises(ValueError):
        Judge.create("nonesuch")


def test_local_judge_constructs_without_loading_weights():
    # No I/O at construction — the GGUF runtime loads lazily on first distill.
    j = LocalJudge()
    assert j.backend == "local"
    assert j._llm is None
    assert j.distill([], TAGS) == []  # no-op for now


def test_build_prompt_lists_tags_and_renders_turn():
    exchanges = [
        {
            "new_messages": [
                {"role": "user", "content": [{"type": "text", "text": "filter reads"}]}
            ],
            "response": [{"type": "text", "text": "done, 90% passed QC"}],
        }
    ]
    prompt = build_distill_prompt(exchanges, TAGS)
    assert "performance" in prompt and "results" in prompt
    assert "[user] filter reads" in prompt
    assert "[assistant] done, 90% passed QC" in prompt
    assert "[]" in prompt  # the empty escape is advertised


def test_parse_validates_tags_and_strips_fences():
    raw = '```json\n[{"fact": "ran in 3s", "tag": "performance"}]\n```'
    assert parse_distill_response(raw, TAGS) == [
        {"text": "ran in 3s", "tag": "performance"}
    ]


def test_parse_drops_off_taxonomy_and_malformed():
    raw = (
        '[{"fact": "x", "tag": "made_up"}, '  # off-set tag → dropped
        '{"fact": "", "tag": "results"}, '  # empty fact → dropped
        '"not a dict", '  # junk → dropped
        '{"fact": "kept", "tag": "results"}]'
    )
    assert parse_distill_response(raw, TAGS) == [{"text": "kept", "tag": "results"}]


def test_parse_empty_escape():
    assert parse_distill_response("[]", TAGS) == []
    assert parse_distill_response("", TAGS) == []


@pytest.mark.integration
def test_local_judge_distills_real_model():
    """End-to-end LocalJudge over the real GGUF (downloads ~1GB on first run).

    Deselected by default (``-m 'not integration'``); exercises grammar-
    constrained inference: a fact-bearing turn yields well-formed, closed-set
    facts, and a trivial turn hits the empty escape.
    """
    from dsagt.memory import STOCK_CATEGORIES

    judge = LocalJudge()
    fact_turn = [
        {
            "new_messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Run fastp at a Q30 quality threshold.",
                        }
                    ],
                }
            ],
            "response": [
                {
                    "type": "text",
                    "text": "Done — fastp filtered at Q30; 92% of reads passed.",
                }
            ],
        }
    ]
    facts = judge.distill(fact_turn, STOCK_CATEGORIES)
    assert isinstance(facts, list) and facts  # deterministic (temp 0): non-empty
    for f in facts:
        assert set(f) == {"text", "tag"}
        assert f["tag"] in STOCK_CATEGORIES  # grammar-enforced closed set
        assert f["text"].strip()

    trivial = [
        {
            "new_messages": [
                {"role": "user", "content": [{"type": "text", "text": "thanks!"}]}
            ],
            "response": [{"type": "text", "text": "You're welcome."}],
        }
    ]
    assert judge.distill(trivial, STOCK_CATEGORIES) == []

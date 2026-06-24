"""Zero-dependency keyword token-overlap scorer for skill discovery.

A faithful reimplementation (not an import) of the Genesis Skills
``skill-search`` engine (``skill_search/catalog.py``: ``_score_skill`` /
``rank_skills``), Apache-2.0, gitlab.osti.gov/genesis/genesis-skills.  This is
the fallback ranker used by :class:`dsagt.skill_discovery.SkillRouter` when no
embedder / KB is configured: keyword overlap only, stdlib only, deterministic.

Scoring (per skill, against a query) — matching Genesis exactly:

* +2 for each query token that also appears in the skill **name**
* +1 for each query token that also appears in the **description**
* then **at most one** substring bonus (mutually exclusive, in priority order):
  +6 if the query equals the name, else +4 if it is a substring of the name,
  else +2 if it is a substring of the description

Tokens are casefolded ``\\w+`` runs with hyphens split, single-character tokens
and stopwords dropped.  Ties break by name (ascending); below ``min_score`` are
dropped.
"""

from __future__ import annotations

import re

#: Stopword set — kept identical to Genesis so ranking parity holds.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "please",
        "the",
        "this",
        "to",
        "use",
        "using",
        "with",
    }
)

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Casefolded word tokens (hyphens split), single-char + stopwords removed."""
    normalized = (text or "").casefold().replace("-", " ")
    return {
        t for t in _TOKEN_RE.findall(normalized) if len(t) > 1 and t not in _STOPWORDS
    }


def score_skill(query: str, name: str, description: str) -> float:
    """Token-overlap score of one skill against *query* (0.0 = no match)."""
    qtokens = _tokens(query)
    normalized_query = (query or "").casefold().strip()
    if not qtokens and not normalized_query:
        return 0.0

    score = 2 * len(qtokens & _tokens(name)) + len(qtokens & _tokens(description))

    if normalized_query:
        name_l = (name or "").casefold()
        if normalized_query == name_l:
            score += 6
        elif normalized_query in name_l:
            score += 4
        elif normalized_query in (description or "").casefold():
            score += 2
    return float(score)


def rank_skills(
    query: str, skills, top_k: int | None = 8, min_score: int = 1
) -> list[tuple[dict, float]]:
    """Rank *skills* (dicts with ``name`` + ``description``) against *query*.

    Returns ``[(skill, score), ...]`` for skills scoring at least *min_score*,
    sorted by score descending then name ascending, truncated to *top_k* (all
    when *top_k* is ``None``).
    """
    scored: list[tuple[dict, float]] = []
    for s in skills:
        sc = score_skill(query, s.get("name", ""), s.get("description", ""))
        if sc >= min_score:
            scored.append((s, sc))
    scored.sort(key=lambda kv: (-kv[1], (kv[0].get("name") or "")))
    return scored[:top_k] if top_k is not None else scored

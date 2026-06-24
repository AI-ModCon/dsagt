# Design Note — Genesis Skills vs DSAGT skill discovery

**Status:** implemented (2026-06-23). Sections 0–6 are the original close read of
the Genesis Skills discovery engine vs DSAGT's. Sections 7–9 record the agreed
design as it stood mid-plan. **§10 records what actually shipped and supersedes
the tier-3 / progressive-disclosure / recency-queue parts of §7.4–7.6 and §9** —
a research pivot (every supported agent natively discovers SKILL.md) collapsed
those. Read §10 for the final architecture; §7–9 are kept for design history.

**Date:** 2026-06-23
**Author:** comparison drafted with Claude Code

- Genesis Skills: <https://gitlab.osti.gov/genesis/genesis-skills>
  (the `skill-search/` engine + a curated 74-skill catalog under `skills/`)
- DSAGT skill discovery: [skills_catalog.py](../src/dsagt/commands/skills_catalog.py),
  [registry.py](../src/dsagt/registry.py) (`SkillRegistry`),
  [registry_server.py](../src/dsagt/commands/registry_server.py) (`search_skills`/`install_skill`),
  [knowledge_server.py](../src/dsagt/commands/knowledge_server.py) (`add_skill_source`/`list_skill_sources`)

---

## 0. Relationship (important framing)

This is **not a competitor** — it's a sibling DOE effort. Genesis Skills is part
of the "Genesis Mission" (`genesis@osti.gov`, gitlab.osti.gov/genesis), and its
catalog includes a **`modcon-skills/`** category (`datacard-generator`,
`croissant-validator`, `hdmf-schema-builder`) — i.e. our own BASE-Data/ModCon
skills. DSAGT (`AI-ModCon`, BASE-Data team) and Genesis are in the same orbit.

DSAGT already **consumes** Genesis as a catalog source (the `genesis` entry in
`KNOWN_SOURCES`, subdir `skills`, ~72 of 74 skills index cleanly; 2 are skipped
for malformed upstream YAML). So the natural relationship is **complementary**:
DSAGT = semantic, multi-source, platform-integrated layer that *indexes* Genesis
(and others); Genesis = portable, standard-conformant content + lightweight
discovery.

---

## 1. What Genesis built

A focused, polished, single-purpose **skill-discovery engine** (`skill-search/`)
plus a curated catalog (74 skills across 10+ domains).

- **Search method:** pure-Python **keyword token-overlap** scoring
  (`skill-search/skill_search/catalog.py`). Name-token matches ×2,
  description-token matches ×1, plus exact/substring bonuses (+6 exact name,
  +4 substring name, +2 substring description), stopword filtering, deterministic
  tie-break by name. **Zero dependencies, no embeddings, no DB, no model
  download.**
- **Two discovery strategies:**
  1. `--query` → top-k keyword matches (keeps full catalog out of context).
  2. `--load-all` → progressive disclosure: compact index of *all* skills
     (name/description/path) upfront, load `SKILL.md` body on demand.
  3. `--include-prompt` → emits an `<available_skills>` XML block for direct
     system-prompt injection.
- **Discovery:** filesystem recursion over nested `skills/<domain>/<skill>/`,
  merged by name with **override semantics** (central catalog + user roots; later
  roots win). No persistent index — re-scanned each call.
- **The engine is itself a Skill** (`SKILL.md`, `allowed-tools: Bash`): the agent
  runs it via Bash and gets JSON back. **No server.**
- **Distribution:** `unpack.sh` flattens/symlinks skills into an agent's native
  dir across **multiple harnesses** (Claude Code, Gemini, Codex `.agents/skills/`);
  LangChain/LangGraph import the `skill_search` package directly.
- **Standard conformance:** explicitly follows the **agentskills.io open
  standard** (`skill_spec.md` mirrors the spec), points at `skills-ref validate`,
  and has real license hygiene (Apache-2.0, `NOTICE` inventory, per-subtree
  licenses for vendored third-party skills).
- **Engineering polish:** standalone package (`catalog`/`frontmatter`/`models`/
  `prompt`/`tooling`/`errors`), ~500 lines of unit tests, layered deployment
  resolution (`--central-root` flag → `SKILL_SEARCH_CENTRAL_ROOT` env → sibling
  `../skills`).

---

## 2. What DSAGT has

Skill discovery is **one feature inside a platform** (KB, tool registry,
provenance, observability, memory), exposed over MCP.

- **Search method:** **semantic embeddings + BM25 hybrid** over ChromaDB
  (local `bge-small-en-v1.5` or hosted via LiteLLM; `route.json` `hybrid: true`,
  `bm25.pkl` present). Better recall on natural-language queries.
- **Index:** persistent **per-source ChromaDB collections**
  (`skills_catalog__<slug>`), built by `sync_source` (clone → index). Sparse
  clone via `subdir`. Machine-global cache at `~/.dsagt/.skill_sources/`.
- **Sources:** many **remote** sources (GitHub *and* GitLab), curated
  `KNOWN_SOURCES` (`scientific`, `anthropic`, `antigravity`, `composio`,
  `genesis`) **plus any git URL / `owner/repo`**.
- **Runtime:** MCP server tools the agent calls directly — `search_skills`,
  `add_skill_source`, `list_skill_sources`, `install_skill` — plus the
  `dsagt skills sync/add/list/search` CLI.
- **Two tiers:** *installed* skills (mirrored into native `.claude/skills/` at
  init/start, auto-invocable) vs *catalog* skills (indexed, not in context).
- **Dependencies:** requires an embedder (~130 MB local model or an API key).

---

## 3. Side-by-side

| Dimension | Genesis `skill-search` | DSAGT |
|---|---|---|
| Search | Keyword token-overlap, deterministic | Semantic embeddings + BM25 hybrid |
| Index | None; re-scans filesystem each call | Persistent per-source ChromaDB |
| Sources | One filesystem tree (its own repo) | Many remote sources (GitHub + GitLab), curated + arbitrary URL |
| Runtime | A Bash-invoked Skill; no server | MCP server tools + CLI |
| Dependencies | Zero (stdlib only) | Embedder (~130 MB local or API) |
| Scope | Just skill discovery/distribution | One feature in a platform (KB/registry/provenance/memory) |
| Standard | agentskills.io conformant + validation | Own SKILL.md convention (close, not formalized) |
| Multi-harness install | `unpack.sh` (Claude/Gemini/Codex) | Mirrors to `.claude/skills/` (per-agent) |
| Context modes | Search top-k, full index, prompt-XML | Search returns summaries; catalog kept out of context |
| Tests/packaging | Dedicated package + ~500 LOC unit tests | Solid, embedded in the larger codebase |
| License hygiene | NOTICE + per-subtree licenses | Not tracked on `install_skill` copy |

---

## 4. Honest assessment

**Where Genesis is better developed (the narrow slice):**
- Open-standard conformance + validation tooling.
- Portability — no server, works across harnesses, the engine ships *as a skill*.
- Zero-dependency, deterministic, instant — no embedder download, no API, reproducible.
- Cleaner standalone package with tests; proper license/attribution for vendored skills.
- A large curated catalog ready to go.

**Where DSAGT is stronger (what matters for the platform):**
- Semantic + hybrid search beats keyword overlap on fuzzy/natural queries.
  (Keyword overlap misses synonymy: "submit a batch job" won't score `slurm`
  unless tokens literally overlap.)
- Multi-source, persistently-indexed catalog federating many remote hosts —
  Genesis searches one tree; DSAGT federates Genesis *plus* K-Dense, Anthropic, …
- Integration: skills live alongside the KB, tool registry, provenance, and
  memory in a reproducible pipeline workflow.

---

## 5. Borrowables (candidate work items, prioritized)

1. **Keyword fallback for `search_skills` when no embedder is configured.**
   Today `search_skills` dead-ends with "requires a configured knowledge base"
   when `kb is None`. Genesis's token-overlap scorer
   (`catalog.py:_score_skill` / `rank_skills`) is exactly the zero-dependency
   fallback that would make search work without the 130 MB model. **Highest
   value / lowest cost**; directly fixes the no-embedder gap.
2. **Adopt the agentskills.io standard explicitly** for DSAGT's SKILL.md
   (already ~compatible) and add a validation step (`skills-ref validate` or
   equivalent). Improves interop — and we now feed from a standard-conformant
   source.
3. **Progressive-disclosure prompt block** (`<available_skills>` XML) — a cheap
   mode for small catalogs that skips the index entirely.
4. **License / NOTICE hygiene on `install_skill`.** When a (possibly
   third-party) catalog skill is copied into a project, preserve its `LICENSE`.
   Genesis tracks this per-subtree; DSAGT currently doesn't.
5. **Lean into complementarity.** Keep DSAGT as the semantic, multi-source,
   platform-integrated layer that indexes Genesis (and others). Consider
   coordinating with the Genesis team — modcon-skills already overlaps, so
   there's a shared-content story.

---

## 6. Open questions for the merge decision (pick up later)

- Do we want DSAGT to *re-export* an agentskills.io-conformant catalog (so other
  tools, incl. Genesis `skill-search`, can consume DSAGT's skills)?
- Should the keyword fallback (#1) be a permanent low-tier search mode (fast,
  cheap, deterministic) selectable even when an embedder *is* available?
- Is there appetite to upstream DSAGT/ModCon skills into Genesis rather than
  maintain parallel copies (the `modcon-skills/` overlap)?
- Packaging: is the Genesis `skill_search` Python package worth depending on
  directly for the fallback, or do we reimplement the (small) scorer to avoid a
  dependency on an external repo?

---

## 7. Decided plan (2026-06-23)

The architecture below is **not a rewrite** — DSAGT's existing chain already *is*
the curation + per-project-install model we want. The plan is a set of insertions
on top of it.

### 7.1 Earmarked borrowables (committed)

1. **Keyword fallback for `search_skills` when `kb is None`.** Reimplement (do
   **not** depend on) Genesis's `_score_skill` / `rank_skills` token-overlap
   scorer — ~50 LOC, vendored with attribution. Insert it *before* the
   `kb is None` dead-end at `registry_server.py:324`. It reads frontmatter off
   the on-disk clone cache (`~/.dsagt/.skill_sources/`) + bundled skills, which
   already exists because `sync_source` clones even when `kb is None`
   (`skills_catalog.py:218`). No embedder, no separate index.
   *Status (built):* `src/dsagt/skill_keyword.py` mirrors Genesis exactly —
   weights (name ×2 / desc ×1), **mutually-exclusive** substring bonus tiers
   (+6 exact / +4 name / +2 desc, via `elif`), the same stopword set, and the
   casefold-`\w+`-hyphen-split tokenizer that drops single-char tokens.
   Verified against the upstream source, not just the prose spec. It is strictly
   the *no-KB* path; when a KB exists the router uses ChromaDB (whose hybrid mode
   already includes BM25), so the scorer and BM25 are mutually exclusive and
   never double-rank.
2. **License / NOTICE hygiene on `install_skill`.** `install_into_project`'s
   `copytree` (`skills_catalog.py:313`) already carries a *skill-dir-local*
   LICENSE. The gap is the source repo's **root NOTICE / per-subtree license
   provenance** — capture it at sync time (Stage A) and stamp it at install.

### 7.2 Shipped-skill cut

DSAGT ships only two skills today — `datacard-generator` (frontmatter name
`generating-datacards`) and `skill-creator` — plus one tool (`scan_directory`).

- **Strike `datacard-generator`** from the repo; it lives in Genesis
  `modcon-skills/`. Users get it via `add_skill_source genesis` (catalog tier).
  **Verify the Genesis copy exists and matches before deleting**, then leave a
  pointer. This makes Genesis the canonical home (open question #3 becomes a real
  coordination dependency).
- **Keep `skill-creator`** as the single minimal shipped skill: it's
  *infrastructure*, not domain content, so it doesn't belong in Genesis's curated
  domain catalog; it's self-referential (the harness can scaffold test skills with
  it); and it's stable. Shipped skill + `scan_directory` tool exist primarily as
  **test-harness fixtures**, since no shipped tool does more than wrap standard CLI.

### 7.3 Current MCP call chain (the anchor)

```
add_skill_source / list_skill_sources   →  search_skills        →  install_skill            →  dsagt start
        (expose catalogs)                   (select a subset)       (draw into project)         (activate natively)
        knowledge_server                    registry_server         registry_server →           agents/claude.py
                                                                     skills_catalog              _mirror_skills_to
```

- **Stage A — Expose.** `add_skill_source` → `resolve_source` → `sync_source`:
  sparse-clone by `subdir` into `~/.dsagt/.skill_sources/<slug>/`, then
  `index_catalog` **wipes + rebuilds** that one source's
  `skills_catalog__<slug>` ChromaDB collection; persists the source to
  `dsagt_config.yaml`. `list_skill_sources` reports `KNOWN_SOURCES` + synced
  state. (`skills_catalog.py:185`, `knowledge_server.py:609`)
- **Stage B — Select.** `search_skills` searches `SKILLS_COLLECTION` (installed)
  **plus every** `skills_catalog__<slug>` collection, merges by score, returns
  top-k tagged `[installed]` / `[catalog · install_skill to add]`. Dead-ends when
  `kb is None` (except exact `skill_name`). (`registry_server.py:305`)
- **Stage C — Draw.** `install_skill` → `find_catalog_skill` (cross-source
  ambiguity guard) → `install_into_project` copies the skill dir into
  `<project>/skills/<name>/`, re-indexes it as `registered`.
  (`registry_server.py:390`, `skills_catalog.py:296`)
- **Stage D — Activate.** Next `dsagt start` mirrors `<project>/skills/` →
  `.claude/skills/` for native auto-invocation — **only `claude.py` does this**;
  goose/cline/roo/codex have no native mirror. (`agents/claude.py:182`,
  `agents/base.py:251`)

Installed skills promoted via `install_skill` become part of the **core installed
set** for future sessions in that project.

### 7.4 Tiering & backend selection (the core design)

> ⚠️ **Superseded by §10.** Tiers 2/3 and the budget threshold below assumed some
> agents lack native skill discovery. Research proved otherwise — *all* supported
> agents are native, so tier 3 (and the disclosure block + recency queue it
> motivated) was never built. The *catalog vs installed* split and the
> *ChromaDB-or-keyword* backend selection survive; the rest is history.


Progressive disclosure and ChromaDB are **not competitors** — they sit on
different axes:

- **What is disclosed:** installed (core, project) vs. catalog (federated, remote).
- **How the index is produced:** *full dump* (deterministic, no query) vs.
  *query-driven selection* (needs a query, returns top-k).

Both produce the **same `<available_skills>` block** (the agentskills.io-style
output contract). Backend is chosen by **context budget**, not by tier:

> Full-dump while the disclosed set fits the budget; switch to query-driven
> *selection* when it doesn't. "Selection" = **ChromaDB if an embedder exists,
> Genesis keyword scorer if not.** ("Fall back on ChromaDB" is shorthand — the
> real fallback is to *selection*, whose backend is itself tiered.)

Three operating tiers:

1. **Catalog (any harness)** — *always* query-driven. Unbounded (Genesis +
   Anthropic + …), so never full-dumped. ChromaDB top-k, or keyword scorer when
   no embedder.
2. **Installed, native harness (Claude)** — dsagt **defers** to the harness.
   `_mirror_skills_to` populates `.claude/skills/`; Claude's own runtime injects
   names/descriptions and lazy-loads bodies (note the `_NATIVE_DESCRIPTION_CAP`
   truncation at `base.py:231` — *Claude's* limit, not dsagt's). dsagt emits no
   block here, so its budget threshold never fires.
3. **Installed, non-native harness (goose/cline/roo/codex)** — dsagt **is** the
   producer of the block, because the harness has no native discovery. Full-dump
   the installed set into the agent's instructions file until the context budget
   is hit; past that, drop to a short pointer + query-driven selection. See 7.5.

### 7.5 The non-native third tier, by example

On goose/cline/roo/codex an installed skill lands in `<project>/skills/<name>/`
but **nothing auto-injects it** — today the agent only finds it by calling
`search_skills`. The third tier closes that gap: dsagt emits the
progressive-disclosure block the harness won't.

*Goose, 5 installed skills* (`skill-creator`, `generating-datacards`,
`slurm-submit`, `croissant-validator`, `fastq-qc`): at `dsagt start`, dsagt writes
an `<available_skills>` block (name + description + path per skill) into
`.goosehints`. ~5 × 40 ≈ **200 tokens** → full-dump. Goose passively knows all
five and reads a SKILL.md body on demand. No embedder, no query.

*Same project, 100 installed skills*: the block is now ~4,000 tokens **every
session**. Past the budget, dsagt stops full-dumping, leaves a short pointer
("call `search_skills`…"), and lets selection carry it (ChromaDB top-k, or keyword
scorer). The agent pulls ~3 relevant skills per task instead of carrying all 100.

This threshold fires **only** in tier 3 — the one case where dsagt both produces
the block and pays its context cost. Codex graduates tier 3 → tier 2 in this pass
by mirroring into its natively-discovered, **project-local** `.agents/skills/`
(§9.7), so it stops going through dsagt's threshold; goose/cline/roo remain
tier 3.

### 7.6 The discovery router (the consolidating abstraction)

All of the policy in 7.4–7.5 — *which backend, which tier, full-dump vs select,
installed vs catalog, defer-to-harness vs emit* — lives in **one** place rather
than smeared across the MCP handler, `base.py`, and each agent file. Otherwise the
decision tree gets re-implemented and drifts at every call site.

**Home:** new module `src/dsagt/skill_discovery.py` (a *module*, not a command),
class `SkillRouter`. It **owns policy and delegates execution.**

```python
class SkillRouter:
    def __init__(self, *, kb, skill_registry, project_dir, agent, config): ...

    def search(self, query=None, *, top_k=8, tag=None, skill_name=None) -> str:
        # Stage B. Owns backend choice (ChromaDB vs keyword vs full-dump),
        # installed+catalog merge, no-embedder fallback, rendering.

    def disclosure_block(self) -> str | None:
        # Stage D. Owns tier resolution + budget threshold. Returns None when
        # the harness owns the tier (native).

    def list_sources(self) -> list[dict]:
        # Stage A view. KNOWN_SOURCES + per-collection indexed count + synced?
        # one consistent view for both the MCP handler and the CLI.
```

- **Owns:** backend selection (`_select`: kb→ChromaDB, else keyword scorer),
  tier resolution, budget threshold (`_mode_for_installed`), and the single
  `<available_skills>` renderer (`_render`, used by *both* `search` and
  `disclosure_block`).
- **Delegates (unchanged):** `kb.search`, `sync_source` / `install_into_project`
  (`skills_catalog`), `_discover_skill_dirs` / `_parse_frontmatter`,
  `_mirror_skills_to`. It is a coordinator, not a reimplementation.
- **Three thin call sites:** `registry_server._handle_search_skills` →
  `router.search`; each agent's `write_dynamic` → `router.disclosure_block`; and
  the **CLI** `dsagt skills search/list` (`cli.py:_cmd_skills`) → `router.search`
  / `router.list_sources`. The CLI is the decisive case: `cli.py:485-509` is
  *already* a drifted copy of the `_handle_search_skills` merge logic (no `tag`,
  no exact-`skill_name`, no `kb is None` path, different render). The router
  collapses both copies into one.
- **`list_skill_sources` exposure status** (`knowledge_server.py:609`) also folds
  into `router.list_sources()` — today the MCP handler and CLI `skills list
  --catalog` compute "what's synced" differently.
- **Nativeness becomes explicit:** each agent declares `native_skills: bool`
  (claude=True, codex=True; goose/cline/roo=False) instead of it being implicit in
  "only claude calls `_mirror_skills_to`." The router reads the flag for tier 2 vs
  3. Codex is tier 2 via a **project-local** `.agents/skills/` mirror only (see §9.7).

**Materialize vs. disclose (don't conflate).** Two separate jobs, both keyed off
`native_skills`:

- *Materialize* = put skill files where the agent expects them. **dsagt does this
  for every tier.** Native (claude): `_mirror_skills_to` → `.claude/skills/`.
  Non-native: skills just live in `<project>/skills/` from install (no native dir).
- *Disclose* = make the agent aware of them in context. Native: the harness does
  it (router returns `None`). Non-native: `router.disclosure_block()` emits the
  block.

So the router defers tier-2 **disclosure** to Claude — **not** materialization.
dsagt still mirrors into `.claude/skills/`. The agent's `write_dynamic` is
`if native_skills: _mirror_skills_to(...) else: write(router.disclosure_block())`.
(`_mirror_skills_to` stays outside the router as materialization *execution*, not
because dsagt is hands-off for native harnesses.)

**Recency queue (dedup + un-bury).** SkillRouter owns a length-bounded,
session-scoped FIFO of recently-exposed skill names. Skills emitted in the
disclosure block or returned by `search` are pushed on; while a skill is *fresh*
(in the queue) `search` suppresses re-surfacing it — it's already in context. As
new exposures push it off the tail it ages out and becomes eligible again (by
then likely buried in the transcript). This unifies the old double-listing and
re-emit threads: `install_skill` marks the new skill fresh, so it isn't
redundantly re-surfaced and no disclosure re-emit is needed. The queue **must be
disk-backed** (`<project>/.dsagt/exposed_skills.json`) because the disclosure
block (agent-setup process) and `search` (MCP server process) are separate
processes that share it. *Refinement:* an explicit query that hits a fresh skill
returns a terse "already available" pointer rather than an empty result.

---

## 8. Implementation surface (router-centric)

Most changes land *inside* `SkillRouter`; call sites stay thin.

| Change | Where | Kind |
|---|---|---|
| `SkillRouter` (backend select, tier, budget, render) | new `src/dsagt/skill_discovery.py` | New module |
| Vendored keyword scorer (genesis-derived, attributed) | new `src/dsagt/skill_keyword.py`, called by `router._select` | New code |
| `search_skills` → router | `registry_server._handle_search_skills` (replace body with `router.search`) | Thin call site |
| `disclosure_block` → router | `agents/{goose,cline,roo,codex}.py` `write_dynamic` (one-line call) | Thin call site |
| **CLI** `skills search/list` → router | `cli.py:_cmd_skills` (`cli.py:485-509` — kill the drifted dup) → `router.search` / `router.list_sources` | Thin call site |
| `list_skill_sources` → router | `knowledge_server.py:609` → `router.list_sources` | Thin call site |
| `native_skills` flag | each `agents/*.py` (declare True/False) | Declaration |
| License/NOTICE capture | `sync_source` + `install_into_project` (`skills_catalog.py:313`) — data-plane, outside router | Augment |
| Strike `datacard-generator` | `src/dsagt/skills/` + `pyproject.toml` package-data | Deletion |

**Stays outside the router (delegated data / execution):** `SkillRegistry`
(installed-skills data + indexing — router reads it); `sync_source` /
`index_catalog` / `find_catalog_skill` / `install_into_project` (catalog
execution); `_mirror_skills_to` / `_truncate_native_description` (tier-2 native
mirror); `resolve_source` / `KNOWN_SOURCES` / `persist_source_to_config` (source
config). `_parse_frontmatter` and `_discover_skill_dirs` are already
single-sourced shared primitives — the router imports them, no move needed.

---

## 9. Resolved decisions (2026-06-23)

1. **Budget threshold** — a **token count** (estimated, e.g. `chars/4` to avoid a
   tokenizer dependency), with a sensible default, as a property of `SkillRouter`
   (per-agent overridable). The router measures it.
2. **Double-exposure** — solved by the recency queue (§7.6), not a static rule.
   While a skill is *fresh* (in the session queue) `search` won't re-surface it.
3. **Mid-session freshness** — **no re-emit.** The agent retains newly installed
   skills in context (`install_skill` returns the SKILL.md), and the queue marks
   them fresh so `search` won't redundantly re-surface them.
4. **agentskills.io conformance** — adopt only the `<available_skills>` **output**
   shape. **Not** full input conformance: strict validation would force
   rewrite-or-exclude on third-party repos and add validation/normalization code —
   *more* complex, not simpler, so it fails the "only if it simplifies" test. Keep
   lenient parsing (parse what we can, skip malformed — as today).
5. **Keyword-scorer latency over large caches** — deferred; revisit only if
   non-KB users hit usability issues.
6. **`datacard-generator`** — strike dsagt's copy; it's stale, the Genesis copy is
   more current. *Pre-deletion check:* confirm the Genesis copy indexes cleanly
   (isn't one of the 2 malformed-YAML skips) before removing ours.
7. **Codex → tier 2 now, project-local only.** Reuse the manifest-tracked
   `_mirror_skills_to` pointed at `<project>/.agents/skills/`. **Never** write
   global `~/.agents/skills/` or `~/.codex` config, and (via `.dsagt-managed.json`)
   never touch user-authored skills — footprint stays confined to the project dir
   and transparent. *Verify:* Codex auto-discovers a project-local `.agents/skills/`.

---

## 10. Implementation outcome (shipped 2026-06-23)

Supersedes the tier-3 / progressive-disclosure / recency-queue parts of §7.4–7.6
and §9.

### 10.1 The research pivot

Before building the agent piece, we verified (primary-source web research, four
independent passes + adversarial check) whether goose/cline/roo actually lack
native skill discovery. **They don't — every supported agent natively
auto-discovers SKILL.md skills:**

| Agent | Native dir (project) | Notes |
|---|---|---|
| claude | `.claude/skills` | on by default |
| codex | `.agents/skills` | project-local (repo-root), on by default |
| goose | `.agents/skills` | built-in extension, on by default (also reads `.goose/`, `.claude/`) |
| cline | `.cline/skills` | **opt-in** — Settings → Features → Enable Skills (v3.48, Jan 2026) |
| roo | `.roo/skills` | v3.38 (May 2026); the "Roo shut down" rumor was false |

**Consequence: tier 3 does not exist.** With no non-native harness there is
nothing to emit a disclosure block *for*. So we did **not** build: the
`<available_skills>` block, the sidecar, the `native_skills` boolean, the budget
threshold, or `disclosure_block()`. And the **recency queue was dropped** — its
sole rationale was the disclosure↔search double-exposure problem, which evaporates
without disclosure. `search` is now stateless.

### 10.2 Final architecture (two concerns, cleanly split)

**Materialization** (agent layer) — `AgentSetup.setup_skills(working_dir, config)`
mirrors installed (bundled + project) skills into each agent's `native_skills_dir`
via the manifest-tracked `_mirror_skills_to`. Called **once centrally** in
`agents/__init__.py:dynamic_agent_record` (covers all agents, BYOA + proxy). Each
agent declares `native_skills_dir` (class attr); gated by `skills.populate_native`
(default true). Codex/goose use the cross-agent `.agents/skills` standard.

**Discovery** (`src/dsagt/skill_discovery.py:SkillRouter`) — stateless:
- `search(query, top_k, tag, skill_name)` — catalog tier + no-embedder keyword
  fallback (`skill_keyword.py`, a faithful Genesis port). Installed skills are
  natively advertised by every agent, so the catalog (not-yet-installed skills)
  is the router's irreplaceable job; the keyword path also covers the
  Cline-skills-disabled case.
- `list_sources()` — consolidated synced/indexed view.
- Three thin call sites: MCP `search_skills`, CLI `skills search/list`,
  `knowledge_server.list_skill_sources`.

### 10.3 What shipped

- `src/dsagt/skill_keyword.py` — Genesis-faithful token-overlap scorer (verified
  against upstream source: weights, `elif` bonus tiers, stopwords, tokenizer).
- `src/dsagt/skill_discovery.py` — stateless `SkillRouter`.
- `agents/base.py` — `native_skills_dir` ClassVar + `setup_skills`; called in
  `dynamic_agent_record`. `native_skills_dir` set on all five agents; claude's
  inline mirror removed (now central).
- Call sites rewired: `registry_server` (search), `cli.py` (search/list),
  `knowledge_server` (list_sources).
- **License/NOTICE capture on install (done).** `install_into_project` →
  `_capture_attribution`: copytree carries skill-local files; ancestor dirs up to
  the cache repo root are walked for `LICENSE` / `NOTICE` / `COPYING` /
  `ATTRIBUTION` (clone_github mirrors repo-root files into the cache even for
  sparse `subdir` clones), nearest wins, and a `PROVENANCE.txt` is stamped.
  `install_skill` surfaces what was preserved.
- **Struck the stale `datacard-generator` shipped skill (done).** Verified first
  via the GitLab API + raw SKILL.md that Genesis's copy
  (`skills/modcon-skills/datacard-generator`, name `generating-datacards`) has
  well-formed frontmatter (not a malformed skip). Removed the dir; `skill-creator`
  is now the only bundled skill. README/docs point users to
  `dsagt skills add <project> genesis` for it.
- Tests: `test_skill_discovery.py` (15), plus `setup_skills` and
  attribution-capture coverage in `test_skills_catalog.py`. **228 passed /
  13 skipped** across all affected suites; ruff + black clean.

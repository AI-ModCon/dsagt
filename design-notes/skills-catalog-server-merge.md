# Design Note — SkillsCatalog encapsulation + MCP server merge

**Status:** shipped — checklist items 1–5 done; only the deferred trace-driven
tool audit (item 6) remains. The implementation/refactor plan that follows from
`genesis-skills-comparison.md` §10 (the locked skills-routing model) and the
`latex/skills-routing.png` diagram. See §5 for what's left.

**Date:** 2026-06-23

---

## 0. Where this comes from

`genesis-skills-comparison.md` settled the *conceptual* model (catalog tier vs
installed tier; `SkillRouter` as the single skill-MCP hub; Skills Catalog
abstracting chroma+cache; Skill Directory for installed+created; `skill-creator`
for authoring). This note is the *engineering* plan to make the code match, in
three parts done in sequence.

Already shipped (Steps A + B, tested):
- **Frontmatter-only catalog indexing** — `index_catalog` embeds `name +
  description + tags`, not the SKILL.md body (progressive-disclosure level 1;
  avoids embedder truncation + signal dilution; consistent with the keyword
  scorer). Description also stored in metadata for clean search summaries.
- **Catalog-only retrieval** — `SkillRouter` searches only `skills_catalog__*`
  collections; the keyword fallback scans only the clone cache. Installed skills
  are no longer search candidates (they're natively discovered).

---

## 1. Part 1 — `SkillsCatalog` module + B2 cleanup

**Goal:** encapsulate all catalog logic behind one module; stop the dead
installed-skill indexing.

- **`SkillsCatalog`** (new module) — *composition over `KnowledgeBase`*, NOT a
  new KB and NOT a copy of the vector primitives. It holds a `KnowledgeBase`
  handle (the host server's existing instance → shared embedder, no second model
  load) + the clone-cache dir, and exposes the skill-domain API:
  - `sync(source, *, force)` — clone + frontmatter-index into `skills_catalog__<slug>`
  - `search(query, *, top_k, tag)` — ChromaDB over catalog collections, keyword
    fallback over the cache when no embedder
  - `install(name, project_dir)` — copy a catalog skill into the project (+ the
    license/PROVENANCE capture already implemented)
  - `list_sources()` — KNOWN_SOURCES + synced/indexed view
  - The skill-specific behavior (frontmatter indexing, keyword fallback, a skill
    `CollectionRoute` preset) lives here; the vector store + embedder are the
    shared KB. ``SkillRouter`` becomes a thin render/MCP facade over it.
- **B2 cleanup** — drop the now-dead indexing into the `skills` collection
  (`SkillRegistry._index_skill` / `save_skill` / `reindex_all`, and the
  `install_skill` re-index). The router no longer searches `skills`, so this is
  wasted embed work. (Touches a few `TestSaveSkill` / `test_reindex_all`
  assertions.)

Server-agnostic: `SkillsCatalog` takes whatever KB it's handed, so it drops
straight into the merged server in Part 2 with no rework.

---

## 2. Part 2 — merge the two MCP servers

**Why:** both `registry_server` and `knowledge_server` already construct their own
`KnowledgeBase` (`registry_server.py:881` + knowledge's `main()`), so the split
is pure duplication — two embedders, two Chroma accesses, and a write-here/
read-there hazard on `skills_catalog__*` (sync in knowledge, search in registry).
The two-process split buys little isolation: the heavy/risky work is already
offloaded (`run_command` → `dsagt-run` subprocess; `kb_ingest` → job thread).

Principle: **process = deployment unit; module = concern boundary.** Merge into
one server; keep `KnowledgeBase` / `ToolRegistry` / `SkillRegistry` /
`SkillsCatalog` / provenance as distinct modules behind one tool-dispatch shell.

**Migration surface (the real cost):**
- Entry points: `dsagt-registry-server` + `dsagt-knowledge-server` → one
  `dsagt-server` (`pyproject [project.scripts]`).
- **Per-agent MCP config generation** — every agent writes *two* server entries
  (`dsagt-registry`, `dsagt-knowledge`) via `_build_mcp_servers_dict` /
  `_mcp_server_args`; collapse to one across all five agents.
- Backward compat: existing projects' configs reference two servers; `dsagt
  start` regenerates to one (write_dynamic overwrite handles it — verify).
- Tests: `test_registry_server`, `test_knowledge_server`, `test_config` (agent
  config gen), `test_server_startup`.

**Ups:** one embedder / one Chroma owner / no cross-process collection hazard;
one MCP server per agent (simpler config, faster startup, one `init_tracing`).
**Downs:** all tools share one process (isolation loss — bounded by the
offloading above); the migration churn above.

---

## 3. Part 3 — tool-surface audit (SEPARATE, evidence-based, later)

Today: **23 MCP tools** (11 registry + 12 knowledge). The agent already sees all
23 (connects to both servers), so the merge is *neutral* on count — proliferation
is pre-existing.

- **Do NOT collapse into `mode=`/`action=` mega-tools.** A union-schema tool with
  a mode enum is harder for a model than distinct, well-named tools (it must pick
  tool *and* mode). Clear names are the discovery signal.
- **Do a trace-driven prune.** Every tool call is in MLflow — audit which tools
  actually get used across real sessions, then remove/defer dead weight (likely
  suspects: `kb_add_vector_db`, `kb_get_suggestions`/`kb_dismiss_suggestion`,
  `kb_job_status`, `reconstruct_pipeline`, maybe `kb_append`).
- Treat as a deliberate pass *after* the merge — not guessed now.

---

## 4. Sequence / checklist

1. [x] `SkillRouter` owns the catalog (compose KB + cache).
   - [x] catalog-only search + cache-only keyword fallback (Step B).
   - [x] frontmatter-only catalog indexing (Step A).
   - [x] `search()` no longer requires a skill registry (catalog needs only KB).
   - [x] add `SkillRouter.sync(source)` (→ `sync_source`) + `SkillRouter.install(name, dir)`
         (→ `install_into_project`) so the router owns all four ops.
   - [x] wire `install_skill` (registry_server) → `router.install`;
         `add_skill_source` (knowledge_server) → `router.sync`.
2. [x] B2: drop dead `skills`-collection indexing — removed
       `SkillRegistry._index_skill` / `reindex_all` + the `save_skill` index call
       + the `install_skill` re-index block. **Also** dropped the bundled-skill
       indexing in `setup_core_kb` (same dead `skills` collection — full cut, not
       in the original enumerated list but the same waste). `SKILLS_COLLECTION`
       kept as a back-compat name only (no reader, no writer). `TestSaveSkill`
       assertions were already file-based (only docstrings mentioned indexing);
       `test_reindex_all` is `ToolRegistry`, untouched. All four skill suites +
       `test_setup_core_kb` green; ruff + black clean.
2b. [x] **`SkillsCatalog` extraction** (folded into this pass). New
       `SkillsCatalog(kb, cache_dir)` class in `commands/skills_catalog.py` owns
       the catalog data plane — `sync` / `install` / `search(→list[dict])` /
       `list_sources` + backend selection (ChromaDB vs keyword fallback).
       `SkillRouter` is now a thin render/MCP facade holding a `SkillsCatalog`
       (built from `kb`/`cache_dir`, or injected via `catalog=` so the server
       shares one). All existing `SkillRouter(kb=…, skill_registry=…)` call sites
       + tests unchanged.
3. [x] Merge servers → one `create_dsagt_server` + `dsagt-server` entry point.
       Extracted `_registry_tools_and_handlers` / `_knowledge_tools_and_handlers`;
       `create_*_server` kept as thin test-facing wrappers; `dsagt_server.py`
       composes both `(tools, handlers)` under one `Server("dsagt")` with a
       type-dispatched `call_tool` (registry str passthrough + knowledge
       dict→json + error wrap) and one shared-KB `main()` (the cross-backend
       guard now lives once, in `_build_kb_from_config`). Old per-server `main()`s
       deleted; their KB-None registry path is gone (merged server requires a KB,
       fails fast on misconfigured `api`).
4. [x] Collapse per-agent MCP config to one `dsagt` entry — `base.py`
       (`_mcp_server_args()` no-arg, flat `_DSAGT_MCP_ALWAYS_ALLOW`,
       `_build_mcp_servers_dict`), `claude` / `goose` (incl. `--with-extension`
       in `interactive_command` + `run_script`) / `codex` / `cline` /
       `opencode` (BYOA + proxy). `roo` rides `_build_mcp_servers_dict`. `info.py`
       span-source buckets + module docstrings updated.
5. [x] Tests + backward-compat. 10 config-shape tests + `test_info` + smoke-test
       comments updated to the single-server shape; new `test_dsagt_server.py`
       (23-tool composition + both return-type contracts). Compat is
       **rebuild-not-migrate** (README upgrade note + cline `.cline-data` caveat);
       no migration code. 338 passed / 13 skipped across affected suites; entry
       point re-registered via `uv sync`.
6. [ ] (later) trace-driven tool-surface audit.

## 5. Current state

**Checklist items 1–5 complete** (this work spanned two sessions). The skills
refactor + server merge has shipped: one `dsagt-server` process, one shared
`KnowledgeBase`, one MCP entry per agent, `SkillsCatalog` owning the catalog data
plane behind a thin `SkillRouter` facade, and the dead `skills`-collection
indexing fully removed. Backward compat is rebuild-not-migrate (README §"MCP
Server" upgrade note). 338 passed / 13 skipped across affected unit suites;
ruff + black clean; `dsagt-server` re-registered.

**Only item 6 remains — the trace-driven tool-surface audit (§3), deliberately
deferred.** It needs real MLflow traces across sessions to decide which of the 23
tools are dead weight; it is *not* a guess-now task. Pick it up when there's
trace data to mine. Likely first suspects (from §3): `kb_add_vector_db`,
`kb_get_suggestions`/`kb_dismiss_suggestion`, `kb_job_status`,
`reconstruct_pipeline`, maybe `kb_append`.

**Key context to re-load in a fresh session:** this note + `genesis-skills-comparison.md`
§7–§10 (the locked model) + the diagram. The long diagram-iteration history is not
needed.

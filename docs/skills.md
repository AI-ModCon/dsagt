# Skills

Skills are instruction-based agent workflows — a directory with a `SKILL.md` and optional reference docs that the agent reads and follows. DSAgt lets the agent discover and install skills from external catalogs on demand, without loading thousands of them into context.

Skills live in `<project>/skills/`. Each is a directory containing a `SKILL.md` file and optional reference documents.

## Two tiers

![DSAgt skill routing](assets/skills-routing.png)

Skills live in **two tiers**, and a single MCP service — the **SkillRouter** — is the one entry point that routes every skill operation between them:

- **Catalog tier** — skills that exist in external repositories but are *not yet installed*. DSAgt federates many sources (`k-dense-ai`, `anthropic`, `antigravity`, `composio`, `genesis`, or any git URL); each is cloned and indexed into its own collection. The agent browses this tier with `search_skills` and manages sources with `add_skill_source` / `list_skill_sources`.
- **Installed + created tier** — skills drawn into the project's Skill Directory (`<project>/skills/`), either installed from the catalog (`install_skill`) or authored in place (e.g. with the bundled `skill-creator`). These are mirrored into each agent's *native* skill directory (`.claude/`, `.agents/`, `.cline/`) at install time (and re-mirrored at `dsagt init`/`start`), where the agent auto-discovers and auto-invokes them.

The lifecycle runs **Discovery** (the router) → **Registration** (the searchable catalog) → **Progressive Exposure** (the native Skill Directory the agent loads on its own).

## Design motivation

- **Search is catalog-only.** Every supported agent (Claude, Codex, Goose, Cline, opencode) natively auto-discovers `SKILL.md` folders, so installed skills never need to be indexed or returned by a tool — the harness already loads them. So `search_skills` handles what native discovery can't reach: a catalog of potentially thousands of *un*installed skills, searchable without holding them all in context. Catalogs are indexed on name, description, and tags, which keeps those summaries compact and avoids diluting the embedding with full SKILL.md bodies.
- **Keyword fallback, no embedder required.** When no embedding model is configured, `search_skills` falls back to a keyword match over the local clones, so it still works (just less fuzzy) — no model or API key needed.
- **One router, not scattered policy.** Backend selection (semantic vs. keyword), the catalog/installed split, and source bookkeeping all live in the SkillRouter rather than being re-implemented at each MCP and CLI call site, so the behavior can't drift between them.
- **Federated and provenance-preserving.** Each source is an independent per-source collection, so re-syncing one never disturbs another; installing a catalog skill preserves its upstream `LICENSE`/`NOTICE` and stamps a `PROVENANCE.txt` into the installed directory.

## Bundled and authored skills

DSAgt ships a `skill-creator` skill (for scaffolding new `SKILL.md` skills). Bundled and installed skills are **not** indexed for search — the agent auto-discovers `SKILL.md` folders natively — so `search_skills` is reserved for the catalog tier. Domain skills, including the MODCON `datacard-generator`, are sourced from external catalogs rather than bundled, so they stay current upstream.

To add one by hand, place a new directory under `<project>/skills/` with a `SKILL.md` describing the workflow; the next `dsagt start` mirrors it into the agent's native skill directory, after which the agent auto-discovers and invokes it — no indexing step.

## Try it

```bash
dsagt init            # follow the prompts: name it `demo`, then pick your agent
dsagt start demo      # launch the agent in the project
```

Then, in the agent:

1. > List the skill sources and their sync status.
2. > Sync the `genesis` catalog and search it for a data-card skill.
3. > Install the one that fits, then use it on this project.

## In practice

See the [Use Cases](use-cases/index.md), which draw on installed skills — such as the MODCON data-card generator — while working a real dataset end to end.

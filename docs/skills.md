# Skills

DSAgt enables an agent to discover and install skills from an external corpus during a session.

Installed skills, the two base skills included, are located in `<project>/skills/`. Each is a directory containing a `SKILL.md` file and optional reference documents.

## Corpus and installed skills

![DSAgt skill routing](assets/skills-routing.png)

Skills fall into two sets — the searchable **corpus** and the project's **installed skills** — and one class, `SkillRouter`, routes every skill operation between them:

- **Corpus** — skills that exist in external repositories but are *not yet installed*. DSAgt federates many sources (the known names below, or any git URL); each is cloned and indexed into its own collection. The agent browses the corpus with `search_skills` and manages sources with `add_skill_source` / `list_skill_sources`.
- **Installed skills** — skills in the project's `<project>/skills/` directory: the base skills every init installs (`skill-creator`, `aidrin`), skills installed from the corpus (`install_skill`), and skills authored in place (with `skill-creator`). These are mirrored into each agent's *native* skills directory (`.claude/`, `.agents/`, `.cline/`) at install time (and re-mirrored at `dsagt init`/`start`), where the agent auto-discovers and auto-invokes them.

## Design motivation

- **Corpus search.** Every supported agent (Claude, Codex, Goose, Cline, opencode) natively auto-discovers `SKILL.md` folders, so installed skills never need to be indexed or returned by a tool. So `search_skills` discovers new skills for an agent to access: a corpus of potentially thousands of *un*installed skills, searchable without holding them all in context. The corpus is indexed on name, description, and tags, which keeps those summaries compact and avoids diluting the embedding with full SKILL.md bodies.
- **Keyword fallback** When no embedding model is configured, `search_skills` falls back to a keyword match over the local clones.

- **Federated and provenance-preserving.** Each source is an independent per-source collection, so re-syncing one never disturbs another; installing a skill from the corpus preserves its upstream `LICENSE`/`NOTICE` and stamps a `PROVENANCE.txt` into the installed directory.

## Sources

`dsagt init` enables the `genesis` source by default. The others are enabled at init (the interactive checkbox, or `--include <name>`) or during a session with the `add_skill_source` tool, which also accepts any git URL. A source is a git repository holding `SKILL.md` directories; discovery is recursive under the configured subdirectory, so a skill added upstream appears after the source is re-synced: `add_skill_source` with `force: true` re-clones a cached source.

| Name | Repository | Contents |
|---|---|---|
| `genesis` (default) | `github.com/AI-ModCon/genesis-skills`, `skills/` | HPC job and site skills (`slurm`, `pbs`, `perlmutter`, `aurora`, `frontier`); BaseData skills (`datacard-generator`, `croissant-validator`, `hdmf-schema-builder`, `well-convert`, `skill-creator`); BaseEval and BaseSAFE skills; plasma simulation (`gkeyll`, `gs2`); AmSC skills (Globus Compute, IRI API, data movement); `academy`, `literature-search`. |
| `k-dense-ai` | `github.com/K-Dense-AI/scientific-agent-skills` | 140+ chemistry, biology, medicine, and materials skills. |
| `anthropic` | `github.com/anthropics/skills`, `skills/` | Anthropic document-editing and design skills. |
| `antigravity` | `github.com/sickn33/antigravity-awesome-skills` | 1,500+ cross-platform agent skills. |
| `composio` | `github.com/ComposioHQ/awesome-claude-skills` | Workflow skills for SaaS applications. |

The `genesis` source is the ModCon aggregation point: skills contributed by ModCon and AmSC teams land there and become searchable on the next sync. The BaseData team's own skills, including `skill-creator`, are maintained under `skills/basedata-skills/` in that repository.

## Base and authored skills

DSAgt holds no skills of its own. Every `dsagt init` installs two base skills into `<project>/skills/` from the repositories that maintain them, re-cloning each so the copy matches upstream:

| Skill | Source |
|---|---|
| `skill-creator` | `genesis`, `skills/basedata-skills/skill-creator/` |
| `aidrin` | `github.com/idtlab/AIDRIN`, `.claude/skills/aidrin/` (branch `develop`). Running it requires the AIDRIN package; the [readiness gate](readiness.md) installs it, and the skill's `reference/installation.md` covers a manual setup. |

Installed skills are **not** indexed for search — the agent auto-discovers `SKILL.md` folders natively — so `search_skills` is reserved for the corpus. Domain skills, including the BaseData `datacard-generator`, are installed from the corpus rather than built in, so they stay current upstream.

To manually add a skill, place a new directory under `<project>/skills/` with a `SKILL.md` describing the workflow; the next `dsagt start` mirrors it into the agent's native skill directory, after which the agent auto-discovers and invokes it — no indexing step.

## Try it

```bash
dsagt init            # follow the prompts: name it `demo`, then pick your agent
dsagt start demo      # launch the agent in the project
```

Then, in the agent:

1. > List the skill sources and their sync status.
2. > Sync the `genesis` source and search it for a data-card skill.
3. > Install the one that fits, then use it on this project.

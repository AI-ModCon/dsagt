# VS Code extension hand-test

Tests claude and roo run through their VS Code extensions instead of the CLI. Both extensions auto-discover dsagt's per-project config files (`.mcp.json` + `CLAUDE.md` for claude; `.roo/mcp.json` + `.roomodes` for roo) when you open VS Code with the project as workspace root.

Replace `<AGENT>` below with `claude` or `roo`.

## Setup (once per agent)

```bash
cd ~/dsagt
source .venv/bin/activate
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# Provider creds in shell — same as CLI BYOA flow.  The VS Code
# extension inherits them when launched from a terminal that has them
# set (zshrc / .zprofile / etc.).
# claude: subscription auth via ~/.claude or `claude /login`
# roo: configure via the extension's UI after opening the project

dsagt rm smoke-<AGENT> -y >/dev/null 2>&1
dsagt init smoke-<AGENT> --agent <AGENT>
```

## Run

```bash
# Terminal 1
dsagt mlflow smoke-<AGENT>

# Terminal 2 — open VS Code at the project root
code ~/dsagt-projects/smoke-<AGENT>
```

Inside VS Code:

- **claude:** open the Claude Code extension (sidebar icon or Cmd-Shift-P → "Claude"). The extension auto-loads `CLAUDE.md` from the workspace root and `.mcp.json` for the dsagt MCP servers.
- **roo:** open the Roo Code extension. **Pick "DSAgt Pipeline Builder" from the mode dropdown** (top of the chat panel) — without this, roo runs in default "code" mode and `.roomodes` `customInstructions` are dropped from the system prompt.

## Scripted prompts (paste into the extension chat one at a time)

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > I have a CSV utility called `csvtool`. Its reference is at `$SMOKE_DIR/knowledge/api_reference.md` — register the `filter` subcommand. Use an underscore in the name.
3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`.
4. > Look at `$SMOKE_DIR/data/samples.csv` and summarize — columns, row count, quality issues.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > What do you remember about the samples dataset?

Substitute `$SMOKE_DIR` with the absolute path the prompts above used (the extension chat doesn't expand env vars).

Close the extension chat / VS Code session when done.

## Verify

```bash
dsagt info smoke-<AGENT>
ls ~/dsagt-projects/smoke-<AGENT>/codes/
ls ~/dsagt-projects/smoke-<AGENT>/trace_archive/
test -s ~/dsagt-projects/smoke-<AGENT>/explicit_memories.yaml && echo OK
```

In MLflow UI:

- **claude extension:** same trace shape as claude CLI — full `api_response_body` + `tool_use` payloads in OTel log events (gated by `CLAUDE_CODE_*` / `OTEL_LOG_*` env vars; the extension reads them from the shell or VS Code's `terminal.integrated.env.osx` setting).
- **roo extension:** no agent LLM-call OTel (roo emits none natively). MCP-server spans (`kb.*`, `registry.*`) and `dsagt-run` `tool.execute` spans only.

## Notes

- **Why only claude + roo:** these are the only two agents we ship support for whose VS Code extensions auto-discover config from the workspace root. Cline's extension reads MCP from a single global path (punted). Codex / opencode / goose have no real VS Code extension.
- **claude keychain:** the extension uses the same `~/.claude/` auth as the CLI. If your environment requires a non-default `ANTHROPIC_BASE_URL` (lab gateway), set it in VS Code's `terminal.integrated.env.osx` setting or in the user shell that VS Code inherits from.
- **roo provider config:** unlike the CLI, the roo extension lets you set `baseURL` per provider via the extension's settings UI. Configure that once before running the prompts.

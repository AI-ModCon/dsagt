# Claude Code

[Claude Code](https://github.com/anthropics/claude-code) runs as a CLI (`claude`) and VS Code extension.

## Setup

```bash
dsagt init my-project --agent claude-code
dsagt start my-project
```

`dsagt start` generates `.mcp.json` and `CLAUDE.md` in the working directory, starts the proxy and MLflow, then launches `claude` with `ANTHROPIC_BASE_URL` pointing at the proxy.

## What gets generated

| File | Purpose |
|---|---|
| `.mcp.json` | MCP server config (registry + knowledge servers with project-specific paths) |
| `CLAUDE.md` | Agent instructions (DSAGT pipeline builder workflow) |
| `.dsagt_env` | Environment variables (for manual use outside `dsagt start`) |

## How the proxy intercept works

Claude Code reads `ANTHROPIC_BASE_URL` from the environment. `dsagt start` sets this to `http://localhost:<proxy_port>`, so all LLM calls route through the local LiteLLM proxy for trace capture and tool execution recording. The proxy forwards requests to the configured LLM provider transparently.

## Manual setup (without dsagt start)

If you need to start things independently:

```bash
source .dsagt_env
dsagt-proxy --port 4000 --records-dir runtime/my-project/trace_archive
claude
```

## Notes

- Claude Code discovers `.mcp.json` at startup automatically.
- Check MCP server status with the `/mcp` command inside Claude Code.
- The `--rerank` flag on the knowledge server triggers a model download on first use. Add it to the args in `.mcp.json` if needed.

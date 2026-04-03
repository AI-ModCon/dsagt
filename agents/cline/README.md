# Cline

[Cline](https://github.com/cline/cline) runs as a VS Code extension (formerly Claude Dev).

## Setup

```bash
dsagt init my-project --agent cline
dsagt start my-project
```

Since Cline is a VS Code extension, `dsagt start` starts the background services (proxy + MLflow), generates the config files, and tells you to open VS Code. Open VS Code, verify MCP servers are connected in the Cline panel, and you're ready.

## What gets generated

| File | Purpose |
|---|---|
| `cline_mcp.json` | MCP server config to merge into Cline's global settings |
| `.clinerules/dsagt_instructions.md` | Agent instructions (DSAGT pipeline builder workflow) |
| `.dsagt_env` | Environment variables (source before launching VS Code) |

## How the proxy intercept works

Cline manages its LLM provider through VS Code extension settings. To route through the proxy, source `.dsagt_env` before launching VS Code so `ANTHROPIC_BASE_URL` is set, or configure the API base URL in Cline's settings directly.

## MCP server config

Cline stores MCP config in a global settings file, not a project-local file. `dsagt start` generates `cline_mcp.json` in the working directory — merge its contents into Cline's settings via **MCP Servers** > **Edit MCP Settings** in the Cline panel. The settings file is at:

- **macOS:** `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Linux:** `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Windows:** `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

## Notes

- Verify servers are connected in the Cline panel's MCP Servers section.
- To auto-approve specific tools, add tool names to `alwaysAllow` in the MCP config.
- Cline may not pass version-manager env vars to MCP subprocesses. Launch VS Code from a terminal where these are set.

# Roo Code

[Roo Code](https://github.com/RooCodeInc/Roo-Code) runs as a VS Code extension.

## Setup

```bash
dsagt init my-project --agent roo
dsagt start my-project
```

Since Roo is a VS Code extension, `dsagt start` starts the background services (proxy + MLflow), generates the config files, and tells you to open VS Code. Open VS Code in the working directory, switch to the **DSAGT Pipeline Builder** mode (Cmd+.), and you're ready.

## What gets generated

| File | Purpose |
|---|---|
| `.roo/mcp.json` | MCP server config (registry + knowledge servers with project-specific paths) |
| `.roomodes` | Custom DSAGT Pipeline Builder mode with agent instructions |
| `.dsagt_env` | Environment variables (source before launching VS Code) |

## How the proxy intercept works

Roo Code manages its LLM provider through the VS Code extension settings. To route through the proxy, set the API base URL in Roo's settings to `http://localhost:<proxy_port>`. Alternatively, source `.dsagt_env` before launching VS Code so `ANTHROPIC_BASE_URL` is set in the environment.

## Notes

- Refresh MCP servers in the Roo settings panel after generating configs.
- The `.roomodes` file goes in the project root (not inside `.roo/`).
- Roo may not pass version-manager env vars (`PYENV_ROOT`, `NVM_DIR`) to MCP subprocesses. Launch VS Code from a terminal where these are set.

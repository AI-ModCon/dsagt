# Using DSAGT with Roo Code

[Roo Code](https://github.com/RooCodeInc/Roo-Code) is an AI coding agent that runs as a VS Code extension. It connects to LLMs via configurable API providers and extends its capabilities through MCP servers. Roo supports custom modes that tailor agent behavior for specific workflows.

## Prerequisites

- DSAGT installed (see the [main README](../../README.md))
- VS Code 1.84.0 or later
- [Roo Code](https://marketplace.visualstudio.com/items?itemName=RooVeterinaryInc.roo-cline) VS Code extension installed

To install the extension: open VS Code, go to Extensions (`Cmd+Shift+X`), search for "Roo Code", and install the one by RooVeterinaryInc.

## Environment Variables

Roo Code manages its LLM provider configuration through the extension settings UI in VS Code (model, API key, endpoint). No shell environment variables are needed for the LLM connection.

If using the DSAGT knowledge server with the API embedding backend, set the embedding API key in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export LLM_API_KEY="your-api-key"
```

VS Code must be launched from a terminal where this variable is set, or you can pass it via the `env` block in `.roo/mcp.json` (see [Configuration](#configuration)).

## Quick Start

1. Open the DSAGT project root in VS Code.

2. Open the Roo Code panel: press `Cmd+Shift+P` and run "Roo Code: Open in New Tab".

3. Copy the MCP server config into the project:

```bash
mkdir -p .roo
cp agents/roo/roo_mcp.json .roo/mcp.json
```

4. Copy the custom mode definition to the project root:

```bash
cp agents/roo/roomodes .roomodes
```

5. In the Roo Code tab, open the **MCP settings panel** (gear icon at the top) and click **Refresh MCP Servers** to pick up the new config.

6. Switch to the **DSAGT Pipeline Builder** mode. You can do this three ways:
   - Click the mode dropdown to the left of the chat input
   - Press `Cmd+.` to cycle through modes
   - Type `/dsagt` at the start of a message

## Configuration

### MCP Servers (`roo_mcp.json`)

Copy to `.roo/mcp.json` in the project root. Roo reads this file to discover and launch MCP servers over stdio.

```json
{
  "mcpServers": {
    "dsagt-registry": {
      "command": "uv",
      "args": ["run", "dsagt-registry-server"],
      "disabled": false
    },
    "dsagt-knowledge": {
      "command": "uv",
      "args": ["run", "dsagt-knowledge-server"],
      "disabled": false
    }
  }
}
```

To customize paths, add flags to `args`. For example:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-registry-server", "--runtime-dir", "/absolute/path/to/session"],
  "disabled": false
}
```

To pass environment variables to the server processes, use the `env` block with literal values:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-knowledge-server"],
  "env": {
    "LLM_API_KEY": "your-api-key"
  },
  "disabled": false
}
```

Note: the `${env:VAR}` variable expansion syntax works in `args` but is unreliable inside the `env` block itself. Use literal values there.

To enable cross-encoder reranking, add `"--rerank"` to the knowledge server args. This triggers a model download on first use.

### MCP Settings Panel

Roo Code has a dedicated MCP settings panel in the sidebar (gear icon at the top). From there you can:

- Toggle individual servers on/off
- Restart individual servers
- Refresh all servers after config changes
- Set network timeouts (default 1 minute, increase if servers are slow to initialize)
- Configure per-tool auto-approval via "Always allow" checkboxes

### Custom Mode (`roomodes`)

Copy to `.roomodes` in the project root (not inside `.roo/`). This defines the **DSAGT Pipeline Builder** mode, which conditions Roo with the DSAGT agent instructions: tool-mediated data access, the tool preference hierarchy, paired check/operation tools, and the iterative pipeline-building cycle.

Without this mode, Roo won't have the context it needs to use the three servers effectively.

You can also place additional per-mode rule files in `.roo/rules-dsagt/` for supplementary instructions specific to the DSAGT mode.

## Smoke Test

Follow the [smoke test instructions](../../README.md#smoke-test) in the main README. For step 2, make sure the `.roo/mcp.json` and `.roomodes` files are in place, refresh MCP servers in the settings panel, and switch to the DSAGT Pipeline Builder mode.

If you don't have an embedding API key, set `"disabled": true` on the `dsagt-knowledge` entry in `.roo/mcp.json` and skip the knowledge base steps.

## Troubleshooting

### Servers not connecting

Open the MCP settings panel and check whether the servers show a connected status. If not, click the restart button next to the failing server to see error details.

Verify uv can find the server commands from within VS Code's integrated terminal:

```bash
uv run which dsagt-registry-server
```

### Python/node version managers (pyenv, nvm, fnm)

Roo Code only passes `PATH` to MCP server subprocesses, not the additional environment variables that version managers require (e.g., `PYENV_ROOT`, `NVM_DIR`). If the servers fail to launch, either:

- Specify the full absolute path to the Python or server executable in `command`
- Add the required environment variables to the `env` block in your config

### MCP servers not showing up after config changes

After editing `.roo/mcp.json`, open the MCP settings panel and click **Refresh MCP Servers**. A full VS Code window reload is not required.

### Server timeout

The default MCP network timeout is 1 minute. If servers take longer to initialize (e.g., downloading the reranker model on first run), increase the timeout in the MCP settings panel dropdown.

### Working directory

Roo launches server processes from the VS Code workspace root. Make sure the DSAGT project root is your workspace so relative paths (`./runtime`, `./kb_index`) resolve correctly. You can also set `"cwd"` explicitly in the server config:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-registry-server"],
  "cwd": "/absolute/path/to/dsagt",
  "disabled": false
}
```

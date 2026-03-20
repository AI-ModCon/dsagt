# Using DSAGT with Cline

[Cline](https://github.com/cline/cline) is an AI coding agent that runs as a VS Code extension (formerly Claude Dev). It connects to LLMs via configurable API providers and extends its capabilities through MCP servers. Cline supports project-level rule files (`.clinerules/`) that tailor agent behavior for specific workflows.

## Prerequisites

- DSAGT installed (see the [main README](../../README.md))
- VS Code 1.84.0 or later
- [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) VS Code extension installed

To install the extension: open VS Code, go to Extensions (`Cmd+Shift+X`), search for "Cline", and install the one by Saoud Rizwan.

## Environment Variables

Cline manages its LLM provider configuration through the extension settings UI in VS Code (model, API key, endpoint). No shell environment variables are needed for the LLM connection.

If using the DSAGT knowledge server with the API embedding backend, set the embedding API key in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export LLM_API_KEY="your-api-key"
```

VS Code must be launched from a terminal where this variable is set, or you can pass it via the `env` block in the MCP settings (see [Configuration](#configuration)).

## Quick Start

1. Open the DSAGT project root in VS Code.

2. Open the Cline panel: click the Cline icon in the sidebar or press `Cmd+Shift+P` and run "Cline: Open In New Tab".

3. Add the MCP server config. Click **MCP Servers** at the top of the Cline panel, then click **Edit MCP Settings**. Merge the contents of `agents/cline/cline_mcp.json` into the settings file that opens:

```json
{
  "mcpServers": {
    "dsagt-registry": {
      "command": "uv",
      "args": ["run", "dsagt-registry-server"],
      "disabled": false,
      "alwaysAllow": []
    },
    "dsagt-knowledge": {
      "command": "uv",
      "args": ["run", "dsagt-knowledge-server"],
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

4. Copy the DSAGT agent instructions into the project:

```bash
mkdir -p .clinerules
cp agents/cline/dsagt_instructions.md .clinerules/dsagt_instructions.md
```

5. Verify the servers are connected: click **MCP Servers** in the Cline panel. All three servers should show a green connected status.

## Configuration

### MCP Servers (`cline_mcp.json`)

Cline stores MCP server configuration in a global settings file. Open it by clicking **MCP Servers** > **Edit MCP Settings** in the Cline panel. The file is located at:

- **macOS:** `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Linux:** `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Windows:** `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

If using Cursor instead of VS Code, replace `Code` with `Cursor` in the path.

To customize paths, add flags to `args`. For example:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-registry-server", "--runtime-dir", "/absolute/path/to/session"],
  "disabled": false,
  "alwaysAllow": []
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
  "disabled": false,
  "alwaysAllow": []
}
```

To enable cross-encoder reranking, add `"--rerank"` to the knowledge server args. This triggers a model download on first use.

To auto-approve specific tools so Cline doesn't prompt for each call, add tool names to `alwaysAllow`:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-registry-server"],
  "disabled": false,
  "alwaysAllow": ["get_registry", "search_registry"]
}
```

### Agent Instructions (`.clinerules/`)

Copy the DSAGT instructions into the project's `.clinerules/` directory:

```bash
mkdir -p .clinerules
cp agents/cline/dsagt_instructions.md .clinerules/dsagt_instructions.md
```

Cline reads all `.md` files from `.clinerules/` and injects them into its system prompt. Without this file, the agent won't have the context it needs to use the three servers effectively.

You can verify the rules are loaded by checking the **Rules** section in the Cline panel sidebar. Rules can be toggled individually on/off.

## Smoke Test

Follow the [smoke test instructions](../../README.md#smoke-test) in the main README. For step 2, make sure the MCP servers are configured and the `.clinerules/dsagt_instructions.md` file is in place.

If you don't have an embedding API key, set `"disabled": true` on the `dsagt-knowledge` entry in the MCP settings and skip the knowledge base steps.

## Troubleshooting

### Servers not connecting

Click **MCP Servers** in the Cline panel to check server status. If a server shows as disconnected, click the restart button next to it to see error details.

Verify uv can find the server commands from within VS Code's integrated terminal:

```bash
uv run which dsagt-registry-server
```

### Python/node version managers (pyenv, nvm, fnm)

Cline may not pass the additional environment variables that version managers require (e.g., `PYENV_ROOT`, `NVM_DIR`). If the servers fail to launch, either:

- Specify the full absolute path to the Python or server executable in `command`
- Add the required environment variables to the `env` block in your config

### MCP servers not showing after config changes

After editing the MCP settings file, click **MCP Servers** in the Cline panel and restart the affected servers. A full VS Code window reload is not required.

### Server timeout

If servers take a long time to initialize (e.g., downloading the reranker model on first run), Cline may report a connection failure. Restart the server from the MCP panel once initialization is complete.

### Working directory

Cline launches server processes from the VS Code workspace root. Make sure the DSAGT project root is your workspace so relative paths (`./runtime`, `./kb_index`) resolve correctly.

# Using DSAGT with Claude Code

[Claude Code](https://github.com/anthropics/claude-code) is Anthropic's official agentic coding tool. It runs as a CLI (`claude`) and as a VS Code extension, connecting to Claude models and extending its capabilities through MCP servers. Claude Code reads project-level `CLAUDE.md` for instructions and `.mcp.json` for MCP server configuration.

## Prerequisites

- DSAGT installed (see the [main README](../../README.md))
- [Claude Code](https://github.com/anthropics/claude-code) installed (`npm install -g @anthropic-ai/claude-code`)

## Environment Variables

Claude Code manages its own API key through `claude login` or the `ANTHROPIC_API_KEY` environment variable. No additional environment variables are needed for the LLM connection.

If using the DSAGT knowledge server with the API embedding backend, set the embedding API key in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export LLM_API_KEY="your-api-key"
```

## Quick Start

1. Open the DSAGT project root in your terminal (CLI) or in VS Code (extension).

2. Copy the MCP server config into the project:

```bash
cp agents/claude-code/claude_code_mcp.json .mcp.json
```

3. Append the DSAGT agent instructions to the project CLAUDE.md:

```bash
cat agents/claude-code/dsagt_instructions.md >> CLAUDE.md
```

4. Start Claude Code:

**CLI:**
```bash
claude
```

**VS Code:** Open the command palette (`Cmd+Shift+P`) and run "Claude Code: Open".

Claude Code discovers `.mcp.json` at startup and connects to the three DSAGT servers automatically.

## Configuration

### MCP Servers (`claude_code_mcp.json`)

Copy to `.mcp.json` in the project root. Claude Code reads this file to discover and launch MCP servers over stdio.

```json
{
  "mcpServers": {
    "dsagt-pipeline": {
      "command": "uv",
      "args": ["run", "dsagt-pipeline-server"]
    },
    "dsagt-registry": {
      "command": "uv",
      "args": ["run", "dsagt-registry-server"]
    },
    "dsagt-knowledge": {
      "command": "uv",
      "args": ["run", "dsagt-knowledge-server"]
    }
  }
}
```

To customize paths, add flags to `args`:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-pipeline-server", "--registry", "/absolute/path/to/registry.yaml"]
}
```

To pass environment variables to the server processes, use the `env` block:

```json
{
  "command": "uv",
  "args": ["run", "dsagt-knowledge-server"],
  "env": {
    "LLM_API_KEY": "your-api-key"
  }
}
```

To enable cross-encoder reranking, add `"--rerank"` to the knowledge server args. This triggers a model download on first use.

### Agent Instructions (`dsagt_instructions.md`)

The `dsagt_instructions.md` file contains the DSAGT pipeline builder instructions. Append it to the project-root `CLAUDE.md`:

```bash
cat agents/claude-code/dsagt_instructions.md >> CLAUDE.md
```

Claude Code reads `CLAUDE.md` from the working directory at startup. Without these instructions, the agent won't have the context it needs to use the three servers effectively.

If the project root doesn't have a `CLAUDE.md` yet, you can copy instead of append:

```bash
cp agents/claude-code/dsagt_instructions.md CLAUDE.md
```

## Smoke Test

Follow the [smoke test instructions](../../README.md#smoke-test) in the main README. For step 2, make sure `.mcp.json` is in place and the DSAGT instructions are in `CLAUDE.md`.

If you don't have an embedding API key, remove the `dsagt-knowledge` entry from `.mcp.json` and skip the knowledge base steps.

## Troubleshooting

### Servers not connecting

Check MCP server status with the `/mcp` command inside Claude Code. This shows which servers are connected and lets you restart them.

Verify uv can find the server commands:

```bash
uv run which dsagt-pipeline-server
```

### Working directory

Claude Code launches server processes from the directory where you start the CLI or the VS Code workspace root. Make sure the DSAGT project root is your working directory so relative paths (`./runtime`, `./kb_index`) resolve correctly.

### Server timeout

If servers take a long time to initialize (e.g., downloading the reranker model on first run), Claude Code will retry the connection automatically. You can also restart individual servers with `/mcp`.

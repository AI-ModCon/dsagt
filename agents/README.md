# Agent Platforms

Each agent platform needs the same three things: MCP server config, agent instructions, and (optionally) platform-specific settings. The tables below map source files in `agents/` to their destinations.

Each platform has a self-contained setup guide at `agents/<platform>/README.md`.

## Available Interfaces

All four platforms offer a VS Code extension. Goose, Claude Code, and Cline also provide a standalone CLI.

| | CLI | VS Code Extension |
|---|---|---|
| **Goose** | `goose session` | [Goose](https://marketplace.visualstudio.com/items?itemName=michaelneale.goose-vscode) |
| **Claude Code** | `claude` | [Claude Code](https://marketplace.visualstudio.com/items?itemName=anthropics.claude-code) |
| **Roo Code** | — | [Roo Code](https://marketplace.visualstudio.com/items?itemName=RooVeterinaryInc.roo-cline) |
| **Cline** | `cline` | [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) |

## MCP Server Configuration

| | Source (in repo) | CLI Destination | VS Code Destination | Format Notes |
|---|---|---|---|---|
| **Goose** | `goose/goose.yaml` | `goose.yaml` (project root) or `~/.config/goose/config.yaml` | Same (extension uses Goose config) | YAML `extensions:` block with `type: stdio` |
| **Claude Code** | `claude-code/claude_code_mcp.json` | `.mcp.json` (project root) | `.mcp.json` (project root) | JSON `mcpServers` object |
| **Roo Code** | `roo/roo_mcp.json` | — | `.roo/mcp.json` (project root) | JSON `mcpServers` object with `disabled` field |
| **Cline** | `cline/cline_mcp.json` | `~/.cline/` config directory | Global settings file (see below) | JSON `mcpServers` object with `alwaysAllow` field |

Cline VS Code global MCP settings file location:
- **macOS:** `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Linux:** `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Windows:** `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

## Agent Instructions (DSAGT Workflow Prompt)

| | Source (in repo) | Destination | Delivery Mechanism |
|---|---|---|---|
| **Goose** | `goose/.goosehints` | `.goosehints` (project root) | Auto-loaded from project root (CLI and VS Code) |
| **Claude Code** | `claude-code/dsagt_instructions.md` | Append to `CLAUDE.md` (project root) | Injected into system prompt (CLI and VS Code) |
| **Roo Code** | `roo/roomodes` | `.roomodes` (project root) | Custom mode with `roleDefinition` + `customInstructions` |
| **Cline** | `cline/dsagt_instructions.md` | `.clinerules/dsagt_instructions.md` | All `.md` files in `.clinerules/` injected into system prompt |

## Platform-Specific Config

| | File | Purpose |
|---|---|---|
| **Goose** | `goose.yaml` | LLM provider, model, mode, extension timeouts |
| **Roo Code** | `.roomodes` | Custom "DSAGT Pipeline Builder" mode with tool group permissions |
| **Cline** | `alwaysAllow` array in MCP config | Auto-approve specific tool calls without prompting |
| **Claude Code** | `CLAUDE.md` | Project architecture context + agent instructions in one file |

## Environment Variables

| | How to pass `LLM_API_KEY` to servers |
|---|---|
| **Goose** | Set in shell profile (`~/.bashrc`); inherited by CLI and VS Code extension |
| **Claude Code** | Set in shell profile; inherited by `claude` CLI and VS Code (launch from that terminal) |
| **Roo Code** | Set in shell profile; launch VS Code from that terminal, or add `"env"` block in `.roo/mcp.json` |
| **Cline** | Set in shell profile, or add `"env": {"LLM_API_KEY": "..."}` block in MCP config |

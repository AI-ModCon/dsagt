# Using DSAGT with Goose

[Goose](https://github.com/block/goose) is an open-source, on-machine AI agent developed by Block. It runs locally, connects to any LLM that supports tool calling via the OpenAI-compatible API, and extends its capabilities through MCP servers. Goose is available as both a CLI and a desktop app, and reads project-level configuration and hints files from the working directory to condition its behavior per-project.

## Prerequisites

- DSAGT installed (see the [main README](../../README.md))
- [Goose](https://github.com/block/goose) installed

## Environment Variables

Goose requires these environment variables to be set. Add them to your `~/.bashrc` or `~/.zshrc` to persist across sessions.

```bash
# LLM provider (Goose uses the OpenAI-compatible interface)
export OPENAI_API_KEY="your-api-key"
export OPENAI_HOST="https://your-llm-endpoint.example.com"  # omit for OpenAI direct
```

`OPENAI_HOST` points Goose at an OpenAI-compatible endpoint. If you're using OpenAI directly, you can omit it. See the [Goose provider docs](https://block.github.io/goose/docs/getting-started/providers/) for the full list of supported providers and their environment variables.

Goose also reads `GOOSE_PROVIDER` and `GOOSE_MODEL` from the `goose.yaml` config file (see [Configuration](#configuration)). If using the OpenAI-compatible interface, set the provider to `openai` and the model to your desired model name.

## Quick Start

From the DSAGT project root, copy the agent instructions and start a session:

```bash
# Copy agent guidance to project root (required)
cp agents/goose/.goosehints .goosehints

# Start a session with both servers
goose session \
  --with-extension 'uv run dsagt-registry-server' \
  --with-extension 'uv run dsagt-knowledge-server'
```

No server flags required — each server uses sensible defaults. The registry server seeds `./runtime/skills/` from the bundled default skills on first run.

To resume a previous session's tool registry:

```bash
goose session \
  --with-extension 'uv run dsagt-registry-server --runtime-dir my_session' \
  --with-extension 'uv run dsagt-knowledge-server'
```

## Configuration

For persistent setup, copy one of the config files included in this directory:

- **`goose.yaml`** — Project-local config. Copy to your project root and run `goose session` or `goose session --config goose.yaml`.
- **`config.yaml`** — Global config. Copy to `~/.config/goose/config.yaml` so the servers are available in every Goose session.

Both files contain the same content:

```yaml
GOOSE_PROVIDER: openai
GOOSE_MODEL: claude-sonnet-4-20250514

extensions:
  registry:
    enabled: true
    type: stdio
    cmd: uv run dsagt-registry-server
    timeout: 300

  knowledge:
    enabled: true
    type: stdio
    cmd: uv run dsagt-knowledge-server
    timeout: 300
```

To customize paths, add `args:` with flags like `--runtime-dir` or `--base-index-dir`. For example:

```yaml
extensions:
  registry:
    enabled: true
    type: stdio
    cmd: uv run dsagt-registry-server
    args:
      - --runtime-dir
      - /absolute/path/to/session
    timeout: 300
```

## Agent Instructions (`.goosehints`)

The `.goosehints` file is the primary mechanism for conditioning Goose's behavior when working with DSAGT. It contains curated instructions that guide the agent through tool registration workflows, pipeline execution patterns, and knowledge base usage. Copy it from this directory to the DSAGT project root:

```bash
cp agents/goose/.goosehints .goosehints
```

Goose automatically reads `.goosehints` from the working directory at session start. Without this file, the agent won't have the context it needs to use the three servers effectively.

## Smoke Test

Follow the [smoke test instructions](../../README.md#smoke-test) in the main README. For step 2, start the session with:

```bash
goose session \
  --with-extension 'uv run dsagt-registry-server' \
  --with-extension 'uv run dsagt-knowledge-server'
```

If you don't have an embedding API key, drop the knowledge server line and skip the knowledge base steps. To enable cross-encoder reranking (triggers a model download on first use), add `--rerank`.

## Troubleshooting

### Servers not connecting

Verify uv can find the server commands:

```bash
uv run which dsagt-registry-server
```

### Extension timeout

The default timeout is 300 seconds. If servers take longer to initialize (e.g., downloading the reranker model on first run), increase the `timeout:` value in your config.

### Working directory

Goose launches server processes from the directory where you run `goose session`. Make sure you run it from the DSAGT project root so relative paths (`./runtime`, `./kb_index`) resolve correctly. Alternatively, use absolute paths in the config `args:`.

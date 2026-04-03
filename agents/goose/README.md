# Goose

[Goose](https://github.com/block/goose) runs as a CLI (`goose`) and desktop app.

## Setup

```bash
dsagt init my-project --agent goose
dsagt start my-project
```

`dsagt start` generates `goose.yaml` and `.goosehints` in the working directory, starts the proxy and MLflow, then launches `goose session` with `OPENAI_HOST` pointing at the proxy.

## What gets generated

| File | Purpose |
|---|---|
| `goose.yaml` | Goose config with MCP extensions (registry + knowledge servers), provider, and model |
| `.goosehints` | Agent instructions (DSAGT pipeline builder workflow) |
| `.dsagt_env` | Environment variables (for manual use outside `dsagt start`) |

## How the proxy intercept works

Goose uses the OpenAI-compatible API. `dsagt start` sets `OPENAI_HOST=http://localhost:<proxy_port>`, routing all LLM calls through the LiteLLM proxy. LiteLLM handles translation to the configured upstream LLM provider.

## Manual setup (without dsagt start)

```bash
source .dsagt_env
dsagt-proxy --port 4000 --records-dir runtime/my-project/trace_archive
goose session
```

## Notes

- Goose reads `goose.yaml` from the working directory or `~/.config/goose/config.yaml`.
- Goose requires `OPENAI_API_KEY` in the environment. When using the proxy, this can be any non-empty string since the proxy handles the real API key.
- Extension timeout defaults to 300 seconds. Increase in `goose.yaml` if servers are slow to initialize.

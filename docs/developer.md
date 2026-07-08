# Developer Guide

Material for contributors and developers.

## Tests

```bash
uv run python -m pytest -m "not integration"     # unit tests, no creds required
uv run python -m pytest -m integration -v        # integration tests (require real credentials / models)
```

For per-flow hand-tests (CLI, VS Code extensions), see the scripts under [`tests/manual_walkthroughs/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/manual_walkthroughs/).

## Troubleshooting

**Agent command not found.** The agent CLI is not installed or is not on PATH. See the [supported agents table](index.md#supported-agents).

**MCP server not connecting.** Verify the server command resolves:

```bash
uv run which dsagt-server
```

If missing, reinstall: `pip install --force-reinstall "git+https://github.com/AI-ModCon/dsagt.git"`.
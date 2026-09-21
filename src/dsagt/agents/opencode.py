"""
OpenCode (sst) agent setup.

Install: ``npm i -g opencode-ai``.
Generates: ``AGENTS.md`` (auto-loaded from cwd, the same convention codex
uses) and ``opencode.json`` (per-project config with MCP servers and
provider interpolation references).

Traces: opencode's turns come from its on-disk session database through the
periodic pass (``traces.OpenCodeReader``), the same way as every agent's;
MCP-server spans (kb.*, registry.*) and dsagt-run tool.execute spans are
emitted live.

Auth: opencode reads credentials via ``{env:VAR}`` interpolation in its
``opencode.json`` provider block, so the file references the user's shell
env vars and no credential value is written to disk.  Config layout per
https://opencode.ai/docs/config/ and the ``mcp.ts`` source.

MCP config: ``./opencode.json``'s top-level ``mcp`` key.  Each entry is
``{"type": "local", "command": [...], "environment": {...}}`` for stdio
servers.  dsagt writes the JSON itself because ``opencode mcp add`` is
interactive only (no flags).

Model list: pass-through for known providers (pulled from models.dev)
and user-controlled for custom providers via ``provider.<id>.models``.
opencode passes model names through unchanged.

Batch mode: ``opencode run --dir <path> --dangerously-skip-permissions
-m <provider/model> <prompt>``.  ``--dir`` is the cwd flag.  Stdin appends
to the prompt when not a TTY.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _write_dsagt_block,
    _load_master_instructions,
    _mcp_env_block,
    _mcp_server_args,
    _run_simple_script,
)


def _render_opencode_config(
    mcp_env: dict,
    present_creds: dict[str, bool],
    opencode_model: str | None = None,
) -> str:
    """Render the ``opencode.json`` body.

    *mcp_env*: env vars written into each MCP server's ``environment``
    block (DSAGT_PROJECT_DIR, MLFLOW_TRACKING_URI, EMBEDDING_*).

    *present_creds*: flags (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``,
    ``ANTHROPIC_API_KEY``, ``ANTHROPIC_BASE_URL``); a provider block is
    emitted only when the user has set its API key.

    *opencode_model*: ``<provider>/<model>`` string from the
    ``OPENCODE_MODEL`` env.  A lab-gateway-aliased name like
    ``claude-haiku-4-5-20251001-v1-project`` is absent from models.dev, so
    opencode rejects it under a standard provider unless it is declared in
    ``provider.<id>.models``.  The model is registered there at init time
    and the top-level ``model`` is set, so an interactive ``opencode``
    session uses it without a ``-m`` flag.
    """
    config: dict = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {},
    }
    entry: dict = {
        "type": "local",
        "command": ["uv"] + _mcp_server_args(),
        "enabled": True,
    }
    if mcp_env:
        entry["environment"] = dict(mcp_env)
    config["mcp"]["dsagt"] = entry

    providers: dict = {}
    if present_creds.get("OPENAI_API_KEY"):
        opts: dict = {"apiKey": "{env:OPENAI_API_KEY}"}
        if present_creds.get("OPENAI_BASE_URL"):
            opts["baseURL"] = "{env:OPENAI_BASE_URL}"
        providers["openai"] = {"options": opts}
    if present_creds.get("ANTHROPIC_API_KEY"):
        opts = {"apiKey": "{env:ANTHROPIC_API_KEY}"}
        if present_creds.get("ANTHROPIC_BASE_URL"):
            opts["baseURL"] = "{env:ANTHROPIC_BASE_URL}"
        providers["anthropic"] = {"options": opts}

    # Register the user's chosen model under its provider's ``models``
    # map so opencode accepts a gateway-aliased name that is absent from
    # models.dev.  Without this, ``-m openai/<custom-name>`` fails with
    # ProviderModelNotFoundError.
    if opencode_model and "/" in opencode_model:
        provider_id, model_id = opencode_model.split("/", 1)
        if provider_id in providers:
            providers[provider_id].setdefault("models", {})[model_id] = {
                "name": model_id,
            }
            config["model"] = opencode_model

    if providers:
        config["provider"] = providers

    return json.dumps(config, indent=2)


class OpenCodeSetup(AgentSetup):
    name = "opencode"
    base_command = ["opencode"]
    static_marker = "AGENTS.md"
    install_hint = "Install with `npm i -g opencode-ai`."
    # The AGENTS.md-convention skills dir codex/goose also use.
    native_skills_dir = ".agents/skills"

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [working_dir / "AGENTS.md", working_dir / "opencode.json"]

    def write_static(self, working_dir: Path, *, auto_assess: bool = True) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions(auto_assess)
        if instructions:
            action = _write_dsagt_block(working_dir / "AGENTS.md", instructions)
            if action:
                actions.append(action)
        return actions

    def write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        """Write ``<pdir>/opencode.json`` with MCP server registrations and
        provider interpolation refs.  Auth keys are never written to disk:
        ``{env:VAR}`` is a reference, and opencode resolves it at run time
        from the user's shell.
        """
        del pdir
        actions: list[str] = []
        mcp_env = _mcp_env_block(config)
        # Decide which provider blocks to emit from the env passed to the
        # agent (which mirrors os.environ).  A block is emitted only for a
        # provider the user has credentials for; an empty ``{env:VAR}``
        # interpolation would leave opencode authenticating with a blank
        # string.
        present = {
            name: bool(env.get(name))
            for name in (
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
            )
        }
        body = _render_opencode_config(
            mcp_env,
            present,
            opencode_model=env.get("OPENCODE_MODEL"),
        )
        config_path = working_dir / "opencode.json"
        config_path.write_text(body + "\n")
        n_providers = sum(
            1 for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if present.get(k)
        )
        actions.append(
            f"Wrote {config_path} ({len(mcp_env)} MCP env vars, "
            f"{n_providers} provider block(s))"
        )
        return actions

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``opencode run`` call with the script as the prompt.

        ``--dir`` is opencode's cwd flag.  ``--dangerously-skip-permissions``
        lets unattended runs auto-approve all tool calls.  ``-m`` sets the
        model from ``OPENCODE_MODEL``, in ``<provider>/<name>`` form, since
        opencode rejects a bare model name without a provider prefix.
        ``max_turns`` is unused; opencode has no turn-cap option.
        """
        del config, max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        model = env.get("OPENCODE_MODEL")
        if not model:
            raise RuntimeError(
                "opencode batch mode requires OPENCODE_MODEL in the shell "
                "env, formatted as '<provider>/<name>' (e.g. "
                "'openai/claude-haiku-4-5-20251001-v1-project'), plus the "
                "matching {ANTHROPIC,OPENAI}_API_KEY / _BASE_URL: the agent "
                "must be configured in the shell before launch."
            )
        cmd = [
            "opencode",
            "run",
            "--dir",
            str(working_dir),
            "--dangerously-skip-permissions",
            "-m",
            model,
            text,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

"""
OpenCode (sst) agent setup.

Install: ``npm i -g opencode-ai``.
Generates: ``AGENTS.md`` (auto-loaded from cwd, same convention codex uses)
and ``opencode.json`` (per-project config with MCP servers + provider
interpolation references).

OTel support: **none** natively (one third-party plugin exists but isn't
wired in by default).  Agent transparency in MLflow is limited to MCP-server
spans (kb.*, registry.*) and dsagt-run tool.execute spans; LLM-call payloads
require Phase 2 proxy mode.

Auth model: opencode reads creds via ``{env:VAR}`` interpolation in its
``opencode.json`` provider block, so we can keep BYOA-pure — the file
references the user's shell env vars rather than baking values.  Tested
config layout per https://opencode.ai/docs/config/ and ``mcp.ts`` source.

MCP config: ``./opencode.json``'s top-level ``mcp`` key.  Each entry is
``{"type": "local", "command": [...], "environment": {...}}`` for stdio
servers.  We write this directly — ``opencode mcp add`` is interactive
only (no flags), so non-interactive setup must hand-write the JSON.

Model whitelist: pass-through for known providers (pulled from models.dev)
and fully user-controlled for custom providers via ``provider.<id>.models``.
Unlike cline/roo, opencode does NOT rewrite gateway-aliased model names.

Batch mode: ``opencode run --dir <path> --dangerously-skip-permissions
-m <provider/model> <prompt>``.  ``--dir`` is the cwd flag (not ``-C`` /
``--cwd``).  Stdin appends to the prompt when not a TTY.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _append_or_write,
    _DSAGT_MARKER,
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
    """Render ``opencode.json`` body.

    *mcp_env* — env vars baked into each MCP server's ``environment``
    block (DSAGT_PROJECT_DIR, MLFLOW_TRACKING_URI, EMBEDDING_*).

    *present_creds* — flags (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``,
    ``ANTHROPIC_API_KEY``, ``ANTHROPIC_BASE_URL``) — only emit provider
    blocks whose API key the user has set.

    *opencode_model* — ``<provider>/<model>`` string from
    ``OPENCODE_MODEL`` env.  Lab-gateway-aliased names like
    ``claude-haiku-4-5-20251001-v1-project`` aren't in models.dev, so
    opencode rejects them under standard providers unless declared in
    ``provider.<id>.models``.  We register the model there at init
    time and set the top-level ``model`` so interactive ``opencode``
    sessions pick it up without a ``-m`` flag.
    """
    config: dict = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {},
    }
    for server in ("registry", "knowledge"):
        entry: dict = {
            "type": "local",
            "command": ["uv"] + _mcp_server_args(server),
            "enabled": True,
        }
        if mcp_env:
            entry["environment"] = dict(mcp_env)
        config["mcp"][f"dsagt-{server}"] = entry

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
    # map so opencode accepts gateway-aliased names that aren't in
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


def _render_opencode_config_proxy(
    mcp_env: dict, proxy_port: int, opencode_model: str | None,
) -> str:
    """Phase-2 proxy-mode opencode.json body.

    All provider routes hit ``http://localhost:<proxy_port>`` regardless
    of wire protocol — the proxy normalizes both /v1/messages and
    /v1/chat/completions to the upstream's /chat/completions endpoint.
    Sentinel API key plants a clear failure mode if a request bypasses
    the proxy (the user sees 401, not silent leakage).

    Like the BYOA renderer, we register the user's chosen model under
    ``provider.<id>.models`` so gateway-aliased names not in models.dev
    don't trip ProviderModelNotFoundError.
    """
    from .base import _PROXY_FORWARDED_SENTINEL
    proxy_url = f"http://localhost:{proxy_port}"
    config: dict = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {},
    }
    for server in ("registry", "knowledge"):
        entry: dict = {
            "type": "local",
            "command": ["uv"] + _mcp_server_args(server),
            "enabled": True,
        }
        if mcp_env:
            entry["environment"] = dict(mcp_env)
        config["mcp"][f"dsagt-{server}"] = entry

    # Both providers point at the proxy — opencode picks via model prefix.
    providers: dict = {
        "openai": {"options": {
            "apiKey": _PROXY_FORWARDED_SENTINEL,
            "baseURL": proxy_url,
        }},
        "anthropic": {"options": {
            "apiKey": _PROXY_FORWARDED_SENTINEL,
            "baseURL": proxy_url,
        }},
    }

    if opencode_model and "/" in opencode_model:
        provider_id, model_id = opencode_model.split("/", 1)
        if provider_id in providers:
            providers[provider_id]["models"] = {
                model_id: {"name": model_id},
            }
            config["model"] = opencode_model

    config["provider"] = providers
    return json.dumps(config, indent=2)


class OpenCodeSetup(AgentSetup):
    name = "opencode"
    base_command = ["opencode"]
    static_marker = "AGENTS.md"
    install_hint = "Install with `npm i -g opencode-ai`."
    otel_payload_support = "none"
    # OpenCode reads provider creds via ``{env:VAR}`` interpolation in
    # its config — the file references these vars, opencode resolves
    # them from the user's shell at runtime.  Same shape as goose's
    # multi-protocol story, no on-disk credential leakage.
    credential_env_vars = (
        "OPENCODE_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    )
    telemetry_env = {}
    credential_hints = (
        ("OPENCODE_MODEL", "model spec '<provider>/<name>' "
         "(e.g. 'openai/claude-haiku-4-5-20251001-v1-project' for a PNNL-shape "
         "openai gateway, or 'anthropic/claude-sonnet-4-5')"),
        ("OPENAI_API_KEY", "if your gateway speaks openai wire protocol"),
        ("OPENAI_BASE_URL", "openai gateway URL "
         "(referenced by opencode.json's provider.openai.options.baseURL)"),
        ("ANTHROPIC_API_KEY", "if your gateway speaks anthropic wire protocol"),
        ("ANTHROPIC_BASE_URL", "anthropic gateway URL"),
    )

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / "AGENTS.md", instructions, _DSAGT_MARKER,
            )
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
        provider interpolation refs.  Auth keys never land on disk —
        ``{env:VAR}`` is a reference, opencode resolves it at run time
        from the user's shell.
        """
        del pdir
        actions: list[str] = []
        mcp_env = _mcp_env_block(config)
        # Detect which provider blocks to emit by probing the env we'll
        # pass to the agent (which mirrors os.environ).  We only emit
        # blocks for providers the user actually has creds for; an empty
        # ``{env:VAR}`` interpolation would leave opencode trying to
        # auth with a blank string.
        present = {
            name: bool(env.get(name))
            for name in (
                "OPENAI_API_KEY", "OPENAI_BASE_URL",
                "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
            )
        }
        body = _render_opencode_config(
            mcp_env, present, opencode_model=env.get("OPENCODE_MODEL"),
        )
        config_path = working_dir / "opencode.json"
        config_path.write_text(body + "\n")
        n_providers = sum(
            1 for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
            if present.get(k)
        )
        actions.append(
            f"Wrote {config_path} ({len(mcp_env)} MCP env vars, "
            f"{n_providers} provider block(s))"
        )
        return actions

    def proxy_write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        """Phase-2 proxy-mode setup.  All provider routes point at
        ``http://localhost:<proxy_port>``; the user's gateway creds live
        in the proxy subprocess via ``LLM_API_KEY``.  Same OPENCODE_MODEL
        env var as BYOA — must be ``<provider>/<name>``.
        """
        del env, pdir
        actions: list[str] = []
        proxy_port = (config.get("proxy") or {}).get("port")
        opencode_model = (config.get("llm") or {}).get("model")
        # opencode_model from config["llm"]["model"] is just a name; the
        # user's OPENCODE_MODEL ``<provider>/<name>`` form takes precedence
        # if set, otherwise default to ``openai/<llm.model>`` since the
        # proxy speaks /chat/completions on every upstream.
        if opencode_model and "/" not in opencode_model:
            opencode_model = f"openai/{opencode_model}"
        if not proxy_port or not opencode_model:
            raise RuntimeError(
                "opencode proxy_write_dynamic requires "
                "config['proxy']['port'] and config['llm']['model']."
            )

        mcp_env = _mcp_env_block(config)
        body = _render_opencode_config_proxy(mcp_env, proxy_port, opencode_model)
        config_path = working_dir / "opencode.json"
        config_path.write_text(body + "\n")
        actions.append(
            f"Wrote {config_path} ({len(mcp_env)} MCP env vars, proxy mode)"
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
        lets unattended runs auto-approve all tool calls.  ``-m`` overrides
        the model from ``OPENCODE_MODEL`` (must be ``<provider>/<name>``
        format — opencode rejects bare model names without a provider
        prefix).  ``max_turns`` is unused — opencode has no turn cap CLI.
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
                "'openai/claude-haiku-4-5-20251001-v1-project').  Plus the "
                "matching {ANTHROPIC,OPENAI}_API_KEY / _BASE_URL.  See "
                "agents/opencode.py credential_hints."
            )
        cmd = [
            "opencode", "run",
            "--dir", str(working_dir),
            "--dangerously-skip-permissions",
            "-m", model,
            text,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

"""
Codex agent setup.

Install: ``npm i -g @openai/codex`` (or ``brew install --cask codex``).
Generates: ``AGENTS.md``, ``.codex-data/`` (the per-project ``CODEX_HOME``),
``.dsagt_env``.

Codex's config layer has no per-workspace file — instead it reads
``$CODEX_HOME/config.toml`` for everything (model, provider base_url, MCP
servers).  We point ``CODEX_HOME`` at ``<working_dir>/.codex-data`` to
keep state isolated per project, and :meth:`CodexSetup.write_dynamic`
writes ``[mcp_servers.*]`` blocks with explicit env (codex doesn't
inherit parent env into MCP children, same as cline/roo).

Post-proxy: the user owns model + provider config in
``$CODEX_HOME/config.toml`` (or via ``OPENAI_API_KEY`` /
``OPENAI_BASE_URL`` env).  We don't write a model_providers block
anymore.

OTel support: **partial** (verified, ``otel_payload_support = "partial"``).
Codex's ``codex-otel`` Rust crate emits OTel spans/logs/metrics, BUT:

  * LLM-call spans (``stream_request``, ``handle_responses``) carry
    only ``tool_name``, ``gen_ai.usage.*_tokens``, and routing
    metadata — **NOT** the request ``messages`` array, **NOT** the
    assistant text response, **NOT** the tool-call arguments.  Cited
    from ``codex-rs/core/src/session/turn.rs:1838-1867`` and
    ``codex-rs/otel/src/events/session_telemetry.rs:292-327``.
  * User prompts go to a separate log event (``codex.user_prompt``)
    that is **REDACTED by default** — only emitted when the
    ``[otel]`` table sets ``log_user_prompt = true``.
  * Codex does not honor standard ``OTEL_EXPORTER_OTLP_*`` env vars;
    config lives in ``~/.codex/config.toml`` ``[otel]`` table.
  * Tool *results* go to ``codex.tool_result`` log events with full
    args + output (``session_telemetry.rs:962-1000``).

Conversation history *is* recoverable from
``$CODEX_HOME/sessions/rollout-<ts>-<uuid>.jsonl`` (full assistant text
+ tool calls + responses).  Reading that side-channel into
``extract_session`` is a TODO — not yet wired.  Until then, memory
extraction will see only token counts and tool names from Codex
turns.

Open Codex issues tracking richer OTel: openai/codex#12913,
#10277, #6153, #16248.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    AgentSetup,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_env_block,
    _mcp_server_args,
    _run_simple_script,
    _toml_quote,
)


def _render_codex_config(mcp_env: dict) -> str:
    """Render the per-project ``$CODEX_HOME/config.toml`` body.

    Emits ``[mcp_servers.*]`` sections plus an ``[otel]`` block.

    No top-level keys are emitted, so the output can be safely appended
    to a copy of the user's ``~/.codex/config.toml`` without colliding
    on top-level keys like ``model`` or ``approval_policy``.  Batch-mode
    approval / sandbox are set on the codex CLI directly
    (``--dangerously-bypass-approvals-and-sandbox``) instead of here.

    The ``[otel]`` block opts codex's partial OTel into MLflow:

    * ``trace_exporter = "otlp-http"`` selects the OTLP-over-HTTP
      exporter, which then reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` /
      ``OTEL_EXPORTER_OTLP_HEADERS`` from env (set by ``agent_env``
      in batch mode; the user sets them in their shell per the
      ``dsagt init`` printout for interactive mode).  Without this
      key codex's traces never leave the process.
    * ``log_user_prompt = true`` flips the ``codex.user_prompt`` log
      event from REDACTED to full text, which is what memory
      extraction needs to see what the user asked for.

    Codex's LLM-call spans still carry only token counts + tool names
    (no request messages, no response bodies) — that's a codex
    bundle limitation, not something we can override.  Tool *results*
    do land in ``codex.tool_result`` log events with full args + output.
    """
    lines: list[str] = []
    lines.append("[mcp_servers.dsagt]")
    lines.append('command = "uv"')
    args = _mcp_server_args()
    args_toml = ", ".join(_toml_quote(a) for a in args)
    lines.append(f"args = [{args_toml}]")
    if mcp_env:
        lines.append("[mcp_servers.dsagt.env]")
        for k, v in mcp_env.items():
            lines.append(f"{k} = {_toml_quote(v)}")
    lines.append("")
    lines.extend(
        [
            "[otel]",
            'trace_exporter = "otlp-http"',
            "log_user_prompt = true",
            "",
        ]
    )
    return "\n".join(lines)


def _render_codex_config_proxy(mcp_env: dict, proxy_port: int, model: str) -> str:
    """Phase-2 proxy-mode config.toml body.

    Adds, on top of the BYOA MCP-server registrations:
      * ``model_provider = "dsagt-proxy"`` (top-level, picks our provider)
      * ``[model_providers.dsagt-proxy]`` block with ``base_url`` pointing
        at the localhost proxy and ``wire_api = "chat"`` (the proxy
        speaks /chat/completions on every upstream).
      * Top-level ``model = <upstream model name>`` so the user doesn't
        need to pass ``-m`` on every ``codex exec`` call.

    These keys go ABOVE the ``[mcp_servers.*]`` sections that
    ``_render_codex_config`` produces — top-level keys must precede
    any table headers in TOML.
    """
    base = _render_codex_config(mcp_env)
    header = "\n".join(
        [
            f'model = "{model}"',
            'model_provider = "dsagt-proxy"',
            "",
            "[model_providers.dsagt-proxy]",
            'name = "DSAGT Proxy"',
            f'base_url = "http://localhost:{proxy_port}/v1"',
            'wire_api = "chat"',
            "",
        ]
    )
    return header + base


class CodexSetup(AgentSetup):
    name = "codex"
    base_command = ["codex"]
    static_marker = "AGENTS.md"
    # Project-local .agents/skills (repo-root, codex-discovered) — never the
    # global ~/.agents/skills or ~/.codex; manifest-tracked, user skills safe.
    native_skills_dir = ".agents/skills"
    install_hint = (
        "Install with `npm i -g @openai/codex` or " "`brew install --cask codex`."
    )
    otel_payload_support = "partial"
    # Codex is openai-protocol native.
    credential_env_vars = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
    # Codex's OTel is config-file driven, not env; spans carry no
    # message payload anyway, so we don't surface OTel hints here.
    telemetry_env = {}
    credential_hints = (
        ("OPENAI_API_KEY", "your OpenAI API key (skip if subscription-authed)"),
        ("OPENAI_BASE_URL", "optional gateway / proxy URL"),
    )

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        (working_dir / ".codex-data").mkdir(parents=True, exist_ok=True)
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / "AGENTS.md",
                instructions,
                _DSAGT_MARKER,
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
        """Set up the per-project ``CODEX_HOME`` directory.

        Three concerns:

        1. **MCP server registration.**  Codex looks up MCP servers in
           ``$CODEX_HOME/config.toml``; children don't inherit parent
           env, so the env block is explicit per server.
        2. **Subscription auth propagation.**  Codex stores
           ChatGPT-subscription tokens in ``~/.codex/auth.json``.  Our
           isolated ``CODEX_HOME`` would have no auth → codex sends an
           anonymous request to ``api.openai.com`` and 401s.  We copy
           ``~/.codex/auth.json`` (if present) into ``.codex-data/``.
           API-key users (``OPENAI_API_KEY`` set) don't need this.
        3. **User config preservation.**  We copy ``~/.codex/config.toml``
           as a base so user prefs (default model, approval mode, etc.)
           carry through, then append our ``[mcp_servers.*]`` sections.
           Top-level keys don't collide because ``_render_codex_config``
           only emits MCP sections.
        """
        del env, pdir
        import shutil

        actions: list[str] = []
        codex_home = Path(working_dir) / ".codex-data"
        codex_home.mkdir(parents=True, exist_ok=True)
        user_codex = Path.home() / ".codex"

        # Propagate subscription auth from user's global codex state.
        # auth.json holds OAuth/API tokens; we don't read its contents,
        # just file-copy it so codex's normal auth flow works under our
        # isolated CODEX_HOME.
        user_auth = user_codex / "auth.json"
        if user_auth.exists():
            dest_auth = codex_home / "auth.json"
            shutil.copy2(user_auth, dest_auth)
            dest_auth.chmod(0o600)
            actions.append(f"Copied {user_auth} → {dest_auth}")

        # Build config.toml = user's prefs + our MCP sections.
        config_path = codex_home / "config.toml"
        user_config = user_codex / "config.toml"
        base_toml = user_config.read_text() if user_config.exists() else ""
        mcp_env = _mcp_env_block(config)
        body = _render_codex_config(mcp_env)
        merged = base_toml.rstrip() + "\n\n" + body if base_toml else body
        config_path.write_text(merged + "\n")
        actions.append(f"Wrote {config_path} ({len(mcp_env)} MCP env vars)")
        return actions

    def launch_oneliner(self, project: str, project_dir: Path) -> str:
        """Codex reads MCP servers from ``$CODEX_HOME/config.toml`` and
        has no ``--config`` flag — must inline ``CODEX_HOME`` pointing
        at our per-project ``.codex-data/``, otherwise codex reads
        ``~/.codex`` and our MCP servers don't register.
        """
        del project
        import shlex

        pdir = shlex.quote(str(project_dir))
        codex_home = shlex.quote(str(project_dir / ".codex-data"))
        return f"cd {pdir} && CODEX_HOME={codex_home} codex"

    def proxy_write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        """Phase-2 proxy-mode setup.  Same MCP / auth / user-config
        copy as BYOA, but renders config.toml with the
        ``[model_providers.dsagt-proxy]`` block so codex routes its
        LLM calls through our localhost proxy.  No subscription auth
        needed in proxy mode — the proxy authenticates upstream with
        ``LLM_API_KEY``; we plant a sentinel API key in the codex env
        so direct calls bypassing the proxy fail loudly.
        """
        del env, pdir
        import shutil

        actions: list[str] = []
        codex_home = Path(working_dir) / ".codex-data"
        codex_home.mkdir(parents=True, exist_ok=True)
        user_codex = Path.home() / ".codex"

        # Still copy user's auth.json + config.toml as a base — they may
        # have other prefs we want to preserve.  Proxy routing layered on
        # top via the _proxy renderer.
        user_auth = user_codex / "auth.json"
        if user_auth.exists():
            dest_auth = codex_home / "auth.json"
            shutil.copy2(user_auth, dest_auth)
            dest_auth.chmod(0o600)
            actions.append(f"Copied {user_auth} → {dest_auth}")

        proxy_port = (config.get("proxy") or {}).get("port")
        model = (config.get("llm") or {}).get("model")
        if not proxy_port or not model:
            raise RuntimeError(
                "codex proxy_write_dynamic requires "
                "config['proxy']['port'] and config['llm']['model']."
            )

        config_path = codex_home / "config.toml"
        user_config = user_codex / "config.toml"
        base_toml = user_config.read_text() if user_config.exists() else ""
        mcp_env = _mcp_env_block(config)
        body = _render_codex_config_proxy(mcp_env, proxy_port, model)
        merged = base_toml.rstrip() + "\n\n" + body if base_toml else body
        config_path.write_text(merged + "\n")
        actions.append(f"Wrote {config_path} ({len(mcp_env)} MCP env vars, proxy mode)")
        return actions

    def runtime_env(self, config: dict) -> dict[str, str]:
        """BYOA infrastructure: per-project ``CODEX_HOME`` (state dir)
        plus inherited telemetry flags.  Isolates per-project MCP /
        config.toml from the global ``~/.codex``.
        """
        env = super().runtime_env(config)
        env["CODEX_HOME"] = str(Path(config["project_dir"]) / ".codex-data")
        return env

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``codex exec`` call in non-interactive mode."""
        del config, max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        cmd = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C",
            str(working_dir),
            text,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

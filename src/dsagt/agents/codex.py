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
inherit parent env into MCP children, same as cline).

The user owns model + provider config in ``$CODEX_HOME/config.toml``
(or via ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env).  We don't write
a model_providers block.

OTel support: **partial** (verified).
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

Conversation history is recovered from
``$CODEX_HOME/sessions/rollout-<ts>-<uuid>.jsonl`` (full assistant text
+ tool calls + responses) by the trace pipeline's Codex reader/translator
on the heartbeat — feeding MLflow and episodic memory like every other agent.

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

    Emits only ``[mcp_servers.*]`` sections.  No top-level keys are
    emitted, so the output can be safely appended to a copy of the user's
    ``~/.codex/config.toml`` without colliding on top-level keys like
    ``model`` or ``approval_policy``.  Batch-mode approval / sandbox are
    set on the codex CLI directly
    (``--dangerously-bypass-approvals-and-sandbox``) instead of here.

    No ``[otel]`` block: DSAGT no longer forces codex's native telemetry
    (nor the ``log_user_prompt`` privacy override).  Codex's conversation
    history is recovered post-hoc from its on-disk session rollout.
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
    return "\n".join(lines)


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

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [working_dir / "AGENTS.md", working_dir / ".codex-data"]

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

    def runtime_env(self, config: dict) -> dict[str, str]:
        """BYOA infrastructure: per-project ``CODEX_HOME`` (state dir).

        Isolates per-project MCP / config.toml from the global
        ``~/.codex``.  Codex has no ``--config`` flag, so the agent must
        be launched with this ``CODEX_HOME`` set for our MCP servers to
        register.
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

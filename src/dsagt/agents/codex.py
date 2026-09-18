"""
Codex agent setup.

Install: ``npm i -g @openai/codex`` (or ``brew install --cask codex``).
Generates: ``AGENTS.md``, ``.codex-data/`` (the per-project ``CODEX_HOME``),
``.dsagt_env``.

Codex reads ``$CODEX_HOME/config.toml`` for everything (model, provider
base_url, MCP servers) and has no per-workspace file.  ``CODEX_HOME`` is
pointed at ``<working_dir>/.codex-data`` to keep state isolated per
project, and :meth:`CodexSetup.write_dynamic` writes ``[mcp_servers.*]``
blocks with explicit env (codex starts MCP children with only the env
written there, as cline does).

The user owns model and provider config in ``$CODEX_HOME/config.toml``
(or via ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env); dsagt writes only
the MCP sections.

OTel support: partial (verified).  Codex's ``codex-otel`` Rust crate emits
OTel spans, logs, and metrics, with these limits:

  * LLM-call spans (``stream_request``, ``handle_responses``) carry
    only ``tool_name``, ``gen_ai.usage.*_tokens``, and routing
    metadata: the request ``messages`` array, the assistant text
    response, and the tool-call arguments are absent.  Cited from
    ``codex-rs/core/src/session/turn.rs:1838-1867`` and
    ``codex-rs/otel/src/events/session_telemetry.rs:292-327``.
  * User prompts go to a separate log event (``codex.user_prompt``)
    that is redacted by default and emitted only when the ``[otel]``
    table sets ``log_user_prompt = true``.
  * Codex reads its OTel exporter settings from the ``[otel]`` table of
    ``~/.codex/config.toml``.
  * Tool results go to ``codex.tool_result`` log events with full
    args and output (``session_telemetry.rs:962-1000``).

Conversation history is recovered from
``$CODEX_HOME/sessions/rollout-<ts>-<uuid>.jsonl`` (full assistant text,
tool calls, and responses) by the Codex reader and translator on the
periodic pass, which feeds MLflow and episodic memory the same way as
for every other agent.

Open Codex issues tracking richer OTel: openai/codex#12913,
#10277, #6153, #16248.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    AgentSetup,
    _write_dsagt_block,
    _load_master_instructions,
    _mcp_env_block,
    _mcp_server_args,
    _run_simple_script,
    _toml_quote,
)

#: Appended to the master instructions in ``AGENTS.md``.  Codex adds an MCP
#: server's tools to the model's tool list only after a ``tool_search`` call,
#: and a search that runs before the server has answered ``initialize``
#: returns nothing, which the model otherwise reads as "no such tools".
_TOOL_SEARCH_NOTE = """
## CODEX: LOADING THE DSAGT TOOLS

The tools named in this file (`kb_remember`, `search_registry`, `save_code_spec`, `reconstruct_pipeline`, and the rest) are MCP tools in the `mcp__dsagt` namespace. Codex adds them to your tool list only after a `tool_search` call: `tool_search(query="dsagt")` returns the `mcp__dsagt` namespace with every dsagt tool, so run it once before the first dsagt tool call of a session. An empty result means the dsagt server was still starting; run the same search again before concluding the tools are absent. A dsagt tool missing from your tool list has not been loaded yet: search for it. Do not substitute a shell command or a Python import for it.
"""


def _render_codex_config(mcp_env: dict) -> str:
    """Render the per-project ``$CODEX_HOME/config.toml`` body.

    Emits only ``[mcp_servers.*]`` sections, so the output appends to a
    copy of the user's ``~/.codex/config.toml`` without colliding on
    top-level keys like ``model`` or ``approval_policy``.  Batch-mode
    approval and sandbox are set on the codex CLI
    (``--dangerously-bypass-approvals-and-sandbox``).

    Codex's native telemetry, including the ``log_user_prompt`` privacy
    setting, stays the user's: codex's conversation history is recovered
    from its on-disk session rollout.
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
    # Project-local .agents/skills (repo-root, codex-discovered), never the
    # global ~/.agents/skills or ~/.codex; manifest-tracked, so user skills
    # are left in place.
    native_skills_dir = ".agents/skills"
    install_hint = (
        "Install with `npm i -g @openai/codex` or " "`brew install --cask codex`."
    )

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [working_dir / "AGENTS.md", working_dir / ".codex-data"]

    def write_static(self, working_dir: Path, *, auto_assess: bool = True) -> list[str]:
        actions: list[str] = []
        (working_dir / ".codex-data").mkdir(parents=True, exist_ok=True)
        instructions = _load_master_instructions(auto_assess)
        if instructions:
            action = _write_dsagt_block(
                working_dir / "AGENTS.md", instructions + _TOOL_SEARCH_NOTE
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

        1. MCP server registration.  Codex looks up MCP servers in
           ``$CODEX_HOME/config.toml``; children receive only the env
           written there, so the env block is explicit per server.
        2. Subscription auth propagation.  Codex stores
           ChatGPT-subscription tokens in ``~/.codex/auth.json``.  Without
           it the isolated ``CODEX_HOME`` has no auth, and codex sends an
           anonymous request to ``api.openai.com`` and gets a 401.
           ``~/.codex/auth.json`` (if present) is copied into
           ``.codex-data/``.  API-key users (``OPENAI_API_KEY`` set) are
           authenticated without it.
        3. User config preservation.  ``~/.codex/config.toml`` is copied
           as the base so user prefs (default model, approval mode, etc.)
           carry through, then the ``[mcp_servers.*]`` sections are
           appended.  ``_render_codex_config`` emits only MCP sections, so
           top-level keys never collide.
        """
        del env, pdir
        import shutil

        actions: list[str] = []
        codex_home = Path(working_dir) / ".codex-data"
        codex_home.mkdir(parents=True, exist_ok=True)
        user_codex = Path.home() / ".codex"

        # Propagate subscription auth from the user's global codex state.
        # auth.json holds OAuth/API tokens; it is copied as a file, unread,
        # so codex's normal auth flow works under the isolated CODEX_HOME.
        user_auth = user_codex / "auth.json"
        if user_auth.exists():
            dest_auth = codex_home / "auth.json"
            shutil.copy2(user_auth, dest_auth)
            dest_auth.chmod(0o600)
            actions.append(f"Copied {user_auth} to {dest_auth}")

        # Build config.toml: the user's prefs plus the MCP sections.
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
        """Per-project ``CODEX_HOME`` (state dir).

        Isolates per-project MCP config and config.toml from the global
        ``~/.codex``.  Codex has no ``--config`` flag, so the agent must
        be launched with this ``CODEX_HOME`` set for the dsagt MCP server
        to register.
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

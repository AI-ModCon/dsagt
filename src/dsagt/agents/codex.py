"""
Codex agent setup.

Install: ``npm i -g @openai/codex`` (or ``brew install --cask codex``).
Generates: ``AGENTS.md``, ``.codex-data/`` (the per-project ``CODEX_HOME``),
``.dsagt_env``.

Codex's config layer has no per-workspace file — instead it reads
``$CODEX_HOME/config.toml`` for everything (model, provider base_url, MCP
servers).  We point ``CODEX_HOME`` at ``<working_dir>/.codex-data`` to keep
state isolated per project, and :meth:`CodexSetup.write_dynamic` writes
``config.toml`` at launch with a custom ``[model_providers.dsagt-proxy]``
block routing to our proxy and ``[mcp_servers.*]`` blocks with explicit
env (codex doesn't inherit parent env into MCP children, same as
cline/roo).

Wire API: Codex's only supported value is ``responses``, so requests hit
``/v1/responses`` on the proxy — the same path claude and roo already
exercise.  ``disable_response_storage = true`` is required because the
upstream gateway behind our proxy translates ``/v1/responses`` →
``/v1/chat/completions`` and can't preserve ``previous_response_id``
state across calls.

*Upstream requirement*: Codex needs an OpenAI-shape upstream (e.g. PNNL
ai-incubator-api).  Codex+Bedrock is **not supported**: codex 0.125
wraps MCP tools in a non-standard ``{type: namespace, ...}`` schema that
Bedrock's Anthropic Messages adapter doesn't unpack, and flattening at
the proxy breaks codex's internal namespace dispatcher on the way back.
For Bedrock use claude, goose, cline, or roo.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    AgentSetup,
    _PROXY_FORWARDED_SENTINEL,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_server_args,
    _mcp_subprocess_env,
    _run_simple_script,
    _toml_quote,
    _write_env_file,
)


def _render_codex_config(config: dict, env: dict, mcp_env: dict) -> str:
    """Render the per-project ``$CODEX_HOME/config.toml`` body.

    Pinned shape (see module docstring for the why):
      - ``model_provider = "dsagt-proxy"`` routes every call through our
        local LiteLLM proxy.  Built-in ``openai`` provider is left intact
        so a future debug session can flip back via ``-c
        model_provider=openai`` without rewriting config.
      - ``wire_api = "responses"`` is Codex's only supported value.
        Proxy already exercises this path for claude and roo.
      - ``requires_openai_auth = false`` skips the ``codex login`` OAuth
        flow — the sentinel ``OPENAI_API_KEY`` env var is enough.
      - ``disable_response_storage = true`` because the upstream behind
        our proxy translates ``/v1/responses`` → ``/v1/chat/completions``
        and can't honor ``previous_response_id`` state.  Without this,
        the second turn 400s with "previous_response_id not found".
      - ``approval_policy = "never"`` and ``sandbox_mode =
        "danger-full-access"`` so batch mode runs without approval
        prompts.  ``codex exec --yolo`` sets the same at the CLI but
        also writing to config makes interactive ``dsagt start``
        sessions behave consistently.
    """
    del env
    proxy_port = config["proxy"]["port"]
    model = config["llm"]["model"]
    base_url = f"http://localhost:{proxy_port}/v1"

    lines = [
        f"model = {_toml_quote(model)}",
        'model_provider = "dsagt-proxy"',
        'approval_policy = "never"',
        'sandbox_mode = "danger-full-access"',
        "disable_response_storage = true",
        "",
        "[model_providers.dsagt-proxy]",
        'name = "DSAgt Proxy"',
        f"base_url = {_toml_quote(base_url)}",
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "responses"',
        "requires_openai_auth = false",
        "",
    ]
    for server in ("registry", "knowledge"):
        lines.append(f"[mcp_servers.dsagt-{server}]")
        lines.append('command = "uv"')
        args = _mcp_server_args(server)
        args_toml = ", ".join(_toml_quote(a) for a in args)
        lines.append(f"args = [{args_toml}]")
        if mcp_env:
            lines.append(f"[mcp_servers.dsagt-{server}.env]")
            for k, v in mcp_env.items():
                lines.append(f"{k} = {_toml_quote(v)}")
        lines.append("")
    return "\n".join(lines)


class CodexSetup(AgentSetup):
    name = "codex"
    base_command = ["codex"]
    static_marker = "AGENTS.md"
    install_hint = (
        "Install with `npm i -g @openai/codex` or "
        "`brew install --cask codex`."
    )

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        (working_dir / ".codex-data").mkdir(parents=True, exist_ok=True)
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
        proxy_port: int,
    ) -> list[str]:
        """Codex's runtime-dependent setup: ``$CODEX_HOME/config.toml`` +
        ``.dsagt_env``.  Codex MCP children, like cline's and roo's, don't
        inherit parent process env — every var the dsagt servers need has
        to be explicit in the per-server ``[mcp_servers.*.env]`` table.
        """
        del proxy_port  # read from config inside _render_codex_config
        actions: list[str] = []
        codex_home = env.get("CODEX_HOME") or str(working_dir / ".codex-data")
        Path(codex_home).mkdir(parents=True, exist_ok=True)
        config_path = Path(codex_home) / "config.toml"
        mcp_env = _mcp_subprocess_env(env)
        config_path.write_text(_render_codex_config(config, env, mcp_env) + "\n")
        actions.append(f"Wrote {config_path} ({len(mcp_env)} MCP env vars)")

        env_path = working_dir / ".dsagt_env"
        _write_env_file(env_path, {
            "CODEX_HOME": str(codex_home),
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
        })
        actions.append(f"Wrote {env_path}")
        return actions

    def env_overrides(self, config: dict, proxy_port: int) -> dict[str, str]:
        # Codex reads model + provider + base_url + MCP servers from
        # $CODEX_HOME/config.toml (no per-workspace file).  Per-project
        # CODEX_HOME isolates state so multiple dsagt projects don't
        # collide on global config.  config.toml is written by
        # write_dynamic.
        #
        # Codex's [model_providers.dsagt-proxy] config.toml entry sets
        # env_key = "OPENAI_API_KEY", so the auth header sent to our proxy
        # is read from this var.  Sentinel value: any direct call
        # bypassing the proxy 401s loudly at the real upstream.
        del proxy_port
        return {
            "CODEX_HOME": str(Path(config["project_dir"]) / ".codex-data"),
            "OPENAI_API_KEY": _PROXY_FORWARDED_SENTINEL,
        }

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``codex exec --yolo --skip-git-repo-check -C <wd> <prompt>``
        call.  ``codex exec`` is Codex's non-interactive mode — runs the
        prompt to completion and exits with the assistant's final message
        on stdout.

        ``--yolo`` is Codex's alias for
        ``--dangerously-bypass-approvals-and-sandbox`` (skips both shell-
        command approval prompts and sandboxing, which batch mode can't
        answer).  ``--skip-git-repo-check`` allows running outside a git
        repo (smoke-test project dirs aren't git repos).

        ``max_turns`` is unused — Codex has no exposed turn cap.  Wall-
        clock cap in the smoke wrapper is the safety net.
        """
        del config, max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        cmd = [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C", str(working_dir),
            text,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

"""
Goose agent setup.

Install: see https://github.com/block/goose.
Generates: ``goose.yaml``, ``.goosehints``, ``.dsagt_env``.

Post-proxy: Goose talks directly to the user's provider configured via
its own ``~/.config/goose/config.yaml`` or ``GOOSE_PROVIDER`` /
``GOOSE_MODEL`` env.  We only inject the OTel endpoint env so Goose's
native ``dispatch_tool_call`` span (with full ``{tool, arguments}`` JSON
in ``input``) lands in the project's MLflow.

OTel support: **full** for native MLflow visibility (verified by
source review of ``crates/goose/src/agents/agent.rs:572-585``).  Goose
creates a ``dispatch_tool_call`` span with attribute
``input = {"tool": <name>, "arguments": {...}}`` and ``session.id``,
activating automatically when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set
(handled by ``agents/__init__.py:agent_env``).  No payload-stripping
flag exists — args are exported verbatim.

Memory extraction: **does NOT work without --enable-proxy**, even
though Goose's traces are visible in the MLflow UI.  The
``dispatch_tool_call`` span is in Goose's domain schema, not the
LiteLLM-autolog ``mlflow.spanInputs`` / ``mlflow.spanOutputs`` shape
that ``memory.drain_session_traces`` reads.  Users who want both
visibility *and* extraction should run ``dsagt start --enable-proxy``.
See ``agents/__init__.py`` module docstring for the design rationale.

Caveat: tool *outputs* are NOT in the span (the ``output`` field is
declared but never written).  ``dsagt-run``'s ``tool.execute`` spans
cover the execution layer with stdout/stderr/exit-code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import (
    AgentSetup,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_server_args,
    _run_simple_script,
)


class GooseSetup(AgentSetup):
    name = "goose"
    base_command = ["goose", "session"]
    static_marker = ".goosehints"
    install_hint = "See https://github.com/block/goose for installation."
    otel_payload_support = "full"
    # Multi-protocol; goose's runtime reads provider-specific creds plus
    # its own GOOSE_PROVIDER / GOOSE_MODEL routing selectors.
    # Goose's openai/anthropic providers read ``OPENAI_HOST`` /
    # ``ANTHROPIC_HOST`` for the base URL (a goose-specific naming
    # convention from its Rust client), NOT the standard
    # ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` everything else uses.
    # Without HOST set, goose ignores BASE_URL and hits the provider's
    # default endpoint — silently for users with a lab gateway.
    credential_env_vars = (
        "GOOSE_PROVIDER", "GOOSE_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_HOST",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_HOST",
    )
    # Goose's Rust client emits OTel automatically when
    # OTEL_EXPORTER_OTLP_ENDPOINT is set — no per-platform flags needed.
    telemetry_env = {}
    credential_hints = (
        ("GOOSE_PROVIDER", "anthropic, openai, etc. (skip if global ~/.config/goose configured)"),
        ("GOOSE_MODEL", "the model name your provider serves"),
        ("ANTHROPIC_API_KEY", "if GOOSE_PROVIDER=anthropic"),
        ("ANTHROPIC_HOST", "if GOOSE_PROVIDER=anthropic and on a gateway / proxy"),
        ("OPENAI_API_KEY", "if GOOSE_PROVIDER=openai"),
        ("OPENAI_HOST", "if GOOSE_PROVIDER=openai and on a gateway / proxy (NOT OPENAI_BASE_URL — goose ignores that)"),
    )

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / ".goosehints", instructions, _DSAGT_MARKER,
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
        """Write ``goose.yaml``.  Goose inherits parent env into MCP
        children, so extension entries don't need an explicit env list.
        """
        del config, env, pdir
        actions: list[str] = []

        goose_config: dict = {"extensions": {}}
        for server in ("registry", "knowledge"):
            args = _mcp_server_args(server)
            goose_config["extensions"][server] = {
                "enabled": True,
                "name": server,
                "type": "stdio",
                "cmd": "uv " + " ".join(args),
                "timeout": 300,
            }

        goose_path = working_dir / "goose.yaml"
        goose_path.write_text(
            yaml.dump(goose_config, default_flow_style=False, sort_keys=False)
        )
        actions.append(f"Wrote {goose_path}")
        return actions

    def env_overrides(self, config: dict) -> dict[str, str]:
        """Phase-2 proxy-mode hook: pin ``GOOSE_PROVIDER`` and
        ``GOOSE_MODEL`` so a user's global ``~/.config/goose/config.yaml``
        doesn't override the project's configured model.

        Provider is forced to ``openai`` because the proxy speaks
        /chat/completions (openai wire protocol) regardless of upstream;
        goose's openai client posts there.  Model is the upstream-served
        name from ``config["llm"]["model"]``.
        """
        model = (config.get("llm") or {}).get("model")
        out: dict[str, str] = {"GOOSE_PROVIDER": "openai"}
        if model and not str(model).startswith("${"):
            out["GOOSE_MODEL"] = model
        return out

    def proxy_env_overrides(self, proxy_port: int) -> dict[str, str]:
        """Goose-specific proxy routing: add OPENAI_HOST / ANTHROPIC_HOST.

        The base-class default sets ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``
        which most agents read; goose ignores those and reads HOST.
        Without this override, ``--enable-proxy`` would silently leave
        goose talking to the upstream provider directly — same root
        cause as the non-proxy path's HOST mapping.
        """
        env = super().proxy_env_overrides(proxy_port)
        proxy_url = f"http://localhost:{proxy_port}"
        env["OPENAI_HOST"] = proxy_url
        env["ANTHROPIC_HOST"] = proxy_url
        return env

    def interactive_command(self, config: dict) -> list[str]:
        """Goose only reads ``~/.config/goose/config.yaml`` for extensions, not
        a project-local file — so the MCP servers are passed via
        ``--with-extension`` flags on the session command to guarantee they
        attach for this project.
        """
        del config
        cmd = list(self.base_command)
        for server in ("registry", "knowledge"):
            cmd.extend(["--with-extension", f"uv run dsagt-{server}-server"])
        return cmd

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``goose run`` call — goose's instructions file IS multi-turn."""
        del config
        env["GOOSE_MODE"] = "auto"
        cmd = ["goose", "run", "--instructions", str(script_path),
               "--max-turns", str(max_turns)]
        for server in ("registry", "knowledge"):
            cmd.extend(["--with-extension", f"uv run dsagt-{server}-server"])
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

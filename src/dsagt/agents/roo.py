"""
Roo Code agent setup.

Install: ``curl -fsSL https://raw.githubusercontent.com/RooCodeInc/Roo-Code/main/apps/cli/install.sh | sh``
(binary lands at ``~/.local/bin/roo`` — not on npm).

Generates: ``.roomodes``, ``.dsagt_env``; per-launch, ``.roo/mcp.json`` is
written by :meth:`RooSetup.write_dynamic` with the full env block (roo,
like cline, doesn't inherit parent env into MCP server children).

Proxy routing: ``ANTHROPIC_BASE_URL`` (the ``--provider anthropic`` path
uses the Anthropic SDK, which honors that env var; the CLI has no
``--base-url`` flag).

Batch mode: ``roo --print --oneshot --prompt-file FILE`` reads multi-line
prompts directly — no looping or argv encoding needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _PROXY_FORWARDED_SENTINEL,
    _append_or_write,
    _build_mcp_servers_dict,
    _DSAGT_MARKER,
    _format_roomodes,
    _load_master_instructions,
    _mcp_subprocess_env,
    _run_simple_script,
    _write_env_file,
)


class RooSetup(AgentSetup):
    name = "roo"
    base_command = ["roo"]
    static_marker = ".roomodes"
    install_hint = (
        "Install via "
        "https://github.com/RooCodeInc/Roo-Code/blob/main/apps/cli/install.sh"
    )

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        (working_dir / ".roo").mkdir(exist_ok=True)
        instructions = _load_master_instructions()
        if instructions:
            # Roo wraps the instructions in a customMode JSON envelope; the
            # master marker text survives the wrap because it's part of the
            # role definition body.
            action = _append_or_write(
                working_dir / ".roomodes",
                _format_roomodes(instructions),
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
        proxy_port: int,
    ) -> list[str]:
        """Roo's runtime-dependent setup: ``.roo/mcp.json`` + ``.dsagt_env``.

        Roo doesn't inherit parent env into MCP server children — every
        var the dsagt servers need (``${EMBEDDING_*}`` substitution
        placeholders, ``MLFLOW_TRACKING_URI``, ``DSAGT_SESSION_ID``) must
        be explicit in the JSON env block.

        ``proxy_port`` is unused — the proxy URL goes onto the agent's
        command line at launch (:meth:`run_script` passes ``--api-key`` /
        ``--model`` directly), and the env block carries everything the
        MCP children need.
        """
        del proxy_port
        actions: list[str] = []
        mcp_path = working_dir / ".roo" / "mcp.json"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_env = _mcp_subprocess_env(env)
        mcp_config = _build_mcp_servers_dict(mcp_env)
        mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
        actions.append(f"Wrote {mcp_path} ({len(mcp_env)} env vars)")

        env_path = working_dir / ".dsagt_env"
        _write_env_file(env_path, {
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
        })
        actions.append(f"Wrote {env_path}")
        return actions

    def env_overrides(self, config: dict, proxy_port: int) -> dict[str, str]:
        # Roo uses --provider anthropic.  ANTHROPIC_BASE_URL points roo at
        # the proxy.  Roo rewrites our PNNL model name into its own
        # default (``claude-sonnet-4-5``) before sending — the proxy
        # aliases that name back to the upstream primary in
        # commands/proxy_server.py _generate_config.
        return {
            "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
            "ANTHROPIC_MODEL": config["llm"]["model"],
            "ANTHROPIC_API_KEY": _PROXY_FORWARDED_SENTINEL,
        }

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``roo --print --oneshot --prompt-file FILE`` call.

        Roo CLI exposes ``--prompt-file`` for batch input — much cleaner
        than cline (no need to argv-encode multi-line) or claude (no need
        to loop with ``--continue``).  ``--print --oneshot`` is roo's
        non-interactive single-task shape.

        Provider is ``anthropic``.  Roo's anthropic provider rewrites
        unrecognized model names (lab-gateway aliases like PNNL's
        ``claude-haiku-4-5-20251001-v1-project``) into its current default
        (``claude-sonnet-4-5`` as of v0.1.x).  We sidestep that by
        aliasing that name in the proxy config (see ``_generate_config``
        in ``commands/proxy_server.py``) so it routes to the same upstream
        as the configured primary.

        ``--mode dsagt`` activates the custom mode we defined in
        ``.roomodes`` (slug: "dsagt", see :func:`_format_roomodes`).
        Without this, roo runs in its default "code" mode and our entire
        customInstructions body — CRITICAL CONSTRAINTS, kb_remember rule,
        dsagt-run rule, everything — is dropped from the system prompt.

        ``max_turns`` is unused — roo has its own consecutive-mistake cap
        (``--consecutive-mistake-limit``) but no overall turn cap.
        """
        del max_turns
        sentinel = _PROXY_FORWARDED_SENTINEL
        cmd = [
            "roo",
            "--print", "--oneshot",
            "--mode", "dsagt",
            "--prompt-file", str(script_path),
            "--provider", "anthropic",
            "--api-key", env.get("ANTHROPIC_API_KEY", sentinel),
            "--model", config["llm"]["model"],
            "--workspace", str(working_dir),
            "--debug",
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

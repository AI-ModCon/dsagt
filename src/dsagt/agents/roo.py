"""
Roo Code agent setup.

Install: ``npm i -g @roo-code/cli`` (or curl install script —
binary lands at ``~/.local/bin/roo``).

Generates: ``.roomodes``; per-init, ``.roo/mcp.json`` is written by
:meth:`RooSetup.write_dynamic` with the full env block (roo, like cline,
doesn't inherit parent env into MCP server children).

**BYOA Phase 1 status: PUNTED for batch / smoke-test.** Roo CLI v0.1.17
hardcodes endpoints per ``--provider``:

  * ``anthropic`` / ``openai-native`` etc. routes through Roo Code Cloud
    (``ROO_CODE_PROVIDER_URL`` defaulting to ``api.roocode.com/proxy``).
    With no ``--base-url`` flag, no `*_BASE_URL` env var honored, and
    no native-SDK fallback, the CLI cannot be pointed at a self-hosted
    gateway.  Without cloud auth it falls back to the upstream provider's
    default URL (``api.openai.com`` etc.) where the lab-gateway key 401s.
  * ``bedrock`` is rejected by the CLI's ``--provider`` validator
    despite the ``AwsBedrockHandler`` existing in the bundle and
    ``awsBedrockEndpoint`` being a configurable schema field.
  * ``vercel-ai-gateway`` has a literally hardcoded URL constant
    (``AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"``); not
    even an env var.
  * The same model-whitelist substitution issue cline has would also
    bite roo's anthropic path — old_code's docstring called it out:
    "Roo rewrites our PNNL model name into its own default before
    sending — the proxy aliases that name back to the upstream primary."

Phase 2 reactivates batch mode via ``dsagt-proxy`` at ``localhost:<port>``
— roo's anthropic SDK happily talks to a localhost endpoint and the
proxy translates names back to upstream-served IDs.

Interactive use is unaffected — ``dsagt init --agent roo`` still writes
the project state, and users running roo via VS Code (where the UI
exposes ``awsBedrockEndpoint``, ``openAiNativeBaseUrl`` etc.) can drive
the project manually.

OTel support: **none** (verified, ``otel_payload_support = "none"``).
Roo Code imports zero ``@opentelemetry/*`` packages anywhere in the
agent runtime (``src/``, ``apps/cli/``, ``packages/core/``,
``packages/telemetry/``, ``packages/cloud/``).  LLM-call sites in
``src/api/providers/*.ts`` have no span/tracer wrapping.  Telemetry is
PostHog-only with payload-free events
(``LLM_COMPLETION``: token counts + cost; ``TASK_CONVERSATION_MESSAGE``:
``taskId`` + ``source``).  Standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
ignored.  Memory extraction will see no agent-conversation traces;
tool execution and KB observability still work via dsagt-run /
MCP-server spans.

Workaround: ``dsagt start --enable-proxy`` makes every roo LLM call
visible in MLflow — each agent turn becomes an inspectable trace you
can audit live or replay later.  Roo's ``--provider anthropic`` honors
``ANTHROPIC_BASE_URL`` env (which ``agent_env`` overrides to the proxy
when the flag is set), so the proxy receives every LLM call and emits
an OTel trace on roo's behalf.  Memory extraction is a downstream
consequence; the primary value is real-time transparency on the
agent's actions.  See ``commands/proxy_server.py``.

Batch mode: ``roo --print --oneshot --prompt-file FILE`` reads multi-line
prompts directly — no looping or argv encoding needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _append_or_write,
    _build_mcp_servers_dict,
    _DSAGT_MARKER,
    _format_roomodes,
    _load_master_instructions,
    _mcp_env_block,
    _run_simple_script,
)


class RooSetup(AgentSetup):
    name = "roo"
    base_command = ["roo"]
    static_marker = ".roomodes"
    native_skills_dir = ".roo/skills"
    install_hint = (
        "Install via "
        "https://github.com/RooCodeInc/Roo-Code/blob/main/apps/cli/install.sh"
    )
    otel_payload_support = "none"
    # Roo's CLI multi-provider story is misleading: openai-native has no
    # ``--base-url`` flag and the SDK behind it doesn't read
    # ``OPENAI_BASE_URL``, so reaching a non-default openai endpoint is
    # impossible with that path.  Only the anthropic provider works for
    # gateways — the Anthropic SDK natively reads ``ANTHROPIC_BASE_URL``.
    credential_env_vars = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
    )
    # Roo emits no OTel — agent-side telemetry only via --proxy_traces.
    telemetry_env = {}
    credential_hints = (
        (
            "ANTHROPIC_API_KEY",
            "your provider API key (works for openai-shape "
            "gateways too — the Anthropic SDK reaches them via ANTHROPIC_BASE_URL)",
        ),
        (
            "ANTHROPIC_BASE_URL",
            "gateway / proxy URL "
            "(roo CLI has no --base-url flag; this env var is the only way)",
        ),
        (
            "ANTHROPIC_MODEL",
            "model name your gateway serves "
            "(e.g. claude-haiku-4-5-20251001-v1-project)",
        ),
    )

    def vscode_hint(self, project_dir: Path) -> list[str]:
        return [
            f"Open {project_dir} in VS Code and start the Roo Code extension.",
            "Pick 'DSAgt Pipeline Builder' from Roo's mode dropdown.",
        ]

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

    def env_overrides(self, config: dict) -> dict[str, str]:
        """Phase-2 proxy-mode hook: pin ``ANTHROPIC_MODEL`` so roo's
        anthropic SDK posts the configured model to the proxy.

        ``proxy_env_overrides`` (base default) sets
        ``ANTHROPIC_BASE_URL`` to the proxy URL and ``ANTHROPIC_API_KEY``
        to the sentinel.  Roo's hardcoded model whitelist will rewrite
        the model name to its current default (``claude-sonnet-4-5``)
        before sending; the proxy aliases that rewrite back to the
        upstream-served name (see ``_AGENT_PRIMARY_ALIASES`` in
        proxy_server.py).
        """
        model = (config.get("llm") or {}).get("model")
        if model and not str(model).startswith("${"):
            return {"ANTHROPIC_MODEL": model}
        return {}

    def write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        """Write ``.roo/mcp.json``.  Roo doesn't inherit parent env into
        MCP children — every var the dsagt servers need
        (``MLFLOW_TRACKING_URI``, ``DSAGT_PROJECT_DIR``, ``EMBEDDING_*``)
        must be explicit in the JSON env block.
        """
        del env, pdir
        actions: list[str] = []
        mcp_path = working_dir / ".roo" / "mcp.json"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_env = _mcp_env_block(config)
        mcp_config = _build_mcp_servers_dict(mcp_env)
        mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
        actions.append(f"Wrote {mcp_path} ({len(mcp_env)} env vars)")
        return actions

    def launch_oneliner(self, project: str, project_dir: Path) -> str:
        """Roo loads customModes from ``.roomodes``; without
        ``--mode dsagt`` it runs in the default ``code`` mode and the
        DSAgt customInstructions are dropped from the system prompt.
        """
        del project
        import shlex

        pdir = shlex.quote(str(project_dir))
        return f"cd {pdir} && roo --mode dsagt"

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Punted in BYOA Phase 1 — see module docstring.

        Roo CLI v0.1.17 has no way to reach a self-hosted gateway (no
        ``--base-url``, no honored env var, ``bedrock`` rejected at the
        provider validator).  Phase 2 ``dsagt-proxy`` provides a
        localhost endpoint roo's anthropic SDK happily talks to.
        """
        del config, env, working_dir, script_path, max_turns
        raise RuntimeError(
            "roo batch mode is punted in Phase 1 BYOA — roo CLI v0.1.17 "
            "has no flag, env var, or non-cloud auth path that points it "
            "at a self-hosted gateway.  Phase 2 reactivates this via "
            "dsagt-proxy.  See agents/roo.py module docstring."
        )

    def proxy_run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Phase-2 proxy-mode batch run.  Un-punts roo: routes through
        ``dsagt-proxy`` at ``ANTHROPIC_BASE_URL`` (set by
        ``proxy_env_overrides`` in the base class), so roo's anthropic
        SDK posts to localhost.  The proxy aliases roo's hardcoded
        ``claude-sonnet-4-5`` rewrite back to the upstream's served
        name (see ``_AGENT_PRIMARY_ALIASES`` in proxy_server.py).

        Sentinel API key flows through ``--api-key`` so any direct
        bypass 401s loudly.  Model is the upstream-served name from
        ``config["llm"]["model"]`` — roo will rewrite it to its
        whitelist default before sending, but the proxy aliases that
        rewrite back to the configured upstream.
        """
        del max_turns
        from .base import _PROXY_FORWARDED_SENTINEL

        model = (config.get("llm") or {}).get("model")
        if not model:
            raise RuntimeError("roo proxy_run_script requires config['llm']['model'].")
        cmd = [
            "roo",
            "--print",
            "--oneshot",
            "--mode",
            "dsagt",
            "--prompt-file",
            str(script_path),
            "--workspace",
            str(working_dir),
            "--debug",
            "--provider",
            "anthropic",
            "--api-key",
            _PROXY_FORWARDED_SENTINEL,
            "--model",
            model,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)

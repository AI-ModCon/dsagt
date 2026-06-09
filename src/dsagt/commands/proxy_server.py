"""
dsagt-proxy: opt-in LiteLLM forwarding proxy for agent transparency.

DSAgt's normal observability path is the agent emitting OTel directly to
MLflow's ``/v1/traces`` (Claude Code, Goose).  For agents that don't
emit OTel with full LLM-call payloads (Cline, Roo Code, Codex
partially), running with ``dsagt start --enable-proxy`` makes the
agent's actions visible at all — every LLM request the agent issues
becomes an MLflow trace you can inspect in real time and replay later:
which model, which messages, which tool_use blocks the assistant emits,
which tool_results came back, full token + cache stats.  Without the
proxy, the same agent runs as a black box from DSAgt's perspective —
``dsagt info`` shows only embedding + tool-execute spans, MLflow shows
no agent turns, and end-of-session memory extraction has no
conversation to read.

Real-time transparency is the primary value.  Memory extraction works
as a downstream consequence because the data it needs (request +
response payloads tagged with the session id) lands in MLflow exactly
because the proxy autologged it.

Architecture: ``init_proxy_tracing()`` installs MLflow's native tracer
provider as the OTel global and plants ``_DSAGTMlflowLogger`` (a
MlflowLogger subclass) into LiteLLM's logger cache.  The subclass
stamps ``mlflow.trace.session``, ``dsagt.source=agent``, and
``dsagt.agent`` on every proxy-captured trace in the narrow window
between trace creation and export — giving rich
``mlflow.spanInputs``/``mlflow.spanOutputs`` traces with full request
and response payloads.

Routing: requests come in on the local port and LiteLLM forwards them
to the user's configured upstream (LLM + embedding) using a minimal
two-route ``model_list`` config.  No multi-provider abstraction beyond
what LiteLLM already provides.

Activation: ``dsagt start --enable-proxy`` sets ``config["proxy"]`` and
``start_services`` spawns this command on a kernel-picked free port.
``agents/__init__.py`` sees the proxy port and overrides the agent's
``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL`` to point at it, plus
plants a sentinel API key so any direct call bypassing the proxy 401s
loudly instead of silently leaking.  Without ``--enable-proxy``, this
command is never started — agents talk to their providers directly and
their visibility depends on whether they emit OTel themselves.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

logger = logging.getLogger(__name__)


# Some agents rewrite the requested model name into one of their hardcoded
# "known" Anthropic IDs before sending to /v1/messages — they don't
# recognize lab-gateway-aliased names like
# ``claude-haiku-4-5-20251001-v1-project`` and silently substitute the
# agent's current default.  Without aliasing, those primary-reasoning
# calls fall through to the sidechannel wildcard (mock) and the agent
# gets MODEL_NO_ASSISTANT_MESSAGES.
#   - roo (v0.1.x): rewrites to ``claude-sonnet-4-5``
#   - cline (1.x):  rewrites to ``claude-sonnet-4-5-20250929``
# Each alias forwards to the configured upstream primary.  Grow this list
# when new agents/versions surface their own defaults.
_AGENT_PRIMARY_ALIASES = (
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
)

# Agent-specific request fields that some upstreams reject.  LiteLLM's
# global ``drop_params: true`` only drops fields it recognizes as
# "supported by some providers, not this one"; unknown fields pass
# through.  ``additional_drop_params`` must be set per-model
# (``litellm_params`` level), not globally — verified empirically.
#   - ``client_metadata``: Codex sends this; Bedrock Anthropic Messages
#     adapter rejects it ("Extra inputs are not permitted").
_DROP_PARAMS_YAML = '      additional_drop_params: ["client_metadata"]\n'


def _generate_config(
    llm_model: str,
    llm_base_url: str,
    llm_provider: str,
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_provider: str | None = None,
) -> str:
    """Render the LiteLLM proxy YAML and return the path to a tempfile.

    Layout:
      * Primary route forwards the configured llm.model to the upstream.
      * One alias route per ``_AGENT_PRIMARY_ALIASES`` entry, also
        forwarded to the upstream primary — so cline/roo's hardcoded
        model rewrites still reach the right model.
      * Embedding route (only when ``embedding_*`` args provided) forwards
        the configured embedding model.  In ``local`` embedding mode the
        knowledge MCP server uses sentence-transformers in-process and
        bypasses the proxy entirely, so the embedding route is skipped.
      * Sidechannel wildcard catches everything else (agent title-gen,
        session-namer, etc.) and returns a canned mock response.
        ``observability.SIDECHANNEL_WILDCARD_ROUTE_YAML`` provides this.
    """
    from dsagt.observability import SIDECHANNEL_WILDCARD_ROUTE_YAML

    aliases_yaml = "".join(f"""  - model_name: {alias}
    litellm_params:
      model: {llm_provider}/{llm_model}
      api_base: {llm_base_url}
      api_key: os.environ/LLM_API_KEY
{_DROP_PARAMS_YAML}""" for alias in _AGENT_PRIMARY_ALIASES)

    embedding_yaml = ""
    if embedding_model and embedding_base_url and embedding_provider:
        embedding_yaml = (
            f"  - model_name: {embedding_model}\n"
            f"    litellm_params:\n"
            f"      model: {embedding_provider}/{embedding_model}\n"
            f"      api_base: {embedding_base_url}\n"
            f"      api_key: os.environ/EMBEDDING_API_KEY\n"
        )

    body = f"""\
model_list:
  - model_name: {llm_model}
    litellm_params:
      model: {llm_provider}/{llm_model}
      api_base: {llm_base_url}
      api_key: os.environ/LLM_API_KEY
{_DROP_PARAMS_YAML}{aliases_yaml}{embedding_yaml}{SIDECHANNEL_WILDCARD_ROUTE_YAML}\
litellm_settings:
  drop_params: true
  num_retries: 5
  request_timeout: 300
"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="dsagt_litellm_",
        delete=False,
    )
    tmp.write(body)
    tmp.close()
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser(prog="dsagt-proxy")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--mlflow-url",
        required=True,
        help="MLflow server URL the OTLP exporter ships traces to.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="DSAGT project name (= MLflow experiment name).",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="DSAGT session id stamped on every trace via OTel resource attrs.",
    )
    parser.add_argument(
        "--records-dir",
        required=True,
        help="Project's trace_archive/ — sidechannel.jsonl lands adjacent.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--provider", required=True)
    # Embedding args optional — only needed when project's embedding
    # backend is ``api``.  In ``local`` mode the knowledge MCP server
    # uses sentence-transformers in-process and never routes through
    # the proxy, so we skip the embedding route entirely.
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [dsagt-proxy] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("LLM_API_KEY"):
        logger.error("LLM_API_KEY not set; the proxy needs it to forward LLM requests.")
        sys.exit(1)
    embedding_routing = bool(
        args.embedding_model and args.embedding_base_url and args.embedding_provider
    )
    if embedding_routing and not os.environ.get("EMBEDDING_API_KEY"):
        logger.error(
            "EMBEDDING_API_KEY not set; the proxy needs it to forward embedding requests."
        )
        sys.exit(1)

    # init_proxy_tracing installs MLflow's tracer provider as the OTel
    # global, plants ``_DSAGTMlflowLogger`` (a MlflowLogger subclass) into
    # LiteLLM's logger cache to stamp ``mlflow.trace.session`` /
    # ``dsagt.source=agent`` / ``dsagt.agent`` on every proxy-captured
    # trace, and registers DSAGTCallback for cache-breakpoint injection +
    # sidechannel-call detection.
    from dsagt.observability import init_proxy_tracing

    init_proxy_tracing(
        mlflow_url=args.mlflow_url,
        project=args.project,
        session_id=args.session,
        records_dir=args.records_dir,
    )

    # Claude Code sends native Anthropic-format requests (POST /v1/messages).
    # Default LiteLLM behavior translates those to /responses, which most
    # project gateways don't expose.  Force the /chat/completions path so
    # any openai-compatible upstream works.
    import litellm

    litellm.use_chat_completions_url_for_anthropic_messages = True

    # Tell the DSAGT callback what the configured primary model is, so its
    # sidechannel detector can distinguish "real upstream call" from
    # "wildcard catchall hit".  See observability.record_sidechannel_call.
    os.environ["DSAGT_PRIMARY_MODEL"] = args.model

    config_path = _generate_config(
        args.model,
        args.base_url,
        args.provider,
        args.embedding_model,
        args.embedding_base_url,
        args.embedding_provider,
    )
    logger.info("Generated LiteLLM config at %s", config_path)
    logger.info("Starting LiteLLM proxy on %s:%d", args.host, args.port)

    # run_server is a Click command.  standalone_mode=False makes Click
    # raise on errors instead of sys.exit; we still catch SystemExit
    # because Click can raise it on clean shutdown.
    from litellm.proxy.proxy_cli import run_server

    try:
        run_server.main(
            args=[
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--config",
                config_path,
            ],
            standalone_mode=False,
        )
    except SystemExit:
        pass


if __name__ == "__main__":
    main()

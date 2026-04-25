"""
dsagt-proxy: Start LiteLLM proxy with MLflow tracing and DSAgt tool records.

Usage:
    dsagt-proxy
    dsagt-proxy --port 4000 --records-dir runtime/trace_archive
    dsagt-proxy --mlflow-url http://localhost:5001
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PORT = 4000
DEFAULT_RECORDS_DIR = "runtime/trace_archive"


def _generate_config(model: str, base_url: str, provider: str) -> str:
    """Generate a LiteLLM proxy config YAML.

    Routes the configured model through the chosen LiteLLM provider pointed
    at *base_url*.  ``provider`` is the LiteLLM provider prefix (e.g.
    ``openai``, ``anthropic``, ``bedrock``) and selects request-format and
    auth handling.  LiteLLM normalizes incoming Anthropic- and OpenAI-format
    requests, so both Claude Code and Goose work against the same config.

    Callbacks (DSAGTCallback for provenance, MlflowLogger for LLM traces)
    are registered in Python *before* ``run_server`` rather than via YAML
    because ``run_server``'s CLI path only calls ``ProxyConfig.get_config``
    (which just parses the YAML) — not ``load_config`` (which would apply
    ``litellm_settings.success_callback``).  Direct registration sidesteps
    that gap and works regardless of LiteLLM's config-loading internals.

    The wildcard fallback that catches agent sidechannel calls comes from
    ``dsagt.observability`` (sidechannel section) — see that module for why
    and how.
    """
    from dsagt.observability import SIDECHANNEL_WILDCARD_ROUTE_YAML

    return f"""\
model_list:
  - model_name: {model}
    litellm_params:
      model: {provider}/{model}
      api_base: {base_url}
      api_key: os.environ/LLM_API_KEY
{SIDECHANNEL_WILDCARD_ROUTE_YAML}\
litellm_settings:
  drop_params: true
"""


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="dsagt-proxy",
        description="Start LiteLLM proxy with MLflow tracing and DSAGT tool records.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--records-dir", default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--session", default=None)
    parser.add_argument("--config", default=None, help="Path to existing LiteLLM config YAML")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--base-url", required=True,
        help="Upstream LLM endpoint (from dsagt_config.yaml llm.base_url)")
    parser.add_argument("--provider", required=True,
        help="LiteLLM provider prefix, e.g. openai, anthropic, bedrock. "
             "See https://docs.litellm.ai/docs/providers for the full list.")
    parser.add_argument("--mlflow-url", default=None,
        help="MLflow tracking URL (enables LiteLLM → MLflow trace autologging)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [dsagt-proxy] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("LLM_API_KEY"):
        logger.error("LLM_API_KEY not set. The proxy needs it to forward requests.")
        sys.exit(1)

    import litellm

    # Claude Code sends native Anthropic-format requests (POST /v1/messages).
    # By default LiteLLM translates those to the OpenAI Responses API
    # (/responses) for openai-compatible upstreams, but most project
    # gateways (e.g. PNNL's ai-incubator-api) only expose /chat/completions.
    # This flag opts out of the Responses-API path so Anthropic requests get
    # translated to /chat/completions, which every openai-compatible gateway
    # supports.  Harmless for goose (which already talks /chat/completions).
    litellm.use_chat_completions_url_for_anthropic_messages = True

    from dsagt.provenance import create_callback

    callback = create_callback(
        records_dir=args.records_dir,
        session_id=args.session,
    )
    litellm.callbacks = [callback]

    records_path = Path(args.records_dir).resolve()
    logger.info("DSAGT callback registered → tool records at %s", records_path)

    if args.mlflow_url:
        import mlflow
        from dsagt.provenance import install_mlflow_logger_with_session_tag

        mlflow.set_tracking_uri(args.mlflow_url)
        # Project name is already the store boundary (each project has its own
        # mlflow/mlflow.db); the experiment is just a container inside that
        # store, so a single stable name keeps all sessions comparable.
        mlflow.set_experiment(os.environ.get("DSAGT_PROJECT", "dsagt"))

        # Pre-seed LiteLLM's logger cache with our MlflowLogger subclass so
        # the string "mlflow" in success_callback resolves to it (subclass
        # passes the isinstance check LiteLLM uses to dedupe loggers).  This
        # is what gets `mlflow.trace.session` stamped onto LLM-completion
        # traces — see install_mlflow_logger_with_session_tag's docstring.
        install_mlflow_logger_with_session_tag()

        # Register the built-in "mlflow" callback by NAME, not instance.
        # LiteLLM's async dispatch (what the proxy uses) resolves string names
        # to logger classes via an internal registry at each call; passing an
        # MlflowLogger *instance* skips that resolution step and the callback
        # gets missed on the async path.
        if "mlflow" not in (litellm.success_callback or []):
            litellm.success_callback = (litellm.success_callback or []) + ["mlflow"]
        if "mlflow" not in (litellm.failure_callback or []):
            litellm.failure_callback = (litellm.failure_callback or []) + ["mlflow"]

        logger.info("MLflow tracing enabled → %s (experiment=%s)",
                    args.mlflow_url, os.environ.get("DSAGT_PROJECT", "dsagt"))
    else:
        logger.info("MLflow tracing disabled (use --mlflow-url to enable)")

    if args.config:
        config_path = args.config
    else:
        config_content = _generate_config(args.model, args.base_url, args.provider)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="dsagt_litellm_", delete=False,
        )
        tmp.write(config_content)
        tmp.close()
        config_path = tmp.name
        logger.info("Generated LiteLLM config at %s", config_path)

    logger.info("Starting LiteLLM proxy on port %d", args.port)

    # Invoke litellm's run_server in-process so the DSAgt callback we
    # registered via litellm.callbacks above is actually attached when
    # requests come in.  There is no subprocess fallback: if this fails,
    # the DSAgt tool-record + observability path is broken and the user
    # needs to know.  start_services() probes the port and will surface
    # the failure to dsagt start with a clean error message.
    #
    # run_server is a Click command, so we invoke it via
    # .main(args=..., standalone_mode=False) — calling it positionally
    # would either error (40+ params) or, in older versions, raise
    # TypeError because Click commands aren't callable with kwargs the
    # same way functions are.  standalone_mode=False makes Click raise
    # on errors instead of calling sys.exit().
    from litellm.proxy.proxy_cli import run_server
    try:
        run_server.main(
            args=[
                "--host", "0.0.0.0",
                "--port", str(args.port),
                "--config", config_path,
            ],
            standalone_mode=False,
        )
    except SystemExit:
        # Click in standalone_mode=False can still raise SystemExit when
        # the underlying server exits cleanly.  Treat that as success;
        # any other exception propagates and crashes dsagt-proxy, which
        # is what start_services()'s readiness probe expects.
        pass


if __name__ == "__main__":
    main()

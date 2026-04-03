"""
dsagt-proxy: Start LiteLLM proxy with OTel tracing and DSAGT tool records.

Configures two callback paths:
  1. OTel (LiteLLM built-in) → spans, metrics, token/cost data → MLflow
  2. DSAGT callback → tool execution records (intent + report layers)

Usage:
    dsagt-proxy
    dsagt-proxy --port 4000 --records-dir runtime/trace_archive
    dsagt-proxy --config my_litellm_config.yaml
    dsagt-proxy --otel-endpoint http://localhost:5000  # MLflow

Agent configuration:
    Claude Code:  export ANTHROPIC_BASE_URL="http://localhost:4000"
    Goose:        export OPENAI_HOST="http://localhost:4000"
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
DEFAULT_OTEL_ENDPOINT = "http://localhost:5000"


def _generate_config(model: str, otel_endpoint: str | None = None) -> str:
    """Generate a LiteLLM proxy config YAML.

    When an OTel endpoint is provided, enables the built-in OTel callback
    for standard trace export (to MLflow, Jaeger, etc.).
    """
    callbacks = []
    env_vars = ""

    if otel_endpoint:
        callbacks.append("otel")
        env_vars = f"""
environment_variables:
  OTEL_EXPORTER_OTLP_ENDPOINT: "{otel_endpoint}"
  OTEL_EXPORTER_OTLP_PROTOCOL: "http/protobuf"
  OTEL_SERVICE_NAME: "dsagt-proxy"
"""

    callbacks_line = f"  callbacks: {callbacks}" if callbacks else ""

    return f"""\
model_list:
  - model_name: {model}
    litellm_params:
      model: anthropic/{model}
      api_key: os.environ/LLM_API_KEY

litellm_settings:
  drop_params: true
{callbacks_line}
{env_vars}"""


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="dsagt-proxy",
        description="Start LiteLLM proxy with OTel tracing and DSAGT tool records.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Listen port (default: {DEFAULT_PORT})")
    parser.add_argument("--records-dir", default=DEFAULT_RECORDS_DIR, help=f"Tool execution records directory (default: {DEFAULT_RECORDS_DIR})")
    parser.add_argument("--session", default=None, help="Session ID (default: $DSAGT_SESSION_ID)")
    parser.add_argument("--config", default=None, help="Path to existing LiteLLM config YAML (skips auto-generation)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name for auto-generated config")
    parser.add_argument("--otel-endpoint", default=None, help=f"OTLP endpoint for trace export, e.g. MLflow (default: none, set to enable)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [dsagt-proxy] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Check for API key early
    if not os.environ.get("LLM_API_KEY"):
        logger.error("LLM_API_KEY not set. The proxy needs it to forward requests.")
        sys.exit(1)

    # Register our callback before LiteLLM starts
    try:
        import litellm
    except ImportError:
        logger.error(
            "litellm is not installed. Install it with:\n"
            "  uv pip install 'litellm[proxy]'"
        )
        sys.exit(1)

    from dsagt.proxy_callback import create_callback

    callback = create_callback(
        records_dir=args.records_dir,
        session_id=args.session,
    )
    litellm.callbacks = [callback]

    records_path = Path(args.records_dir).resolve()
    logger.info("DSAGT callback registered → tool records at %s", records_path)
    if args.otel_endpoint:
        logger.info("OTel export enabled → %s", args.otel_endpoint)
    else:
        logger.info("OTel export disabled (use --otel-endpoint to enable)")

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_content = _generate_config(args.model, args.otel_endpoint)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="dsagt_litellm_", delete=False,
        )
        tmp.write(config_content)
        tmp.close()
        config_path = tmp.name
        logger.info("Generated LiteLLM config at %s", config_path)

    # Start the proxy
    logger.info("Starting LiteLLM proxy on port %d", args.port)
    logger.info("Claude Code:  export ANTHROPIC_BASE_URL=\"http://localhost:%d\"", args.port)
    logger.info("Goose:        export OPENAI_HOST=\"http://localhost:%d\"", args.port)

    try:
        from litellm.proxy.proxy_cli import run_server
        run_server(
            host="0.0.0.0",
            port=args.port,
            config=config_path,
        )
    except (ImportError, TypeError):
        # Fallback: run via subprocess if the internal API changed
        import subprocess
        cmd = [
            sys.executable, "-m", "litellm",
            "--config", config_path,
            "--port", str(args.port),
        ]
        logger.info("Fallback: running %s", " ".join(cmd))
        # Note: subprocess won't inherit our litellm.callbacks registration,
        # so this path only works if litellm loads the callback from config.
        logger.warning(
            "Subprocess fallback cannot pass the callback in-process. "
            "Add the callback to your LiteLLM config YAML instead."
        )
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

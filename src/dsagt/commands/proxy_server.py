"""
dsagt-proxy: Start LiteLLM proxy with OTel tracing and DSAgt tool records.

Usage:
    dsagt-proxy
    dsagt-proxy --port 4000 --records-dir runtime/trace_archive
    dsagt-proxy --otel-endpoint http://localhost:5000
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


def _generate_config(model: str, otel_endpoint: str | None = None) -> str:
    """Generate a LiteLLM proxy config YAML."""
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
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--records-dir", default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--session", default=None)
    parser.add_argument("--config", default=None, help="Path to existing LiteLLM config YAML")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--otel-endpoint", default=None)
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

    from dsagt.provenance import create_callback

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

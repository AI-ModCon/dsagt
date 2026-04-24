"""
Contract tests for dsagt-proxy.

Spawns the real proxy subprocess against a LiteLLM ``mock_response`` upstream
(no network, no credentials), sends requests in both wire shapes the proxy
has to accept, then queries the MLflow sqlite store to verify the
observability invariants the smoke test relies on:

- **Dispatch parity** — every proxy request produces at least one
  ``litellm-*`` trace.  If LiteLLM's async callback ever drops a trace,
  this test fails where smoke would pass (smoke tolerates extras from
  memory extraction; contract counts incrementally).
- **Session / source / agent metadata** — traces carry
  ``mlflow.trace.session``, ``dsagt.source=agent``, ``dsagt.agent=<platform>``.
  These are what the MLflow UI's filters key on, and they're stamped by the
  ``_DSAGTMlflowLogger`` subclass — a subtle install that'd silently regress
  if LiteLLM's logger-cache conventions change.
- **Wire-shape translation** — ``/v1/messages`` (Anthropic-native, sent by
  claude-code/roo/cline) reaches the same mock_response as
  ``/chat/completions`` (OpenAI-native, sent by goose).  Validates
  ``use_chat_completions_url_for_anthropic_messages`` is in force.

Runs without credentials — mock_response short-circuits upstream at the
LiteLLM layer, so ``http://localhost:19999`` is a black hole that's never
dialed.  ~6s for the full file (5s proxy boot + ~1s of requests + metadata
query); shared via a module-scoped fixture.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SESSION_ID = "contract-test-session-001"
AGENT_NAME = "claude-code"
MODEL_NAME = "contract-test-model"
PROJECT_NAME = "contract-test"
MOCK_RESPONSE = "Contract-test canned reply"


def _free_port() -> int:
    """Bind 0 to grab a free port; release before the proxy binds it.

    Small race window, but parallel tests don't run in the same tmp dir
    and this is a module-scoped fixture, so collisions are vanishingly rare.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_ready(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    """Poll /health/readiness until the proxy answers, or fail fast.

    Bails early if the subprocess already exited — otherwise we'd wait the
    full timeout for a server that's never coming back.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(
                f"Proxy exited prematurely (rc={proc.returncode})\n{stderr}"
            )
        try:
            r = httpx.get(f"http://localhost:{port}/health/readiness", timeout=1.0)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Proxy on port {port} did not become ready in {timeout}s")


@pytest.fixture(scope="module")
def proxy(tmp_path_factory):
    """Spawn dsagt-proxy against a mock_response upstream; yield a handle.

    Module-scoped because boot is the dominant cost (~5s).  Each test makes
    its own requests and counts new MLflow traces incrementally so test
    order doesn't matter.
    """
    tmp = tmp_path_factory.mktemp("proxy_contract")
    port = _free_port()
    mlflow_db = tmp / "mlflow.db"
    records_dir = tmp / "trace_archive"
    records_dir.mkdir()

    # Build the config via the real _generate_config so the contract test
    # validates what production spawns (primary route + wildcard mock), not
    # a hand-rolled YAML that could drift.  Override the primary route's
    # mock_response so the primary path is testable without a real upstream.
    from dsagt.commands.proxy_server import _generate_config
    config_body = _generate_config(MODEL_NAME, "http://localhost:19999")
    # Inject mock_response into the primary entry.  The wildcard mock is
    # already present; primary needs the inline mock to stay network-free.
    primary_marker = f"      api_key: os.environ/LLM_API_KEY\n"
    config_body = config_body.replace(
        primary_marker,
        primary_marker + f'      mock_response: "{MOCK_RESPONSE}"\n',
        1,
    )
    config_path = tmp / "litellm_config.yaml"
    config_path.write_text(config_body)

    env = {
        **os.environ,
        "LLM_API_KEY": "test-upstream-key",
        "DSAGT_PROJECT": PROJECT_NAME,
        "DSAGT_SESSION_ID": SESSION_ID,
        "DSAGT_AGENT": AGENT_NAME,
        "DSAGT_PRIMARY_MODEL": MODEL_NAME,
    }

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "dsagt.commands.proxy_server",
            "--port", str(port),
            "--model", MODEL_NAME,
            "--base-url", "http://localhost:19999",
            "--config", str(config_path),
            "--mlflow-url", f"sqlite:///{mlflow_db}",
            "--session", SESSION_ID,
            "--records-dir", str(records_dir),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_ready(port, proc)
        yield SimpleNamespace(
            port=port,
            mlflow_db=mlflow_db,
            records_dir=records_dir,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def _count_litellm_traces(mlflow_db: Path) -> int:
    """Count traces with at least one ``litellm-*`` span in their span list.

    Same filter smoke_test uses — the name "litellm-acompletion" (and
    variants) is LiteLLM's signal that a completion fired, separate from
    our own ``tool.execute`` / ``kb.*`` spans.
    """
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    exp = mlflow.get_experiment_by_name(PROJECT_NAME)
    if exp is None:
        return 0
    traces = mlflow.search_traces(
        locations=[exp.experiment_id], max_results=500,
    )
    if traces.empty:
        return 0
    return int(
        traces["spans"]
        .apply(lambda spans: any(s["name"].startswith("litellm-") for s in spans))
        .sum()
    )


def _latest_trace_metadata(mlflow_db: Path) -> dict:
    """Return the most recent trace's metadata dict.

    MLflow's ``search_traces`` orders by start_time DESC by default, so the
    first row is newest.
    """
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    exp = mlflow.get_experiment_by_name(PROJECT_NAME)
    assert exp is not None, "contract-test experiment missing"
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id], max_results=1,
    )
    assert not traces.empty, "no traces yet"
    return dict(traces.iloc[0]["trace_metadata"])


def _post_openai(port: int) -> dict:
    r = httpx.post(
        f"http://localhost:{port}/v1/chat/completions",
        headers={"Authorization": "Bearer any-bearer"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def _post_anthropic(port: int) -> dict:
    r = httpx.post(
        f"http://localhost:{port}/v1/messages",
        headers={
            "x-api-key": "any-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Wire-shape tests
# ---------------------------------------------------------------------------

def test_openai_chat_completions(proxy):
    """OpenAI-shape request returns the mock_response in choices[0].message."""
    data = _post_openai(proxy.port)
    assert data["choices"][0]["message"]["content"] == MOCK_RESPONSE


def test_anthropic_messages_translation(proxy):
    """Anthropic-shape request gets translated to /chat/completions upstream.

    Verifies ``litellm.use_chat_completions_url_for_anthropic_messages`` is
    honored: without it, LiteLLM would route to /responses and 404 against
    the mock (and against most lab gateways).
    """
    data = _post_anthropic(proxy.port)
    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    assert texts == [MOCK_RESPONSE], f"unexpected content shape: {data}"


# ---------------------------------------------------------------------------
# Observability invariants
# ---------------------------------------------------------------------------

def test_dispatch_parity(proxy):
    """Every proxy request produces at least one litellm-* trace.

    LiteLLM's async callback dispatch has historically been the fragile
    bit: register an instance instead of a string name and traces get
    silently dropped.  This is the cheapest possible assertion that the
    wire survives.
    """
    before = _count_litellm_traces(proxy.mlflow_db)
    for _ in range(3):
        _post_openai(proxy.port)
    # Callbacks fire asynchronously; give them a moment to land.
    # 2s is well over observed latency for mock_response on localhost.
    time.sleep(2.0)
    after = _count_litellm_traces(proxy.mlflow_db)
    new = after - before
    assert new >= 3, f"expected 3 new traces, saw {new} (before={before}, after={after})"


def test_trace_metadata_session_and_source(proxy):
    """_DSAGTMlflowLogger stamps session + source + agent on every trace.

    Issues one request, then reads the latest trace's metadata.  These are
    the three keys the MLflow UI filters on.
    """
    _post_openai(proxy.port)
    time.sleep(1.5)
    md = _latest_trace_metadata(proxy.mlflow_db)
    assert md.get("mlflow.trace.session") == SESSION_ID
    assert md.get("dsagt.source") == "agent"
    assert md.get("dsagt.agent") == AGENT_NAME


def test_wildcard_catches_unknown_model(proxy):
    """A request for a model the proxy has never heard of must NOT 400.

    The wildcard route catches it and returns the canned mock_response.
    This is the generalization that replaces the per-agent mock list: we
    don't maintain a growing registry of "known hardcoded sidechannel
    models" (goose's gpt-4o-mini, claude-code's haiku, ...); any unknown
    name gets mocked.
    """
    r = httpx.post(
        f"http://localhost:{proxy.port}/v1/chat/completions",
        headers={"Authorization": "Bearer any"},
        json={
            "model": "never-heard-of-this-model-7b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
        timeout=10.0,
    )
    # Non-400 is the primary invariant.  The wildcard's canned response is
    # "session" (see _generate_config); asserting on it keeps exact-match
    # priority honest — the primary route's MOCK_RESPONSE must NOT win for
    # a model name the proxy hasn't been told about.
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    assert content != MOCK_RESPONSE, "primary route incorrectly matched unknown model"
    assert content == "session", f"wildcard canned reply changed: got {content!r}"

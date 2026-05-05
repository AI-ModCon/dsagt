"""
DSAgt CLI — project initialization and session management.

Two flows:

1. **BYOA** (default): ``dsagt init --agent <name>`` writes per-agent
   MCP config artifacts and prints the env vars + one-liner the user
   needs to launch their own agent.  No ``dsagt start``.  ``dsagt mlflow``
   brings up tracing in a separate terminal; ``dsagt memory`` extracts
   episodic memory from accumulated traces.
2. **Proxy mode** (Phase 2, not yet wired): ``dsagt init --agent <name>
   --proxy_traces`` opts into the dsagt-managed proxy that intercepts
   the agent's LLM calls for trace capture.  ``dsagt start`` then
   launches the agent + proxy under dsagt supervision.

Usage:
    dsagt init <project> --agent <platform> [--mlflow-port <n>] [--location <path>]
    dsagt mlflow <project>
    dsagt memory --project <project>
    dsagt start <project> [--agent <platform>] [--mlflow-port <n>]
    dsagt info <project> [--json]
    dsagt stop <project>
    dsagt smoke-test
    dsagt setup-kb [--collection <name>] [--embedding-* flags]
    dsagt list
    dsagt mv <project> <location>
    dsagt rm <project> [-y] [--keep-files]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsagt.agents import (
    agent_env,
    dynamic_agent_record,
    launch_agent,
    static_agent_files_present,
    static_agent_record,
    AGENTS,
)
from dsagt.session import (
    VALID_AGENTS,
    list_projects,
    load_config,
    init_project,
    mlflow_command,
    move_project,
    persist_agent_choice,
    pick_free_port,
    project_dir,
    remove_project,
    run_extraction,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)


def _cmd_init(args):
    """Create a new BYOA project.

    Writes the agent's static files (instructions, state dirs) and the
    runtime artifacts (MCP config: ``.mcp.json`` for claude, ``goose.yaml``
    for goose, etc.) populated with the MLflow port pinned at init time.
    Then prints the env-var block + launch one-liner the user needs to
    run their own agent.
    """
    location = Path(args.location).resolve() if args.location else None
    pdir, mlflow_port = init_project(
        args.project, args.agent,
        mlflow_port=args.mlflow_port,
        location=location,
    )
    print(f"Project created: {pdir}")
    print(f"  Agent:        {args.agent}")
    print(f"  MLflow port:  {mlflow_port}")
    print()

    config = load_config(args.project)
    for action in static_agent_record(config, args.agent, pdir):
        print(f"  {action}")
    # Pass the user's shell env so per-agent ``write_dynamic`` can read
    # provider creds (e.g., cline.write_dynamic invokes ``cline auth``
    # with ANTHROPIC_API_KEY / ANTHROPIC_MODEL from the shell).
    for action in dynamic_agent_record(config, env=dict(os.environ), working_dir=pdir):
        print(f"  {action}")

    setup = AGENTS[args.agent]()
    mlflow_url = f"http://localhost:{mlflow_port}"

    print()
    print("Native-OTel trace export (claude / goose emit OTel; cline / roo")
    print("don't, codex partially does — set these only if you want the")
    print(f"agent's traces in this project's MLflow at {mlflow_url}):")
    print()
    print(f"  export MLFLOW_TRACKING_URI={mlflow_url}")
    print(f"  export OTEL_EXPORTER_OTLP_ENDPOINT={mlflow_url}/v1/traces")
    print(f"  export OTEL_RESOURCE_ATTRIBUTES=service.name={args.agent}")
    print()
    if setup.telemetry_env:
        print(f"Agent-specific telemetry verbosity for {args.agent} — set these")
        print("if you want full LLM-call payloads (messages, tool_use blocks,")
        print("response bodies) in MLflow.  Skip if you already have your own")
        print("telemetry setup configured:")
        print()
        for k, v in sorted(setup.telemetry_env.items()):
            print(f"  export {k}={v}")
        print()

    cred_hints = setup.byoa_env_hints(mlflow_port, args.project, pdir)
    if cred_hints:
        print("Provider credentials (set in your shell; skip any your agent is")
        print("already configured to handle for plain chat/coding):")
        print()
        for var, hint in cred_hints:
            print(f"  export {var}=...   # {hint}")
        print()

    print(f"In one terminal, start MLflow:")
    print(f"  dsagt mlflow {args.project}")
    print()
    print(f"In another terminal, launch your agent:")
    print(f"  {setup.launch_oneliner(args.project, pdir)}")
    vscode_lines = setup.vscode_hint(pdir)
    if vscode_lines:
        print()
        print("Or with the VS Code extension:")
        for line in vscode_lines:
            print(f"  {line}")
    print()
    print(f"After your session, extract memory:")
    print(f"  dsagt memory --project {args.project}")


def _cmd_start(args):
    """Start a project session: resolve agent → pick ports → start services →
    write agent configs → launch.

    Order matters: services start before the agent's runtime configs are
    written so the actually-bound ports flow into MCP env blocks.
    Static files (instructions) are written here only if missing — when
    the user ran ``dsagt init --agent X``, the static record was already
    written at init time and we don't touch it.
    """
    config = load_config(args.project)
    pdir = Path(config["project_dir"])

    # Step 1: resolve agent.  CLI overrides YAML.  If neither is set,
    # fail with a one-line message naming both options.
    yaml_agent = config.get("agent")
    agent = args.agent or yaml_agent
    if not agent:
        raise ValueError(
            "No agent specified.  Pass --agent <platform> or set 'agent:' in "
            f"{pdir / 'dsagt_config.yaml'}."
        )
    if agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")
    config["agent"] = agent

    # Step 2: persist agent into YAML if this is the project's first
    # encounter with one.  CLI overrides on later runs are per-run only
    # — they don't touch the YAML default.
    if not yaml_agent:
        persist_agent_choice(args.project, agent)

    # Step 3: write the static record on demand — when init didn't run
    # with --agent, when the user switched agents, or when a marker
    # file was deleted.  Idempotent: skips when already present.
    if not static_agent_files_present(agent, pdir):
        for action in static_agent_record(config, agent, pdir):
            print(f"  {action}")

    # Step 4: synthesize session id (one per start, threaded through every
    # subprocess via DSAGT_SESSION_ID so the MLflow UI can filter to one
    # session).
    config["session_id"] = (
        f"{config['project']}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    # Step 5: start MLflow (and optionally dsagt-proxy).  Picks free ports
    # automatically and writes them to <project>/.runtime; CLI overrides
    # are honored if present.  EVERYTHING from here on must be inside the
    # try/finally — otherwise an exception between service start and
    # ``launch_agent`` leaks subprocesses we just spawned.
    if args.mlflow_port:
        config.setdefault("mlflow", {})["port"] = args.mlflow_port
    if args.enable_proxy:
        # Marker that triggers proxy subprocess in start_services + URL
        # injection in agent_env.  Port is picked by start_services via
        # pick_free_port — same path as MLflow.
        config.setdefault("proxy", {})
    ports = start_services(config)
    if "proxy" in ports:
        print(f"  Ports: mlflow={ports['mlflow']}, proxy={ports['proxy']}")
    else:
        print(f"  Ports: mlflow={ports['mlflow']}")

    try:
        # Step 7: build env, write dynamic agent configs (with the actual
        # MLflow port baked in), launch.
        env = agent_env(config)
        for action in dynamic_agent_record(config, env, pdir):
            print(f"  {action}")

        print()
        print(f"  Project:  {config['project']}")
        print(f"  Agent:    {config['agent']}")
        print(f"  Dir:      {pdir}")
        print(f"  MLflow:   http://localhost:{config['mlflow']['port']}")
        print()

        return launch_agent(
            config, env, pdir,
            script_path=args.script,
            max_turns=args.max_turns,
        )
    finally:
        print()
        # Extraction is best-effort — don't let its failures keep us from
        # cleaning up services.
        try:
            result = run_extraction(args.project)
            n_indexed = result.get("tool_use_indexed", 0)
            if n_indexed:
                print(f"  Indexed {n_indexed} tool execution(s) into tool_use")
            if result.get("status") == "ok":
                print(f"  Extracted {result['total_entries']} memories from session")
            elif result.get("status") == "empty":
                print("  No session exchanges to extract")
            # ``tool_use_only`` is the BYOA default (no DSAGT_MEMORY_*
            # configured); the indexed-count line above already covered it.
        except Exception as e:
            print(f"  WARNING: extraction failed: {e}")

        # Phase 2 proxy mode: surface any sidechannel-call hits from this
        # session so the user can spot a typo in their primary llm.model
        # vs. a harmless agent title-gen / session-namer call.  No-op
        # when the proxy didn't run or no hits were logged.
        try:
            from dsagt.observability import print_sidechannel_warning
            print_sidechannel_warning(pdir, config.get("session_id"))
        except Exception as e:
            logger.debug("sidechannel warning failed: %s", e)

        _stop_one(args.project)


def _cmd_list(args):
    """List all registered projects with their status."""
    projects = list_projects()
    if not projects:
        print("  No projects registered. Run 'dsagt init <name> --agent <platform>' to create one.")
        return

    for name, path in projects.items():
        pdir = Path(path)
        config_path = pdir / "dsagt_config.yaml"

        # Best-effort: if the config is readable, show agent + service status.
        # If the project dir is gone or config is broken, just show the path.
        agent = ""
        status = ""
        if config_path.exists():
            try:
                config = load_config(name)
                agent = config.get("agent", "")
            except (FileNotFoundError, ValueError):
                pass

        runtime_path = pdir / ".runtime"
        if runtime_path.exists():
            state = json.loads(runtime_path.read_text())
            ports = state.get("ports", {})
            status = f"running (mlflow:{ports.get('mlflow','?')})"
        else:
            status = "stopped"

        print(f"  {name:<20} {agent:<14} {status:<40} {path}")


def _cmd_mv(args):
    """Move a project to a new location."""
    location = Path(args.location).resolve()
    new_path = move_project(args.project, location)
    print(f"  Moved {args.project} → {new_path}")


def _cmd_rm(args):
    """Unregister a project and (by default) delete its directory."""
    pdir = project_dir(args.project)

    if args.keep_files:
        action = f"Unregister '{args.project}' (keep files at {pdir})"
    else:
        action = f"Delete project '{args.project}' at {pdir} and unregister it"

    if not args.yes:
        resp = input(f"{action}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("  Cancelled.")
            return 0

    remove_project(args.project, keep_files=args.keep_files)
    if args.keep_files:
        print(f"  Unregistered {args.project} (files kept at {pdir})")
    else:
        print(f"  Removed {args.project}")


def _cmd_setup_kb(args):
    """Build the core knowledge base collections."""
    from dsagt.commands.setup_core_kb import run_setup_kb
    run_setup_kb(args)


def _cmd_mlflow(args):
    """Run MLflow in the foreground.

    Pins the port from the project's internal config so MCP servers
    (which bake the URL into their artifacts at init time) agree on
    where traces land.  Reaps any prior MLflow we left behind on this
    project before binding so a stale leftover doesn't block the new
    one.  If the port is still busy after reap, surfaces the offender
    via ``lsof`` so the user knows what to kill.
    """
    config = load_config(args.project)
    pdir = Path(config["project_dir"])

    # Reap any prior dsagt-spawned MLflow on this project so it doesn't
    # hold the pinned port.  Same machinery dsagt start uses.
    from dsagt.session import reap_runtime
    reaped = reap_runtime(pdir / ".runtime")
    for msg in reaped:
        print(f"  {msg}")

    port = config.get("mlflow", {}).get("port")
    if port is None:
        port = pick_free_port()

    busy = _port_holder(port)
    if busy:
        print(
            f"Error: port {port} is busy. Holder:\n  {busy}\n"
            f"Stop that process or re-init with `dsagt init <new-project> "
            f"--mlflow-port <other>`.",
            file=sys.stderr,
        )
        return 1

    cmd = mlflow_command(pdir, config.get("mlflow", {}), port=port)

    proc = subprocess.Popen(cmd, start_new_session=True)
    runtime_file = pdir / ".runtime"
    runtime_file.write_text(json.dumps({
        "pids": {"mlflow": proc.pid},
        "ports": {"mlflow": port},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    # Best-effort: poll briefly so MLflow is ready before the agent
    # starts (the launch shim curls MLflow at exec time to resolve the
    # experiment-id header — better that resolution succeeds rather
    # than dropping the header).
    experiment_id = _wait_and_resolve_experiment(port, args.project, timeout=20.0)

    print(f"  MLflow UI: http://localhost:{port}")
    print(f"  Project:   {args.project}")
    print(f"  Dir:       {pdir / 'mlflow'}")
    if experiment_id is not None:
        print(f"  Experiment id: {experiment_id}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0
    finally:
        runtime_file.unlink(missing_ok=True)


def _port_holder(port: int) -> str | None:
    """Return a one-line description of the process holding *port*, or None.

    Uses ``lsof`` (macOS + Linux).  Returns None when the port is free
    or when ``lsof`` isn't available (in which case the bind attempt
    will surface the failure anyway).
    """
    try:
        result = subprocess.run(
            ["lsof", "-iTCP", f"-sTCP:LISTEN", "-P", "-n", f"-i:{port}"],
            capture_output=True, text=True, check=False, timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    lines = [l for l in result.stdout.splitlines() if l and not l.startswith("COMMAND")]
    return lines[0] if lines else None


def _wait_and_resolve_experiment(
    port: int, project: str, timeout: float,
) -> str | None:
    """Poll MLflow until it answers, then look up / create the experiment.

    Returns the numeric experiment id on success, None on timeout or
    error (caller still prints the URL — the user can find the id in
    the UI).
    """
    import socket
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.25)
    else:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(f"http://localhost:{port}")
        return str(mlflow.set_experiment(project).experiment_id)
    except Exception as e:
        logger.debug("could not resolve experiment id: %s", e)
        return None


def _stop_one(project: str) -> int:
    """Stop services for a single project via the .runtime state file.

    Returns the number of services killed (0 when nothing was running),
    so callers can report "nothing to do" cleanly.  Idempotent — safe
    to call when no project state file exists.
    """
    try:
        load_config(project)  # validates the project is registered + parsable
    except (FileNotFoundError, ValueError) as e:
        # Registered project whose config is missing or malformed — skip
        # rather than abort a multi-project sweep.
        print(f"  [{project}] skipping: {e}")
        return 0

    msgs = stop_services(project)
    for msg in msgs:
        print(f"  [{project}] {msg}")
    if not msgs:
        print(f"  [{project}] no running services.")
    return len(msgs)


def _cmd_stop(args):
    """Stop running services.

    Without a project argument, sweeps every registered project so the
    common "I just want to clean up whatever's running" case doesn't
    require remembering which project name was active.  With a project,
    behaves as before — single-target.

    Output rules: every action gets a line, prefixed with the project
    name; when nothing happened anywhere we still print a final summary.
    """
    if args.project:
        _stop_one(args.project)
        return

    projects = list_projects()
    if not projects:
        print("  No projects registered.")
        return

    total = 0
    for name in sorted(projects):
        total += _stop_one(name)

    if total == 0:
        print(f"  Swept {len(projects)} project(s); nothing to stop.")


def _cmd_info(args):
    """Triage summary of a project's MLflow traces."""
    from dsagt.commands.info import run
    return run(args.project, as_json=args.json)


def _cmd_memory(args):
    """Extract episodic memory from accumulated session traces.

    Tracks a high-water-mark timestamp in
    ``<project>/.dsagt/extracted_at.json``.  Each invocation extracts
    traces newer than the mark and updates it.  In BYOA mode (no
    ``DSAGT_SESSION_ID`` minted by ``dsagt start``), session boundaries
    are fuzzy — we batch all unprocessed traces into one extraction.
    """
    config = load_config(args.project)
    pdir = Path(config["project_dir"])
    state_path = pdir / ".dsagt" / "extracted_at.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    last_extracted = None
    if state_path.exists():
        last_extracted = json.loads(state_path.read_text()).get("last_extracted_at")
        print(f"  Last extraction watermark: {last_extracted}")
    else:
        print("  No prior extraction recorded — processing all available traces.")

    now = datetime.now(timezone.utc).isoformat()
    result = run_extraction(args.project)
    status = result.get("status", "unknown")
    n_indexed = result.get("tool_use_indexed", 0)
    if n_indexed:
        print(f"  Indexed {n_indexed} tool execution(s) into tool_use")
    if status == "ok":
        print(f"  Extracted {result.get('total_entries', 0)} memories")
        state_path.write_text(json.dumps({
            "last_extracted_at": now,
            "previous": last_extracted,
        }, indent=2) + "\n")
    elif status == "empty":
        print("  No new traces to extract")
    elif status == "tool_use_only":
        print(
            "  LLM-based memory extraction skipped: set "
            "DSAGT_MEMORY_API_KEY and DSAGT_MEMORY_MODEL in your shell "
            "to enable.  Optional: DSAGT_MEMORY_BASE_URL, "
            "DSAGT_MEMORY_PROVIDER."
        )
    else:
        print(f"  Extraction returned status={status}: {result}")
    return 0


def _cmd_smoke_test(args):
    """Run the end-to-end smoke test (non-interactive, with assertions).

    Thin wrapper around ``tests/smoke_test/run.sh`` so the script stays the
    source of truth — bash is the right shape for orchestrating processes
    and assertion checks.  CLI exposure is just for ergonomics.

    With ``--all``, run the harness in parallel for every agent in
    ``VALID_AGENTS``.  Each agent has its own project name (``smoke-test-X``)
    so they don't collide on MLflow ports, kb_index, or registry entries.
    Output is per-agent log files; the summary prints in finish order.
    """
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    script = pkg_dir / "tests" / "smoke_test" / "run.sh"
    if not script.exists():
        print(f"Error: smoke test script not found at {script}", file=sys.stderr)
        return 1

    if args.all:
        return _run_smoke_all(script)

    agent = args.agent or "goose"
    return subprocess.run(["bash", str(script), agent]).returncode


def _run_smoke_all(script: Path) -> int:
    """Run the smoke harness for every VALID_AGENTS entry in parallel.

    Streams each agent's stdout/stderr to a per-agent log file so the
    terminal stays readable.  Prints the verdict for each agent as it
    finishes — fastest-first, not input order — so the operator can see
    progress instead of waiting for the slowest agent before any output.
    """
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    agents = list(VALID_AGENTS)
    log_dir = Path(tempfile.mkdtemp(prefix="dsagt-smoke-"))

    print(f"[smoke-all] launching {len(agents)} parallel runs", flush=True)
    print(f"[smoke-all] log dir: {log_dir}", flush=True)
    for a in agents:
        print(f"[smoke-all]   {a:7}  → {log_dir / f'smoke-{a}.log'}", flush=True)

    def _run_one(agent: str) -> tuple[str, int, float]:
        start = time.monotonic()
        with (log_dir / f"smoke-{agent}.log").open("w") as fh:
            rc = subprocess.run(
                ["bash", str(script), agent],
                stdout=fh, stderr=subprocess.STDOUT,
            ).returncode
        return agent, rc, time.monotonic() - start

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        futs = {ex.submit(_run_one, a): a for a in agents}
        for fut in as_completed(futs):
            agent, rc, elapsed = fut.result()
            verdict = "PASS" if rc == 0 else "FAIL"
            print(
                f"[smoke-all] {verdict}  {agent:7}  {elapsed:5.0f}s  "
                f"(rc={rc}, log={log_dir / f'smoke-{agent}.log'})",
                flush=True,
            )
            results[agent] = rc

    n_pass = sum(1 for rc in results.values() if rc == 0)
    print(
        f"[smoke-all] {n_pass}/{len(agents)} passed", flush=True,
    )
    return 0 if n_pass == len(agents) else 1


# User-facing exception types that should produce a clean one-line error
# message rather than a traceback.  Everything else crashes loudly.
_USER_ERRORS = (FileNotFoundError, FileExistsError, ValueError, RuntimeError)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dsagt", description="DSAgt project and session management.")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create a new project")
    p_init.add_argument("project", help="Project name (human-readable alias)")
    p_init.add_argument("--agent", choices=VALID_AGENTS, required=True,
        help="Agent platform (required)")
    p_init.add_argument("--mlflow-port", type=int, default=None,
        help="MLflow port to pin (default: pick a free one).  Written to "
             "the internal config so MCP servers + dsagt mlflow agree on it.")
    p_init.add_argument("--location", default=None,
        help="Parent directory for the project (default: ~/dsagt-projects/)")

    p_start = sub.add_parser("start", help="Start a project session")
    p_start.add_argument("project", help="Project name")
    p_start.add_argument("--agent", choices=VALID_AGENTS, default=None,
        help="Agent platform.  Required on first start if init didn't set one; "
             "thereafter, a per-run override (doesn't update the YAML default).")
    p_start.add_argument("--mlflow-port", type=int, default=None,
        help="Override the MLflow port from dsagt_config.yaml.  Useful when "
             "the configured port is permanently taken on your machine.")
    p_start.add_argument("--enable-proxy", action="store_true",
        help="Spawn dsagt-proxy and route the agent's LLM calls through it. "
             "For agents that don't natively emit OTel traces with full "
             "LLM-call payloads (cline, roo, codex partial), the proxy is "
             "what makes their conversations visible in MLflow at all — "
             "every agent turn becomes an inspectable trace (real-time "
             "audit, replay, debugging) and memory extraction works as a "
             "downstream consequence.  Agents with "
             "otel_payload_support='full' (claude, goose) emit their own "
             "traces and don't need the flag.  Port is auto-picked.")
    p_start.add_argument("--script", default=None,
        help="Path to a goose-run instructions file. When set, the agent runs "
             "non-interactively (GOOSE_MODE=auto) against this script — used by "
             "the smoke test to share the full dsagt start lifecycle (config "
             "generation, services, memory extraction, cleanup) with manual runs.")
    p_start.add_argument("--max-turns", type=int, default=30,
        help="Cap on agent turn count when --script is set (default: 30).")

    p_mlflow = sub.add_parser("mlflow", help="Run MLflow in the foreground against a project's store")
    p_mlflow.add_argument("project", help="Project name")
    p_mlflow.add_argument("--port", type=int, default=None,
        help="Override the port from dsagt_config.yaml")

    p_info = sub.add_parser("info",
        help="Summarize a project's MLflow traces (tokens, errors, by session/source)")
    p_info.add_argument("project", help="Project name")
    p_info.add_argument("--json", action="store_true",
        help="Emit the structured report as JSON instead of formatted text")

    p_memory = sub.add_parser("memory",
        help="Extract episodic memory from accumulated session traces "
             "(BYOA: run after one or more agent sessions to populate the KB)")
    p_memory.add_argument("--project", required=True, help="Project name")

    p_stop = sub.add_parser("stop",
        help="Stop project services (including orphans on configured ports). "
             "Without a project argument, sweeps every registered project.")
    p_stop.add_argument("project", nargs="?", default=None,
        help="Project name (omit to sweep all registered projects)")

    p_smoke = sub.add_parser("smoke-test",
        help="Run the end-to-end smoke test (sources DSAGT/.env, drives the agent non-interactively, asserts artifacts)")
    smoke_group = p_smoke.add_mutually_exclusive_group()
    smoke_group.add_argument("--agent", choices=list(VALID_AGENTS), default=None,
        help="Which agent to drive (default: goose).")
    smoke_group.add_argument("--all", action="store_true",
        help="Run the smoke harness in parallel for every agent.  Per-agent "
             "logs go to a temp dir; verdicts print in finish order.")

    p_setup_kb = sub.add_parser("setup-kb", help="Build the core knowledge base collections")
    from dsagt.commands.setup_core_kb import add_setup_kb_args
    add_setup_kb_args(p_setup_kb)

    sub.add_parser("list", help="List all registered projects and their status")

    p_mv = sub.add_parser("mv", help="Move a project to a new location")
    p_mv.add_argument("project", help="Project name")
    p_mv.add_argument("location", help="New parent directory")

    p_rm = sub.add_parser("rm", help="Unregister a project and delete its directory")
    p_rm.add_argument("project", help="Project name")
    p_rm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_rm.add_argument("--keep-files", action="store_true",
        help="Unregister only; leave the project directory on disk")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [dsagt] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        return 1

    cmds = {
        "init": _cmd_init,
        "start": _cmd_start,
        "mlflow": _cmd_mlflow,
        "memory": _cmd_memory,
        "info": _cmd_info,
        "stop": _cmd_stop,
        "smoke-test": _cmd_smoke_test,
        "setup-kb": _cmd_setup_kb,
        "list": _cmd_list,
        "mv": _cmd_mv,
        "rm": _cmd_rm,
    }
    try:
        return cmds[args.command](args)
    except _USER_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

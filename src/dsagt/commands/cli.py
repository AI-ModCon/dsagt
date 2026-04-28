"""
DSAgt CLI — project initialization and session management.

Usage:
    dsagt init <project> [--agent <platform>] [--location <path>]
    dsagt start <project> [--agent <platform>] [--proxy-port <n>] [--mlflow-port <n>]
    dsagt mlflow <project> [--port <n>]
    dsagt info <project> [--json]
    dsagt stop <project>
    dsagt smoke-test
    dsagt setup-kb [--collection <name>] [--embedding-* flags]
    dsagt list
    dsagt mv <project> <location>
    dsagt rm <project> [-y] [--keep-files]

The ``agent`` choice may be made at init OR at first start — the flag is
required exactly once across the project's lifetime.  After that it's
recorded in ``dsagt_config.yaml`` and ``dsagt start`` runs without it.
Passing ``--agent`` on a later start is a per-run override and doesn't
update the YAML default.
"""

import argparse
import json
import logging
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
)
from dsagt.session import (
    VALID_AGENTS,
    kill_processes_on_port,
    list_projects,
    load_config,
    init_project,
    mlflow_command,
    move_project,
    persist_agent_choice,
    pick_free_port,
    port_in_use,
    project_dir,
    remove_project,
    run_extraction,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)


def _cmd_init(args):
    """Create a new project.  ``--agent`` is optional here.

    Without ``--agent``: writes the agent-agnostic project (YAML, kb_index/,
    mlflow/, skills/, trace_archive/) and the YAML omits the ``agent:``
    field.  ``dsagt start --agent X`` will be required on first start.

    With ``--agent X``: also writes the agent's static files (instructions,
    state directories) so the user can edit them before first start.
    """
    location = Path(args.location).resolve() if args.location else None
    pdir = init_project(args.project, args.agent, location=location)
    print(f"Project created: {pdir}")
    if args.agent:
        print(f"  Agent: {args.agent} (recorded in dsagt_config.yaml)")
        print(f"Edit {pdir / 'dsagt_config.yaml'} or the agent instructions file, "
              f"then run: dsagt start {args.project}")
    else:
        print(f"Edit {pdir / 'dsagt_config.yaml'}, then run: "
              f"dsagt start {args.project} --agent <platform>")


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

    # Step 5: pick ports.  CLI overrides take precedence over YAML.
    # ``pick_free_port`` falls back to the next free port if the
    # preferred one is taken, with a one-line warning so users don't
    # silently accumulate orphan services.
    proxy_pref = args.proxy_port if args.proxy_port else config["proxy"]["port"]
    mlflow_pref = args.mlflow_port if args.mlflow_port else config["mlflow"]["port"]
    config["proxy"]["port"], proxy_warn = pick_free_port(proxy_pref)
    config["mlflow"]["port"], mlflow_warn = pick_free_port(mlflow_pref)
    if proxy_warn:
        print(f"  WARNING: {proxy_warn}")
    if mlflow_warn:
        print(f"  WARNING: {mlflow_warn}")
    print(
        f"  Ports: proxy={config['proxy']['port']}, "
        f"mlflow={config['mlflow']['port']}"
    )

    # Step 6: start services.  EVERYTHING from here on must be inside the
    # try/finally — otherwise an exception between service start and
    # ``launch_agent`` (e.g. cline auth subprocess failure inside
    # ``dynamic_agent_record``) leaks the proxy + MLflow we just spawned.
    # That was the root cause of "ports occupied after a clean smoke
    # test": a previous run for a *different* agent crashed in step 7
    # before ``launch_agent``, the finally never ran, and the orphans
    # survived into the next smoke run for a new agent that succeeded
    # but ran on fallback ports while the original ports were still
    # held.
    start_services(config)

    try:
        # Step 7: build env, write dynamic agent configs (with the actual
        # ports baked in), launch.
        env = agent_env(config)
        for action in dynamic_agent_record(config, env, pdir):
            print(f"  {action}")

        print()
        print(f"  Project:  {config['project']}")
        print(f"  Agent:    {config['agent']}")
        print(f"  Dir:      {pdir}")
        print(f"  Proxy:    http://localhost:{config['proxy']['port']}")
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
        # cleaning up services.  A wrapped try/except around the body would
        # also work; explicit guard here is more readable.
        try:
            result = run_extraction(args.project)
            if result.get("status") == "ok":
                print(f"  Extracted {result['total_entries']} memories from session")
            elif result.get("status") == "empty":
                print("  No session exchanges to extract")
            elif result.get("status") == "skipped":
                print(f"  Extraction skipped: {result.get('reason', 'unknown')}")
        except Exception as e:
            print(f"  WARNING: extraction failed: {e}")

        # Full per-project sweep: SIGTERM via stop_services (by pid), then
        # port-based sweep for orphans, then a final port_in_use check.
        # Pass the *actually-bound* ports from the in-memory config —
        # pick_free_port may have fallen back from the YAML default,
        # and the YAML still says the original port.  Cleaning the YAML
        # port would leave the real orphan (on the fallback port) alive.
        _stop_one(
            args.project,
            proxy_port=config["proxy"]["port"],
            mlflow_port=config["mlflow"]["port"],
        )

        try:
            from dsagt.observability import print_sidechannel_warning as _sidechannel_warning
            _sidechannel_warning(pdir, config.get("session_id"))
        except Exception as e:
            print(f"  WARNING: sidechannel summary failed: {e}")


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

        pid_path = pdir / ".pids"
        if pid_path.exists():
            pids = json.loads(pid_path.read_text())
            status = f"{len(pids)} service(s) running"
        else:
            status = "stopped"

        print(f"  {name:<20} {agent:<14} {status:<24} {path}")


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
    """Run MLflow in the foreground for post-session trace inspection."""
    config = load_config(args.project)
    pdir = Path(config["project_dir"])
    port = args.port if args.port is not None else config["mlflow"]["port"]

    if port_in_use(port):
        print(f"Error: port {port} is already in use.", file=sys.stderr)
        print(f"  Run 'dsagt stop {args.project}' to clear stale services, "
              f"or pass --port N to use a different port.", file=sys.stderr)
        return 1

    cmd = mlflow_command(pdir, config["mlflow"], port=port)

    print(f"  MLflow UI: http://localhost:{port}")
    print(f"  Project:   {args.project}")
    print(f"  Dir:       {pdir / 'mlflow'}")
    print("  Press Ctrl+C to stop.")
    print()

    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0


def _stop_one(
    project: str,
    *,
    proxy_port: int | None = None,
    mlflow_port: int | None = None,
) -> int:
    """Stop services + sweep ports for a single project.  Returns the
    number of actions taken (so the caller can report "nothing to do"
    when the whole sweep was idle).

    Tries the PID-file path first (fast, specific via ``stop_services``,
    which kills by pid → wait → SIGKILL), then sweeps any processes
    still listening on the project's proxy/mlflow ports — the common
    case after a session exits before its finally-block cleanup could
    run, or after a port-fallback start where the YAML port is stale.

    ``proxy_port`` / ``mlflow_port`` override the YAML values.
    ``_cmd_start`` passes the actually-bound ports here because
    ``pick_free_port`` may have fallen back from the YAML default —
    cleaning the YAML port instead of the actual port would leak the
    real orphan.  External callers (``dsagt stop``) leave both None
    and we read from YAML.

    Final step: verify both ports are actually free.  If anything's
    still listening after we've done everything we can, surface a loud
    message naming what's left so the user can act on it before the
    next ``dsagt start`` races a half-shutdown orphan.
    """
    from dsagt.session import port_held_by_foreign_process, port_in_use

    try:
        config = load_config(project)
    except (FileNotFoundError, ValueError) as e:
        # Registered project whose config is missing or malformed — skip
        # rather than abort the whole sweep so other projects still get
        # cleaned up.
        print(f"  [{project}] skipping: {e}")
        return 0

    actions = 0
    proxy_port = proxy_port if proxy_port is not None else config["proxy"]["port"]
    mlflow_port = mlflow_port if mlflow_port is not None else config["mlflow"]["port"]

    pids_msgs = stop_services(project)
    if pids_msgs != ["No running services found."]:
        for msg in pids_msgs:
            print(f"  [{project}] {msg}")
            actions += 1

    for name, port in (("proxy", proxy_port), ("mlflow", mlflow_port)):
        killed = kill_processes_on_port(port)
        for pid in killed:
            print(f"  [{project}] Killed orphan {name} on port {port} (pid {pid})")
            actions += 1
        if not killed and port_held_by_foreign_process(port):
            print(
                f"  [{project}] Port {port} ({name}) is held by an unrelated process — "
                f"left alone (run lsof -iTCP:{port} to inspect)"
            )
            actions += 1

    # Verify: did the cleanup actually free the ports?  ``port_in_use``'s
    # bind probe catches half-stuck listeners that ``connect_ex`` alone
    # would miss.  Loud message so users notice before the next start
    # picks a fallback port and accumulates more orphans.
    for name, port in (("proxy", proxy_port), ("mlflow", mlflow_port)):
        if port_in_use(port):
            print(
                f"  [{project}] WARNING: port {port} ({name}) is STILL in use after "
                f"cleanup.  Run `lsof -iTCP:{port} -sTCP:LISTEN` to identify the "
                f"holder.  Next `dsagt start` will fall back to a different port."
            )
            actions += 1

    if actions == 0:
        print(
            f"  [{project}] no services or orphans "
            f"(checked proxy:{proxy_port}, mlflow:{mlflow_port})."
        )
    return actions


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


def _cmd_smoke_test(args):
    """Run the end-to-end smoke test (non-interactive, with assertions).

    Thin wrapper around ``tests/smoke_test/run.sh`` so the script stays the
    source of truth — bash is the right shape for orchestrating processes
    and assertion checks.  CLI exposure is just for ergonomics.
    """
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    script = pkg_dir / "tests" / "smoke_test" / "run.sh"
    if not script.exists():
        print(f"Error: smoke test script not found at {script}", file=sys.stderr)
        return 1
    return subprocess.run(["bash", str(script), args.agent]).returncode


# User-facing exception types that should produce a clean one-line error
# message rather than a traceback.  Everything else crashes loudly.
_USER_ERRORS = (FileNotFoundError, FileExistsError, ValueError, RuntimeError)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dsagt", description="DSAgt project and session management.")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create a new project")
    p_init.add_argument("project", help="Project name (human-readable alias)")
    p_init.add_argument("--agent", choices=VALID_AGENTS, default=None,
        help="Agent platform (optional; if omitted, supply --agent at first start)")
    p_init.add_argument("--location", default=None,
        help="Parent directory for the project (default: ~/dsagt-projects/)")

    p_start = sub.add_parser("start", help="Start a project session")
    p_start.add_argument("project", help="Project name")
    p_start.add_argument("--agent", choices=VALID_AGENTS, default=None,
        help="Agent platform.  Required on first start if init didn't set one; "
             "thereafter, a per-run override (doesn't update the YAML default).")
    p_start.add_argument("--proxy-port", type=int, default=None,
        help="Override the proxy port from dsagt_config.yaml.  Useful when the "
             "configured port is permanently taken on your machine.")
    p_start.add_argument("--mlflow-port", type=int, default=None,
        help="Override the MLflow port from dsagt_config.yaml.")
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

    p_stop = sub.add_parser("stop",
        help="Stop project services (including orphans on configured ports). "
             "Without a project argument, sweeps every registered project.")
    p_stop.add_argument("project", nargs="?", default=None,
        help="Project name (omit to sweep all registered projects)")

    p_smoke = sub.add_parser("smoke-test",
        help="Run the end-to-end smoke test (sources DSAGT/.env, drives the agent non-interactively, asserts artifacts)")
    p_smoke.add_argument("--agent", choices=["goose", "claude-code", "cline", "roo", "codex"], default="goose",
        help="Which agent to drive (default: goose).")

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

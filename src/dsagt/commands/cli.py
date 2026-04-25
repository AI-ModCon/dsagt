"""
DSAgt CLI — project initialization and session management.

Usage:
    dsagt init <project> --agent <platform> [--location <path>]
    dsagt start <project>
    dsagt mlflow <project> [--port <n>]
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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsagt.agents import generate_agent_configs, launch_agent
from dsagt.session import (
    VALID_AGENTS,
    kill_processes_on_port,
    list_projects,
    load_config,
    init_project,
    mlflow_command,
    move_project,
    port_in_use,
    project_dir,
    remove_project,
    run_extraction,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)


def _cmd_init(args):
    """Create a new project."""
    location = Path(args.location).resolve() if args.location else None
    pdir = init_project(args.project, args.agent, location=location)
    print(f"Project created: {pdir}")
    print(f"Edit {pdir / 'dsagt_config.yaml'} then run: dsagt start {args.project}")


def _cmd_start(args):
    """Start a project session: services + agent."""
    config = load_config(args.project)
    pdir = Path(config["project_dir"])

    # One session id per `dsagt start` — threaded through every subprocess
    # via DSAGT_SESSION_ID so proxy, MCP servers, and dsagt-run all share it
    # and the MLflow UI can filter one session at a time.
    config["session_id"] = (
        f"{config['project']}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    actions = generate_agent_configs(config, pdir)
    for action in actions:
        print(f"  {action}")

    start_services(config)

    print()
    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Dir:      {pdir}")
    print(f"  Proxy:    http://localhost:{config['proxy']['port']}")
    print(f"  MLflow:   http://localhost:{config['mlflow']['port']}")
    print()

    try:
        return launch_agent(
            config, pdir,
            script_path=args.script,
            max_turns=args.max_turns,
        )
    finally:
        print()
        result = run_extraction(args.project)
        if result.get("status") == "ok":
            print(f"  Extracted {result['total_entries']} memories from session")
        elif result.get("status") == "empty":
            print("  No session exchanges to extract")
        elif result.get("status") == "skipped":
            print(f"  Extraction skipped: {result.get('reason', 'unknown')}")

        for r in stop_services(args.project):
            print(f"  {r}")

        from dsagt.observability import print_sidechannel_warning as _sidechannel_warning
        _sidechannel_warning(pdir, config.get("session_id"))


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


def _cmd_stop(args):
    """Stop running services for a project, including orphans.

    Tries the PID-file path first (fast, specific), then sweeps any
    processes still listening on the project's proxy/mlflow ports — the
    common case after a session exits before its finally-block cleanup
    could run.
    """
    config = load_config(args.project)
    for msg in stop_services(args.project):
        print(f"  {msg}")

    for name, port in (("proxy", config["proxy"]["port"]),
                       ("mlflow", config["mlflow"]["port"])):
        killed = kill_processes_on_port(port)
        for pid in killed:
            print(f"  Killed orphan {name} on port {port} (pid {pid})")


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
    p_init.add_argument("--agent", required=True, choices=VALID_AGENTS, help="Agent platform")
    p_init.add_argument("--location", default=None,
        help="Parent directory for the project (default: ~/dsagt-projects/)")

    p_start = sub.add_parser("start", help="Start a project session")
    p_start.add_argument("project", help="Project name")
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

    p_stop = sub.add_parser("stop", help="Stop project services (including orphans on configured ports)")
    p_stop.add_argument("project", help="Project name")

    p_smoke = sub.add_parser("smoke-test",
        help="Run the end-to-end smoke test (sources DSAGT/.env, drives the agent non-interactively, asserts artifacts)")
    p_smoke.add_argument("--agent", choices=["goose", "claude-code"], default="goose",
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

"""
DSAgt CLI — project initialization and session management.

Usage:
    dsagt init <project> --agent <platform> [--location <path>]
    dsagt start <project>
    dsagt setup-kb [--collection <name>] [--embedding-* flags]
    dsagt list
    dsagt mv <project> <location>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dsagt.agents import generate_agent_configs, launch_agent
from dsagt.session import (
    VALID_AGENTS,
    list_projects,
    load_config,
    init_project,
    move_project,
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
        return launch_agent(config, pdir)
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


def _cmd_setup_kb(args):
    """Build the core knowledge base collections."""
    from dsagt.commands.setup_core_kb import run_setup_kb
    run_setup_kb(args)


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

    p_setup_kb = sub.add_parser("setup-kb", help="Build the core knowledge base collections")
    from dsagt.commands.setup_core_kb import add_setup_kb_args
    add_setup_kb_args(p_setup_kb)

    sub.add_parser("list", help="List all registered projects and their status")

    p_mv = sub.add_parser("mv", help="Move a project to a new location")
    p_mv.add_argument("project", help="Project name")
    p_mv.add_argument("location", help="New parent directory")

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
        "setup-kb": _cmd_setup_kb,
        "list": _cmd_list,
        "mv": _cmd_mv,
    }
    try:
        return cmds[args.command](args)
    except _USER_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

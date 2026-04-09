"""
DSAgt CLI — project initialization and session management.

Usage:
    dsagt init <project> --agent <platform> [--location <path>]
    dsagt start <project>
    dsagt stop <project>
    dsagt extract <project>
    dsagt status <project>
    dsagt list
    dsagt mv <project> <location>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dsagt.agents import agent_command, generate_agent_configs, launch_agent
from dsagt.session import VALID_AGENTS, list_projects, load_config, project_dir
from dsagt.session import (
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
    try:
        pdir = init_project(args.project, args.agent, location=location)
    except FileExistsError as e:
        logger.error(str(e))
        return 1

    print(f"Project created: {pdir}")
    print(f"Edit {pdir / 'dsagt_config.yaml'} then run: dsagt start {args.project}")
    return 0


def _cmd_start(args):
    """Start a project session: services + agent."""
    try:
        config = load_config(args.project)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    pdir = Path(config["project_dir"])

    # Generate agent configs into the project directory
    actions = generate_agent_configs(config, pdir)
    for action in actions:
        print(f"  {action}")

    # Start background services
    start_services(config)

    proxy_port = config["proxy"]["port"]
    mlflow_port = config["mlflow"]["port"]

    print()
    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Dir:      {pdir}")
    print(f"  Proxy:    http://localhost:{proxy_port}")
    print(f"  MLflow:   http://localhost:{mlflow_port}")
    print()

    # Launch the agent from the project directory.
    # Blocks until the agent exits, then cleans up.
    try:
        exit_code = launch_agent(config, pdir)
    finally:
        print()
        try:
            result = run_extraction(args.project)
            if result.get("status") == "ok":
                print(f"  Extracted {result['total_entries']} memories from session")
            elif result.get("status") == "empty":
                print("  No session exchanges to extract")
            elif result.get("status") == "skipped":
                print(f"  Extraction skipped: {result.get('reason', 'unknown')}")
        except Exception as e:
            logger.warning("Extraction failed: %s", e)

        for r in stop_services(args.project):
            print(f"  {r}")

    return exit_code


def _cmd_stop(args):
    """Stop orphaned services for a project.

    Normally dsagt start cleans up automatically when the agent exits.
    Use this only if services were left running (e.g. after a crash).
    """
    results = stop_services(args.project)
    for r in results:
        print(f"  {r}")
    return 0


def _cmd_extract(args):
    """Extract memories from the current session log."""
    try:
        result = run_extraction(args.project)
    except (FileNotFoundError, Exception) as e:
        logger.error("Extraction failed: %s", e)
        return 1

    if result.get("status") == "ok":
        print(f"  Extracted {result['facts']} facts, {result['insights']} insights, {result['summary']} summary")
        print(f"  Total entries stored: {result['total_entries']}")
        print(f"  Session: {result['session_id']}")
    elif result.get("status") == "empty":
        print("  No session exchanges to extract")
    elif result.get("status") == "skipped":
        print(f"  Extraction skipped: {result.get('reason', 'unknown')}")

    return 0


def _cmd_status(args):
    """Show project status."""
    try:
        config = load_config(args.project)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    pdir = Path(config["project_dir"])

    pid_path = pdir / ".pids"
    running = {}
    if pid_path.exists():
        running = json.loads(pid_path.read_text())

    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Dir:      {pdir}")
    print(f"  Services: {len(running)} running" if running else "  Services: stopped")
    for name, pid in running.items():
        print(f"    {name}: pid {pid}")

    return 0


def _cmd_list(args):
    """List all registered projects."""
    projects = list_projects()
    if not projects:
        print("  No projects registered. Run 'dsagt init <name> --agent <platform>' to create one.")
        return 0

    for name, path in projects.items():
        print(f"  {name:<30} {path}")
    return 0


def _cmd_mv(args):
    """Move a project to a new location."""
    location = Path(args.location).resolve()
    try:
        new_path = move_project(args.project, location)
    except (FileNotFoundError, FileExistsError) as e:
        logger.error(str(e))
        return 1

    print(f"  Moved {args.project} → {new_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dsagt", description="DSAgt project and session management.")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Create a new project")
    p_init.add_argument("project", help="Project name (human-readable alias)")
    p_init.add_argument("--agent", required=True, choices=VALID_AGENTS, help="Agent platform")
    p_init.add_argument("--location", default=None,
        help="Parent directory for the project (default: ~/dsagt-projects/)")

    # start
    p_start = sub.add_parser("start", help="Start a project session")
    p_start.add_argument("project", help="Project name")

    # stop
    p_stop = sub.add_parser("stop", help="Stop orphaned services (after a crash)")
    p_stop.add_argument("project", help="Project name")

    # extract
    p_extract = sub.add_parser("extract", help="Extract memories from current session log")
    p_extract.add_argument("project", help="Project name")

    # status
    p_status = sub.add_parser("status", help="Show project status")
    p_status.add_argument("project", help="Project name")

    # list
    sub.add_parser("list", help="List all registered projects")

    # mv
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
        "init": _cmd_init, "start": _cmd_start, "stop": _cmd_stop,
        "extract": _cmd_extract, "status": _cmd_status,
        "list": _cmd_list, "mv": _cmd_mv,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

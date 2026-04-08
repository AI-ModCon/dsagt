"""
DSAGT CLI — project initialization and session management.

Usage:
    dsagt init <project> --agent <platform>
    dsagt start <project> [--working-dir .]
    dsagt stop <project>
    dsagt extract <project>
    dsagt status <project>
"""

import argparse
import logging
import sys
from pathlib import Path

from dsagt.config import VALID_AGENTS, load_config, project_dir
from dsagt.session import (
    generate_agent_configs,
    init_project,
    launch_agent,
    run_extraction,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)


def _cmd_init(args):
    """Create a new project."""
    try:
        pdir = init_project(args.project, args.agent)
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

    working_dir = Path(args.working_dir).resolve()

    # Generate agent configs
    actions = generate_agent_configs(config, working_dir)
    for action in actions:
        print(f"  {action}")

    # Start background services
    start_services(config)

    proxy_port = config["proxy"]["port"]
    mlflow_port = config["mlflow"]["port"]
    pdir = Path(config["project_dir"])

    print()
    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Proxy:    http://localhost:{proxy_port}")
    print(f"  MLflow:   http://localhost:{mlflow_port}")
    print(f"  Records:  {pdir / 'trace_archive'}")
    print()

    # Launch the agent with the correct env
    return launch_agent(config, working_dir)


def _cmd_stop(args):
    """Stop a project session: extract memories, then stop services."""
    # Extract memories before stopping
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

    # Stop services
    results = stop_services(args.project)
    for r in results:
        print(f"  {r}")
    return 0


def _cmd_extract(args):
    """Extract memories from the current session log."""
    pdir = project_dir(args.project)
    if not pdir.exists():
        print(f"Project not found: {pdir}")
        return 1

    try:
        result = run_extraction(args.project)
    except Exception as e:
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
    pdir = project_dir(args.project)

    if not pdir.exists():
        print(f"Project not found: {pdir}")
        return 1

    try:
        config = load_config(args.project)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        return 1

    pid_path = pdir / ".pids"
    running = {}
    if pid_path.exists():
        import json
        running = json.loads(pid_path.read_text())

    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Dir:      {pdir}")
    print(f"  Services: {len(running)} running" if running else "  Services: stopped")
    for name, pid in running.items():
        print(f"    {name}: pid {pid}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dsagt", description="DSAGT project and session management.")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Create a new project")
    p_init.add_argument("project", help="Project name (human-readable alias)")
    p_init.add_argument("--agent", required=True, choices=VALID_AGENTS, help="Agent platform")

    # start
    p_start = sub.add_parser("start", help="Start a project session")
    p_start.add_argument("project", help="Project name")
    p_start.add_argument("--working-dir", default=".", help="Agent working directory (default: current dir)")

    # stop
    p_stop = sub.add_parser("stop", help="Stop a project session")
    p_stop.add_argument("project", help="Project name")

    # extract
    p_extract = sub.add_parser("extract", help="Extract memories from current session log")
    p_extract.add_argument("project", help="Project name")

    # status
    p_status = sub.add_parser("status", help="Show project status")
    p_status.add_argument("project", help="Project name")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [dsagt] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        return 1

    cmds = {"init": _cmd_init, "start": _cmd_start, "stop": _cmd_stop, "extract": _cmd_extract, "status": _cmd_status}
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

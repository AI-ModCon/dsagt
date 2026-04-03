"""
DSAGT CLI — project initialization and session management.

Usage:
    dsagt init <project> --agent <platform>
    dsagt start <project> [--working-dir .]
    dsagt stop <project>
    dsagt status <project>
"""

import argparse
import logging
import sys
from pathlib import Path

from dsagt.config import VALID_AGENTS, load_config, project_dir_for
from dsagt.session import (
    generate_agent_configs,
    init_project,
    launch_agent,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)


def _cmd_init(args):
    """Create a new project."""
    try:
        project_dir = init_project(args.project, args.agent, args.runtime_base)
    except FileExistsError as e:
        logger.error(str(e))
        return 1

    print(f"Project created: {project_dir}")
    print(f"Edit {project_dir / 'dsagt_config.yaml'} then run: dsagt start {args.project}")
    return 0


def _cmd_start(args):
    """Start a project session: services + agent."""
    project_dir = project_dir_for(args.project, args.runtime_base)
    try:
        config = load_config(project_dir)
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

    print()
    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Proxy:    http://localhost:{proxy_port}")
    print(f"  MLflow:   http://localhost:{mlflow_port}")
    print(f"  Records:  {project_dir / 'trace_archive'}")
    print()

    # Launch the agent with the correct env
    return launch_agent(config, working_dir)


def _cmd_stop(args):
    """Stop a project session."""
    project_dir = project_dir_for(args.project, args.runtime_base)
    results = stop_services(project_dir)
    for r in results:
        print(f"  {r}")
    return 0


def _cmd_status(args):
    """Show project status."""
    project_dir = project_dir_for(args.project, args.runtime_base)

    if not project_dir.exists():
        print(f"Project not found: {project_dir}")
        return 1

    try:
        config = load_config(project_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        return 1

    pid_path = project_dir / ".pids"
    running = {}
    if pid_path.exists():
        import json
        running = json.loads(pid_path.read_text())

    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Dir:      {project_dir}")
    print(f"  Services: {len(running)} running" if running else "  Services: stopped")
    for name, pid in running.items():
        print(f"    {name}: pid {pid}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dsagt", description="DSAGT project and session management.")
    parser.add_argument("--runtime-base", default="runtime", help="Base directory for project data (default: runtime)")
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

    cmds = {"init": _cmd_init, "start": _cmd_start, "stop": _cmd_stop, "status": _cmd_status}
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

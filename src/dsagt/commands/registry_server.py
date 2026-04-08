"""
dsagt-registry-server entry point.

Usage:
    dsagt-registry-server
    dsagt-registry-server --runtime-dir ./runtime
"""

import argparse
import asyncio

from dsagt.mcp_utils import run_stdio
from dsagt.registry import ToolRegistry
from dsagt.registry_server import create_registry_server


def main():
    parser = argparse.ArgumentParser(description="DSAGT Registry Builder MCP Server")
    parser.add_argument("--runtime-dir", default="./runtime")
    parser.add_argument("--source-skills-dir", default=None)
    args = parser.parse_args()

    registry = ToolRegistry(
        source_skills_dir=args.source_skills_dir,
        runtime_dir=args.runtime_dir,
    )
    server = create_registry_server(registry)
    asyncio.run(run_stdio(server, "registry"))


if __name__ == "__main__":
    main()

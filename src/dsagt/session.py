"""
DSAGT session management.

Handles project initialization, agent-specific config generation,
and service lifecycle (proxy + MLflow).

Each agent platform gets its configs generated from the single
dsagt_config.yaml — MCP server entries, agent instructions, env vars,
and proxy routing are all derived mechanically.
"""

import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from dsagt.config import (
    VALID_AGENTS,
    default_config_content,
    load_config,
    project_dir as resolve_project_dir,
)
from dsagt.extraction import delete_session_log, extract_session
from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

# Where bundled agent instruction templates live (relative to dsagt package)
_AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"


# ---------------------------------------------------------------------------
# Project initialization
# ---------------------------------------------------------------------------

def init_project(project_name: str, agent: str) -> Path:
    """Create a new project directory with default config and subdirectories.

    Returns the project directory path.
    """
    if agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")

    project_dir = resolve_project_dir(project_name)

    if (project_dir / "dsagt_config.yaml").exists():
        raise FileExistsError(f"Project already exists: {project_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("trace_archive", "mlflow", "skills", "kb_index"):
        (project_dir / subdir).mkdir(exist_ok=True)

    config_content = default_config_content(project_name, agent)
    (project_dir / "dsagt_config.yaml").write_text(config_content)

    return project_dir


# ---------------------------------------------------------------------------
# Agent config generation
# ---------------------------------------------------------------------------

def generate_agent_configs(config: dict, working_dir: str | Path) -> list[str]:
    """Generate agent-platform-specific config files in the working directory.

    Reads the dsagt_config.yaml values and writes the files each agent
    platform expects: MCP server configs, agent instructions, env setup.

    Returns a list of descriptions of what was written.
    """
    agent = config["agent"]
    working_dir = Path(working_dir)
    project_dir = Path(config["project_dir"])
    proxy_port = config["proxy"]["port"]

    generators = {
        "claude-code": _generate_claude_code,
        "goose": _generate_goose,
        "roo": _generate_roo,
        "cline": _generate_cline,
    }

    return generators[agent](config, working_dir, project_dir, proxy_port)


def _mcp_server_args(server: str, project_dir: Path) -> list[str]:
    """Build the args list for an MCP server entry."""
    base = ["run", f"dsagt-{server}-server"]
    if server == "registry":
        base += ["--runtime-dir", str(project_dir)]
    elif server == "knowledge":
        base += [
            "--base-index-dir", str(project_dir / "kb_index"),
            "--runtime-dir", str(project_dir),
        ]
    return base


def _mcp_env_block(config: dict) -> dict:
    """Build the env block for MCP server entries."""
    env = {}
    embedding_key = config.get("embedding", {}).get("api_key", "")
    if embedding_key and not embedding_key.startswith("${"):
        env["LLM_API_KEY"] = embedding_key
    return env


def _generate_claude_code(config, working_dir, project_dir, proxy_port) -> list[str]:
    actions = []
    env_block = _mcp_env_block(config)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server, project_dir),
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry

    mcp_path = working_dir / ".mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path}")

    # Copy agent instructions
    instructions = _load_instructions("claude-code", "dsagt_instructions.md")
    if instructions:
        claude_md = working_dir / "CLAUDE.md"
        if claude_md.exists():
            existing = claude_md.read_text()
            if "DSAGT Pipeline Builder" not in existing:
                claude_md.write_text(existing + "\n\n" + instructions)
                actions.append(f"Appended DSAGT instructions to {claude_md}")
        else:
            claude_md.write_text(instructions)
            actions.append(f"Wrote {claude_md}")

    # Write env helper
    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(project_dir),
    })
    actions.append(f"Wrote {env_path} (source this or export the variables)")

    return actions


def _generate_goose(config, working_dir, project_dir, proxy_port) -> list[str]:
    actions = []
    model = config["llm"]["model"]

    goose_config = {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": model,
        "extensions": {},
    }

    for server in ("registry", "knowledge"):
        args = _mcp_server_args(server, project_dir)
        goose_config["extensions"][server] = {
            "enabled": True,
            "name": server,
            "type": "stdio",
            "cmd": "uv " + " ".join(args),
            "timeout": 300,
        }

    goose_path = working_dir / "goose.yaml"
    goose_path.write_text(yaml.dump(goose_config, default_flow_style=False, sort_keys=False))
    actions.append(f"Wrote {goose_path}")

    # Copy agent instructions
    instructions = _load_instructions("goose", ".goosehints")
    if instructions:
        hints_path = working_dir / ".goosehints"
        hints_path.write_text(instructions)
        actions.append(f"Wrote {hints_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "OPENAI_HOST": f"http://localhost:{proxy_port}",
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(project_dir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


def _generate_roo(config, working_dir, project_dir, proxy_port) -> list[str]:
    actions = []
    env_block = _mcp_env_block(config)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server, project_dir),
            "disabled": False,
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry

    roo_dir = working_dir / ".roo"
    roo_dir.mkdir(exist_ok=True)
    mcp_path = roo_dir / "mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path}")

    # Copy roomodes
    roomodes = _load_instructions("roo", "roomodes")
    if roomodes:
        roomodes_path = working_dir / ".roomodes"
        roomodes_path.write_text(roomodes)
        actions.append(f"Wrote {roomodes_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(project_dir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


def _generate_cline(config, working_dir, project_dir, proxy_port) -> list[str]:
    actions = []
    env_block = _mcp_env_block(config)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server, project_dir),
            "disabled": False,
            "alwaysAllow": [],
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry

    # Write to project dir (user merges into global cline settings)
    mcp_path = working_dir / "cline_mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path} (merge into Cline MCP settings)")

    # Copy agent instructions
    instructions = _load_instructions("cline", "dsagt_instructions.md")
    if instructions:
        rules_dir = working_dir / ".clinerules"
        rules_dir.mkdir(exist_ok=True)
        instr_path = rules_dir / "dsagt_instructions.md"
        instr_path.write_text(instructions)
        actions.append(f"Wrote {instr_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(project_dir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


def _load_instructions(agent_dir: str, filename: str) -> str | None:
    """Load an instruction template from the agents directory."""
    path = _AGENTS_DIR / agent_dir / filename
    if path.exists():
        return path.read_text()
    logger.warning("Instruction template not found: %s", path)
    return None


def _write_env_file(path: Path, env_vars: dict) -> None:
    """Write a sourceable env file."""
    lines = [f'export {k}="{v}"' for k, v in env_vars.items()]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Service start / stop
# ---------------------------------------------------------------------------

def _pid_file(project_dir: Path) -> Path:
    return project_dir / ".pids"


def start_services(config: dict) -> dict[str, int]:
    """Start the proxy and MLflow for a project. Returns {name: pid}."""
    project_dir = Path(config["project_dir"])
    pids = {}

    # Start MLflow
    mlflow_port = config["mlflow"]["port"]
    mlflow_backend = config["mlflow"]["backend"]
    mlflow_dir = project_dir / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)

    if mlflow_backend == "sqlite":
        backend_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
    else:
        backend_uri = str(mlflow_dir)

    mlflow_cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", backend_uri,
        "--default-artifact-root", str(mlflow_dir / "artifacts"),
        "--host", "0.0.0.0",
        "--port", str(mlflow_port),
    ]

    mlflow_log = project_dir / "mlflow.log"
    mlflow_proc = subprocess.Popen(
        mlflow_cmd,
        stdout=open(mlflow_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pids["mlflow"] = mlflow_proc.pid
    logger.info("MLflow started (pid %d) → http://localhost:%d", mlflow_proc.pid, mlflow_port)

    # Start proxy
    proxy_port = config["proxy"]["port"]
    otel_endpoint = f"http://localhost:{mlflow_port}"
    trace_dir = str(project_dir / "trace_archive")

    proxy_cmd = [
        sys.executable, "-m", "dsagt.proxy",
        "--port", str(proxy_port),
        "--records-dir", trace_dir,
        "--session", config["project"],
        "--otel-endpoint", otel_endpoint,
        "--model", config["llm"]["model"],
    ]

    proxy_log = project_dir / "proxy.log"
    proxy_proc = subprocess.Popen(
        proxy_cmd,
        stdout=open(proxy_log, "w"),
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(project_dir),
            "DSAGT_EXTRACTION_THRESHOLD": str(
                config.get("extraction", {}).get("threshold", 0)
            ),
        },
        start_new_session=True,
    )
    pids["proxy"] = proxy_proc.pid
    logger.info("Proxy started (pid %d) → http://localhost:%d", proxy_proc.pid, proxy_port)

    # Save PIDs
    pid_path = _pid_file(project_dir)
    pid_path.write_text(json.dumps(pids, indent=2) + "\n")

    return pids


def stop_services(project_name: str) -> list[str]:
    """Stop running services for a project. Returns descriptions of what was stopped."""
    pid_path = _pid_file(resolve_project_dir(project_name))
    stopped = []

    if not pid_path.exists():
        return ["No running services found."]

    pids = json.loads(pid_path.read_text())

    for name, pid in pids.items():
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped.append(f"Stopped {name} (pid {pid})")
        except (ProcessLookupError, PermissionError):
            stopped.append(f"{name} (pid {pid}) was not running")

    pid_path.unlink(missing_ok=True)
    return stopped


def run_extraction(project_name: str) -> dict:
    """Run memory extraction for a project and clean up the session log.

    Loads the session log, sends one LLM call to extract facts/summary/insights,
    stores results in episodic_memory, then deletes the session log.  If extraction
    fails, the session log is still deleted (transient buffer, MLflow has the truth).

    Returns the extraction result dict, or a status dict on error/empty.
    """
    config = load_config(project_name)
    project_dir = Path(config["project_dir"])
    trace_dir = project_dir / "trace_archive"

    api_key = config.get("llm", {}).get("api_key", "") or os.environ.get("LLM_API_KEY", "")
    model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    session_id = config.get("project", "")
    categories = config.get("categories", {})

    if not api_key or api_key.startswith("${"):
        logger.warning("No API key available for extraction, skipping")
        delete_session_log(trace_dir)
        return {"status": "skipped", "reason": "no_api_key"}

    kb = KnowledgeBase(index_dir=project_dir / "kb_index")
    try:
        return extract_session(
            trace_dir=trace_dir,
            kb=kb,
            api_key=api_key,
            model=model,
            session_id=session_id,
            categories=categories if categories else None,
            runtime_dir=project_dir,
            outlier_sensitivity=float(
                config.get("extraction", {}).get("outlier_sensitivity", 0)
            ),
        )
    finally:
        kb.close()
        # Safety net: ensure session log files are gone even if extraction raised.
        # drain_session_log handles the normal case; this catches edge cases.
        for suffix in (".jsonl", ".consumed"):
            leftover = trace_dir / f"session_log{suffix}"
            if leftover.exists():
                leftover.unlink()


# ---------------------------------------------------------------------------
# Agent launch
# ---------------------------------------------------------------------------

def agent_env(config: dict) -> dict:
    """Build the environment dict an agent process needs to inherit.

    Merges the current environment with DSAGT-specific variables so the
    proxy intercept, project identity, and embedding key are all set.
    """
    project_dir = config["project_dir"]
    proxy_port = config["proxy"]["port"]
    agent = config["agent"]

    env = dict(os.environ)
    env["DSAGT_PROJECT"] = config["project"]
    env["DSAGT_PROJECT_DIR"] = project_dir

    # Proxy routing — agent-specific env var
    if agent in ("claude-code", "roo", "cline"):
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{proxy_port}"
    if agent == "goose":
        env["OPENAI_HOST"] = f"http://localhost:{proxy_port}"

    # Embedding API key for MCP servers
    embedding_key = config.get("embedding", {}).get("api_key", "")
    if embedding_key and not embedding_key.startswith("${"):
        env["LLM_API_KEY"] = embedding_key

    return env


def agent_command(config: dict) -> list[str] | None:
    """Return the shell command to launch the agent, or None for VS Code agents."""
    commands = {
        "claude-code": ["claude"],
        "goose": ["goose", "session"],
    }
    return commands.get(config["agent"])


def launch_agent(config: dict, working_dir: str | Path) -> int:
    """Launch the agent process in the foreground with the correct environment.

    For CLI agents (Claude Code, Goose): runs the agent interactively and
    returns its exit code.

    For VS Code agents (Roo, Cline): prints instructions and returns 0.
    """
    working_dir = Path(working_dir)
    env = agent_env(config)
    cmd = agent_command(config)

    if cmd is None:
        # VS Code agents can't be launched from the CLI
        agent = config["agent"]
        print(f"\n  Open VS Code in: {working_dir}")
        if agent == "roo":
            print("  Switch to the DSAGT Pipeline Builder mode (Cmd+.)")
        elif agent == "cline":
            print("  Open the Cline panel and verify MCP servers are connected")
        print()
        return 0

    logger.info("Launching: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, env=env, cwd=str(working_dir))
        return result.returncode
    except FileNotFoundError:
        logger.error("Command not found: %s", cmd[0])
        logger.error("Is %s installed?", config["agent"])
        return 1
    except KeyboardInterrupt:
        return 0

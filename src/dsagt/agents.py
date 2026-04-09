"""
Agent platform configuration and launch.

Generates platform-specific config files (MCP server entries, agent instructions,
env vars, proxy routing) from the single dsagt_config.yaml. Launches the agent
process in the foreground and blocks until it exits.

Supported platforms:

- **Claude Code** — Install: ``npm i -g @anthropic-ai/claude-code``
  Generates: ``.mcp.json``, ``CLAUDE.md``, ``.dsagt_env``
  Proxy routing: ``ANTHROPIC_BASE_URL``

- **Goose** — Install: see https://github.com/block/goose
  Generates: ``goose.yaml``, ``.goosehints``, ``.dsagt_env``
  Proxy routing: ``OPENAI_HOST``

- **Roo Code** — Install: ``npm i -g @roo-code/cli``
  Generates: ``.roo/mcp.json``, ``.roomodes``, ``.dsagt_env``
  Proxy routing: ``ANTHROPIC_BASE_URL``

- **Cline** — Install: ``npm i -g cline``
  Generates: ``cline_mcp.json``, ``.clinerules/dsagt_instructions.md``, ``.dsagt_env``
  Proxy routing: ``ANTHROPIC_BASE_URL``
"""

import json
import logging
import os
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Master instructions ship with the package
_INSTRUCTIONS_PATH = Path(__file__).parent / "dsagt_instructions.md"

_AGENT_COMMANDS = {
    "claude-code": ["claude"],
    "goose": ["goose", "session"],
    "roo": ["roo"],
    "cline": ["cline"],
}


# ---------------------------------------------------------------------------
# Config generation helpers
# ---------------------------------------------------------------------------

def _embedding_args(config: dict) -> list[str]:
    """Build embedding CLI flags from config. Shared by registry and knowledge servers."""
    args = []
    emb = config.get("embedding", {})
    if emb.get("base_url"):
        args += ["--embedding-base-url", emb["base_url"]]
    if emb.get("model"):
        args += ["--embedding-model", emb["model"]]
    api_key = emb.get("api_key", "")
    if api_key and not api_key.startswith("${"):
        args += ["--embedding-api-key", api_key]
    return args


def _mcp_server_args(server: str, project_dir: Path, config: dict) -> list[str]:
    """Build the args list for an MCP server entry."""
    base = ["run", f"dsagt-{server}-server"]
    if server == "registry":
        base += ["--runtime-dir", str(project_dir)]
        base += _embedding_args(config)
    elif server == "knowledge":
        base += [
            "--base-index-dir", str(project_dir / "kb_index"),
            "--runtime-dir", str(project_dir),
        ]
        base += _embedding_args(config)
    base += _otel_args(config)
    return base


def _otel_args(config: dict) -> list[str]:
    """Build --otel-endpoint and --session-id args for an MCP server."""
    args: list[str] = []
    mlflow_port = config.get("mlflow", {}).get("port")
    if mlflow_port:
        args += ["--otel-endpoint", f"http://localhost:{mlflow_port}"]
    project = config.get("project")
    if project:
        args += ["--session-id", project]
    return args


def _mcp_env_block(config: dict) -> dict:
    """Build the env block for MCP server entries."""
    env = {}
    emb = config.get("embedding", {})
    api_key = emb.get("api_key", "")
    if api_key and not api_key.startswith("${"):
        env["LLM_API_KEY"] = api_key
    base_url = emb.get("base_url", "")
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    model = emb.get("model", "")
    if model:
        env["EMBEDDING_MODEL"] = model
    return env


def _load_master_instructions() -> str | None:
    """Load the master DSAgt instructions."""
    if _INSTRUCTIONS_PATH.exists():
        return _INSTRUCTIONS_PATH.read_text()
    logger.warning("Master instructions not found: %s", _INSTRUCTIONS_PATH)
    return None


def _format_roomodes(instructions: str) -> str:
    """Format master instructions as a Roo Code .roomodes JSON file."""
    parts = instructions.split("\n## ", 1)
    role = parts[0].strip()
    custom = ("## " + parts[1]).strip() if len(parts) > 1 else ""

    return json.dumps({
        "customModes": [{
            "slug": "dsagt",
            "name": "DSAgt Pipeline Builder",
            "roleDefinition": role,
            "customInstructions": custom,
            "groups": ["read", "edit", "browser", "command", "mcp"],
            "source": "project",
        }]
    }, indent=2)


def _write_env_file(path: Path, env_vars: dict) -> None:
    """Write a sourceable env file."""
    lines = [f'export {k}="{v}"' for k, v in env_vars.items()]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Per-platform generators
# ---------------------------------------------------------------------------

def generate_agent_configs(config: dict, working_dir: str | Path) -> list[str]:
    """Generate agent-platform-specific config files in the working directory.

    Returns a list of descriptions of what was written.
    """
    agent = config["agent"]
    working_dir = Path(working_dir)
    pdir = Path(config["project_dir"])
    proxy_port = config["proxy"]["port"]

    generators = {
        "claude-code": _generate_claude_code,
        "goose": _generate_goose,
        "roo": _generate_roo,
        "cline": _generate_cline,
    }

    return generators[agent](config, working_dir, pdir, proxy_port)


def _generate_claude_code(config, working_dir, pdir, proxy_port) -> list[str]:
    actions = []
    env_block = _mcp_env_block(config)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server, pdir, config),
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry

    mcp_path = working_dir / ".mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path}")

    instructions = _load_master_instructions()
    if instructions:
        claude_md = working_dir / "CLAUDE.md"
        if claude_md.exists():
            existing = claude_md.read_text()
            if "DSAgt Pipeline Builder" not in existing:
                claude_md.write_text(existing + "\n\n" + instructions)
                actions.append(f"Appended DSAgt instructions to {claude_md}")
        else:
            claude_md.write_text(instructions)
            actions.append(f"Wrote {claude_md}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


def _generate_goose(config, working_dir, pdir, proxy_port) -> list[str]:
    actions = []
    model = config["llm"]["model"]

    goose_config = {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": model,
        "extensions": {},
    }

    for server in ("registry", "knowledge"):
        args = _mcp_server_args(server, pdir, config)
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

    instructions = _load_master_instructions()
    if instructions:
        hints_path = working_dir / ".goosehints"
        hints_path.write_text(instructions)
        actions.append(f"Wrote {hints_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "OPENAI_HOST": f"http://localhost:{proxy_port}",
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


def _generate_roo(config, working_dir, pdir, proxy_port) -> list[str]:
    actions = []
    env_block = _mcp_env_block(config)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server, pdir, config),
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

    instructions = _load_master_instructions()
    if instructions:
        roomodes_path = working_dir / ".roomodes"
        roomodes_path.write_text(_format_roomodes(instructions))
        actions.append(f"Wrote {roomodes_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


def _generate_cline(config, working_dir, pdir, proxy_port) -> list[str]:
    actions = []
    env_block = _mcp_env_block(config)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server, pdir, config),
            "disabled": False,
            "alwaysAllow": [],
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry

    mcp_path = working_dir / "cline_mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path} (merge into Cline MCP settings)")

    instructions = _load_master_instructions()
    if instructions:
        rules_dir = working_dir / ".clinerules"
        rules_dir.mkdir(exist_ok=True)
        instr_path = rules_dir / "dsagt_instructions.md"
        instr_path.write_text(instructions)
        actions.append(f"Wrote {instr_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")

    return actions


# ---------------------------------------------------------------------------
# Agent environment and launch
# ---------------------------------------------------------------------------

def agent_env(config: dict) -> dict:
    """Build the environment dict an agent process needs to inherit."""
    pdir = config["project_dir"]
    proxy_port = config["proxy"]["port"]
    agent = config["agent"]

    env = dict(os.environ)
    env["DSAGT_PROJECT"] = config["project"]
    env["DSAGT_PROJECT_DIR"] = pdir

    if agent in ("claude-code", "roo", "cline"):
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{proxy_port}"
    if agent == "goose":
        env["OPENAI_HOST"] = f"http://localhost:{proxy_port}"

    emb = config.get("embedding", {})
    embedding_key = emb.get("api_key", "")
    if embedding_key and not embedding_key.startswith("${"):
        env["LLM_API_KEY"] = embedding_key
    if emb.get("base_url"):
        env["OPENAI_BASE_URL"] = emb["base_url"]
    if emb.get("model"):
        env["EMBEDDING_MODEL"] = emb["model"]

    # OTel endpoint inherited by every subprocess (dsagt-run, etc.).
    mlflow_port = config.get("mlflow", {}).get("port")
    if mlflow_port:
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://localhost:{mlflow_port}"
        env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

    return env


def agent_command(config: dict) -> list[str]:
    """Return the shell command to launch the agent."""
    return _AGENT_COMMANDS[config["agent"]]


def launch_agent(config: dict, working_dir: str | Path) -> int:
    """Launch the agent process in the foreground with the correct environment.

    Blocks until the agent exits. Returns the agent's exit code.
    """
    working_dir = Path(working_dir)
    env = agent_env(config)
    cmd = agent_command(config)

    logger.info("Launching: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, env=env, cwd=str(working_dir))
        return result.returncode
    except FileNotFoundError:
        agent = config["agent"]
        logger.error("Command not found: %s", cmd[0])
        logger.error("Install %s first — see src/dsagt/agents/README.md", agent)
        return 1
    except KeyboardInterrupt:
        return 0

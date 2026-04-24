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

def _mcp_server_args(server: str) -> list[str]:
    """Build the args list for an MCP server entry.

    Both servers read all configuration from DSAGT_PROJECT_DIR (env var)
    and dsagt_config.yaml.  No CLI args needed.
    """
    return ["run", f"dsagt-{server}-server"]


def _mcp_env_block(config: dict) -> dict:
    """Build the env block for MCP server entries.

    Servers read all configuration from env vars and dsagt_config.yaml.
    DSAGT_PROJECT_DIR tells them where to find the project directory.
    Embedding credentials flow as env vars rather than CLI flags.
    """
    env = {"DSAGT_PROJECT_DIR": config["project_dir"]}
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
            "args": _mcp_server_args(server),
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
        args = _mcp_server_args(server)
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
            "args": _mcp_server_args(server),
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
            "args": _mcp_server_args(server),
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
    # Proxy's MlflowLogger subclass reads this to stamp dsagt.agent on
    # every agent-turn trace — lets the MLflow UI tell goose/claude-code/
    # roo/cline sessions apart without inspecting the request shape.
    env["DSAGT_AGENT"] = agent

    if agent in ("claude-code", "roo", "cline"):
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{proxy_port}"
        # Pin the model.  Without this Claude Code falls back to its built-in
        # default (currently claude-sonnet-4-6) which won't exist on project
        # gateways like PNNL's ai-incubator-api and the proxy will 400 every
        # request.
        env["ANTHROPIC_MODEL"] = config["llm"]["model"]
    if agent == "goose":
        env["OPENAI_HOST"] = f"http://localhost:{proxy_port}"
        # Override any global ~/.config/goose/config.yaml so the project's
        # configured model is what actually runs.  Without these, a user's
        # global GOOSE_MODEL (e.g. a model their upstream doesn't offer)
        # silently wins over dsagt_config.yaml.
        env["GOOSE_PROVIDER"] = "openai"
        env["GOOSE_MODEL"] = config["llm"]["model"]

    # Plant a sentinel key (not the real one) into the agent's environment
    # under the provider-specific name.  Two things are true simultaneously:
    #
    #   (a) Claude Code / goose silently ignore ANTHROPIC_BASE_URL /
    #       OPENAI_HOST and fall back to OAuth or default endpoints when the
    #       *_API_KEY var is empty, so we have to set *something*.
    #   (b) Setting the real upstream key means that if the proxy is
    #       unreachable (orphan on the port, proxy crashed, misconfigured
    #       base URL), the agent happily falls back to calling the real
    #       upstream directly — silently bypassing our provenance + tracing
    #       pipeline.
    #
    # A sentinel value satisfies (a) without enabling (b): the proxy accepts
    # any bearer token at ingress and forwards with its own LLM_API_KEY, but
    # a direct call to api.anthropic.com / api.openai.com with the sentinel
    # returns 401 — failing loudly at the real boundary instead of silently
    # bypassing us.
    sentinel_key = "dsagt-proxy-forwarded-disable-direct-calls"
    if agent in ("claude-code", "roo", "cline"):
        env["ANTHROPIC_API_KEY"] = sentinel_key
    if agent == "goose":
        env["OPENAI_API_KEY"] = sentinel_key

    emb = config.get("embedding", {})
    embedding_key = emb.get("api_key", "")
    if embedding_key and not embedding_key.startswith("${"):
        env["LLM_API_KEY"] = embedding_key
    if emb.get("base_url"):
        env["OPENAI_BASE_URL"] = emb["base_url"]
    if emb.get("model"):
        env["EMBEDDING_MODEL"] = emb["model"]

    # MLflow tracking URI inherited by every subprocess (dsagt-run, MCP
    # servers) so their mlflow.start_span() calls land in the project's
    # store alongside the proxy's LiteLLM-autologged traces.
    mlflow_port = config.get("mlflow", {}).get("port")
    if mlflow_port:
        env["MLFLOW_TRACKING_URI"] = f"http://localhost:{mlflow_port}"

    # Session id inherited by every subprocess so tool.execute / kb.*
    # spans share the same session tag as the proxy's LLM traces.
    if config.get("session_id"):
        env["DSAGT_SESSION_ID"] = config["session_id"]

    return env


def agent_command(config: dict) -> list[str]:
    """Return the shell command to launch the agent interactively.

    Goose only reads ``~/.config/goose/config.yaml`` for extensions, not a
    project-local file — so the MCP servers are passed via ``--with-extension``
    flags on the session command to guarantee they attach for this project.

    For non-interactive batch (``--script``) mode, see ``launch_agent`` —
    each agent has its own per-prompt invocation pattern that doesn't fit a
    single argv.
    """
    agent = config["agent"]
    cmd = list(_AGENT_COMMANDS[agent])
    if agent == "goose":
        for server in ("registry", "knowledge"):
            cmd.extend(["--with-extension", f"uv run dsagt-{server}-server"])
    return cmd


def launch_agent(
    config: dict,
    working_dir: str | Path,
    script_path: str | Path | None = None,
    max_turns: int = 30,
) -> int:
    """Launch the agent in the foreground with the correct environment.

    Blocks until the agent exits.  Returns the agent's exit code.

    When ``script_path`` is set, the agent runs in non-interactive batch
    mode and we dispatch to a per-agent runner — each agent has a different
    shape for "run these prompts in sequence":

      goose:        single ``goose run --instructions FILE`` call
      claude-code:  loop ``claude -p PROMPT`` then ``claude --continue -p ...``

    Other agents raise — add a runner here when adding agent support.
    """
    working_dir = Path(working_dir)
    env = agent_env(config)

    if script_path is not None:
        agent = config["agent"]
        runner = _SCRIPT_RUNNERS.get(agent)
        if runner is None:
            raise ValueError(
                f"--script is not supported for agent {agent!r}. "
                f"Supported: {sorted(_SCRIPT_RUNNERS)}."
            )
        return runner(config, env, working_dir, Path(script_path), max_turns)

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


# ---------------------------------------------------------------------------
# Per-agent script runners (non-interactive --script mode)
# ---------------------------------------------------------------------------

def _run_goose_script(
    config: dict, env: dict, working_dir: Path, script_path: Path, max_turns: int,
) -> int:
    """Single ``goose run`` call — goose's instructions file IS multi-turn."""
    env["GOOSE_MODE"] = "auto"
    cmd = ["goose", "run", "--instructions", str(script_path),
           "--max-turns", str(max_turns)]
    for server in ("registry", "knowledge"):
        cmd.extend(["--with-extension", f"uv run dsagt-{server}-server"])
    logger.info("Launching: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error("Command not found: goose. Install first.")
        return 1
    except KeyboardInterrupt:
        return 0


def _run_claude_script(
    config: dict, env: dict, working_dir: Path, script_path: Path, max_turns: int,
) -> int:
    """Loop prompts: ``claude -p`` for the first, ``claude --continue -p`` after.

    Claude Code has no native multi-turn batch shape — ``-p`` is one-shot,
    ``--continue`` resumes the most recent session in the same cwd.  Looping
    sends prompts as discrete turns sharing one logical session.

    ``max_turns`` here caps the number of *prompts* we send (one per loop
    iteration), not Claude's internal tool-call loop within a prompt — Claude
    Code doesn't expose the latter.  Wall-clock cap in the smoke wrapper is
    the safety net.
    """
    prompts = _read_script_prompts(script_path)
    if not prompts:
        logger.error("No prompts found in %s", script_path)
        return 1
    if len(prompts) > max_turns:
        logger.warning(
            "Script has %d prompts but max_turns=%d; truncating",
            len(prompts), max_turns,
        )
        prompts = prompts[:max_turns]

    base = ["claude", "--dangerously-skip-permissions"]
    for i, prompt in enumerate(prompts):
        cmd = base + (["--continue"] if i > 0 else []) + ["-p", prompt]
        logger.info("Launching prompt %d/%d", i + 1, len(prompts))
        try:
            result = subprocess.run(cmd, env=env, cwd=str(working_dir))
        except FileNotFoundError:
            logger.error("Command not found: claude. Install Claude Code first.")
            return 1
        except KeyboardInterrupt:
            return 0
        if result.returncode != 0:
            logger.warning(
                "claude prompt %d/%d exited %d; stopping script",
                i + 1, len(prompts), result.returncode,
            )
            return result.returncode
    return 0


def _read_script_prompts(path: Path) -> list[str]:
    """Split a script file into one prompt per blank-line-separated paragraph."""
    text = path.read_text()
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


_SCRIPT_RUNNERS = {
    "goose": _run_goose_script,
    "claude-code": _run_claude_script,
}

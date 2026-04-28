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

- **Roo Code** — Install: ``curl -fsSL
  https://raw.githubusercontent.com/RooCodeInc/Roo-Code/main/apps/cli/install.sh | sh``
  (binary lands at ``~/.local/bin/roo`` — not on npm).  Generates:
  ``.roomodes``, ``.dsagt_env``; per-launch, ``.roo/mcp.json`` is written
  by ``_bootstrap_roo`` with the full env block (roo, like cline,
  doesn't inherit parent env into MCP server children).  Proxy routing:
  ``ANTHROPIC_BASE_URL`` (the ``--provider anthropic`` path uses the
  Anthropic SDK, which honors that env var; the CLI has no ``--base-url``
  flag).  Batch mode: ``roo --print --oneshot --prompt-file FILE`` reads
  multi-line prompts directly — no looping or argv encoding needed.

- **Cline** — Install: ``npm i -g cline``
  Generates: ``.clinerules/dsagt_instructions.md``, ``.dsagt_env``;
  ``cline auth`` and ``cline mcp add`` are run per-launch in
  ``_bootstrap_cline_auth`` and write to ``$CLINE_DIR/data/``.
  Proxy routing: ``cline auth -p openai -b http://localhost:<proxy_port>``
  (Cline ignores ``ANTHROPIC_BASE_URL`` and only allows ``--baseurl`` on
  the openai provider, so cline talks to our proxy via OpenAI-format
  ``/chat/completions`` — the same path goose uses.)  MCP config: hand-
  writing ``cline_mcp_settings.json`` is silently ignored; the only path
  cline loads is via ``cline mcp add``, which writes a stripped schema
  (no ``env`` block).  Per-server env has to come from process-env
  inheritance — ``DSAGT_PROJECT_DIR`` / ``LLM_API_KEY`` /
  ``OPENAI_BASE_URL`` / ``EMBEDDING_MODEL`` are set in ``agent_env`` and
  flow cline → MCP-server-subprocess.

- **Codex** — Install: ``npm i -g @openai/codex`` (or ``brew install
  --cask codex``).  Generates: ``AGENTS.md``, ``.codex-data/`` (the
  per-project ``CODEX_HOME``), ``.dsagt_env``.  Codex's config layer
  has no per-workspace file — instead it reads ``$CODEX_HOME/config.toml``
  for everything (model, provider base_url, MCP servers).  We point
  ``CODEX_HOME`` at ``<working_dir>/.codex-data`` to keep state isolated
  per project, and ``_bootstrap_codex`` writes ``config.toml`` at launch
  with a custom ``[model_providers.dsagt-proxy]`` block routing to our
  proxy and ``[mcp_servers.*]`` blocks with explicit env (codex doesn't
  inherit parent env into MCP children, same as cline/roo).  Wire API:
  Codex's only supported value is ``responses``, so requests hit
  ``/v1/responses`` on the proxy — the same path claude-code and roo
  already exercise.  ``disable_response_storage = true`` is required
  because the upstream gateway behind our proxy translates
  ``/v1/responses`` → ``/v1/chat/completions`` and can't preserve
  ``previous_response_id`` state across calls.
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
    "codex": ["codex"],
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


_PROXY_FORWARDED_SENTINEL = "dsagt-proxy-forwarded-disable-direct-calls"


def _mcp_env_block(config: dict, proxy_port: int) -> dict:
    """Build the env block for MCP server entries (Claude Code .mcp.json).

    Embedding requests from dsagt-knowledge-server route through our local
    LiteLLM proxy at localhost:<proxy_port>, same as agent LLM calls — the
    proxy holds the real upstream credentials and translates to whatever
    provider is configured in its model_list.  MCP children only see the
    sentinel API key and the proxy URL; if anything misroutes around the
    proxy, a direct call to api.openai.com / api.cohere.com / etc. will
    401 the sentinel and fail loudly instead of silently bypassing.
    """
    proxy_url = f"http://localhost:{proxy_port}"
    return {
        "DSAGT_PROJECT_DIR": config["project_dir"],
        "EMBEDDING_BASE_URL": proxy_url,
        "EMBEDDING_API_KEY": _PROXY_FORWARDED_SENTINEL,
        # OPENAI_BASE_URL kept aligned with EMBEDDING_BASE_URL so any
        # LiteLLM internal openai-client fallback also goes through proxy.
        "OPENAI_BASE_URL": proxy_url,
        "EMBEDDING_MODEL": config.get("embedding", {}).get("model", ""),
    }


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


# Marker string we look for to decide whether an instructions file already
# carries the dsagt body.  Present in the master instructions header and in
# every per-agent format we emit (Claude Code's CLAUDE.md, codex's AGENTS.md,
# goose's .goosehints, cline's .clinerules/dsagt_instructions.md, roo's
# .roomodes which embeds the instructions in JSON).  Lets ``_append_or_write``
# be idempotent so users can edit instructions files between init and start
# without losing their edits on the next start.
_DSAGT_MARKER = "DSAgt Pipeline Builder"


def _append_or_write(path: Path, content: str, marker: str) -> str | None:
    """Idempotent write for instructions files.

    - File doesn't exist → write content.
    - File exists, marker absent → append content (preserves user prefix).
    - File exists, marker present → no-op (preserves user edits).

    Returns a one-line action description, or None on no-op.
    """
    if path.exists():
        existing = path.read_text()
        if marker in existing:
            return None
        path.write_text(existing + "\n\n" + content)
        return f"Appended DSAgt instructions to {path}"
    path.write_text(content)
    return f"Wrote {path}"


# ---------------------------------------------------------------------------
# Static record: instructions + empty state dirs.  No env, no ports, no
# session id.  Safe to call from ``dsagt init`` (so the user can edit
# CLAUDE.md / AGENTS.md / .goosehints before first start) or from
# ``dsagt start`` (when the user ran init without --agent, or switched
# agents, or accidentally deleted a marker file).
# ---------------------------------------------------------------------------

# Marker files used by ``static_agent_files_present`` to decide whether
# the active agent has had its static record written into this project.
# Each entry is the file ``static_agent_record`` writes that uniquely
# identifies an agent's setup having run — avoid using directories
# (.codex-data, .cline-data) since those can exist with empty contents
# from a partially-cleaned previous run.
_STATIC_MARKER_FILES = {
    "claude-code": "CLAUDE.md",
    "goose": ".goosehints",
    "roo": ".roomodes",
    "cline": ".clinerules/dsagt_instructions.md",
    "codex": "AGENTS.md",
}


def static_agent_record(
    config: dict, agent: str, working_dir: str | Path,
) -> list[str]:
    """Write the agent's static project files: instructions + state dirs.

    Idempotent.  If the dsagt marker is already in the instructions file,
    the write is skipped — preserves any user edits made between init
    and start.  Empty state directories use ``mkdir(exist_ok=True)``.

    The ``config`` arg is reserved for future use (e.g. project-specific
    instruction header injection); today it's unused but keeps the
    signature symmetric with ``dynamic_agent_record``.

    Returns a list of one-line descriptions of what was written.
    """
    del config  # reserved
    working_dir = Path(working_dir)
    statics = {
        "claude-code": _static_claude_code,
        "goose": _static_goose,
        "roo": _static_roo,
        "cline": _static_cline,
        "codex": _static_codex,
    }
    return statics[agent](working_dir)


def static_agent_files_present(agent: str, working_dir: str | Path) -> bool:
    """True if the agent's marker file exists in the working dir.

    Used by ``dsagt start`` to decide whether to call
    ``static_agent_record`` on its way to launch.  Cheap stat check, no
    parsing.
    """
    return (Path(working_dir) / _STATIC_MARKER_FILES[agent]).exists()


def _static_claude_code(working_dir: Path) -> list[str]:
    actions = []
    instructions = _load_master_instructions()
    if instructions:
        action = _append_or_write(
            working_dir / "CLAUDE.md", instructions, _DSAGT_MARKER,
        )
        if action:
            actions.append(action)
    return actions


def _static_goose(working_dir: Path) -> list[str]:
    actions = []
    instructions = _load_master_instructions()
    if instructions:
        action = _append_or_write(
            working_dir / ".goosehints", instructions, _DSAGT_MARKER,
        )
        if action:
            actions.append(action)
    return actions


def _static_roo(working_dir: Path) -> list[str]:
    actions = []
    (working_dir / ".roo").mkdir(exist_ok=True)
    instructions = _load_master_instructions()
    if instructions:
        # Roo wraps the instructions in a customMode JSON envelope; the
        # master marker text survives the wrap because it's part of the
        # role definition body.
        action = _append_or_write(
            working_dir / ".roomodes",
            _format_roomodes(instructions),
            _DSAGT_MARKER,
        )
        if action:
            actions.append(action)
    return actions


def _static_cline(working_dir: Path) -> list[str]:
    actions = []
    (working_dir / ".cline-data").mkdir(parents=True, exist_ok=True)
    instructions = _load_master_instructions()
    if instructions:
        rules_dir = working_dir / ".clinerules"
        rules_dir.mkdir(exist_ok=True)
        action = _append_or_write(
            rules_dir / "dsagt_instructions.md",
            instructions,
            _DSAGT_MARKER,
        )
        if action:
            actions.append(action)
    return actions


def _static_codex(working_dir: Path) -> list[str]:
    actions = []
    (working_dir / ".codex-data").mkdir(parents=True, exist_ok=True)
    instructions = _load_master_instructions()
    if instructions:
        action = _append_or_write(
            working_dir / "AGENTS.md", instructions, _DSAGT_MARKER,
        )
        if action:
            actions.append(action)
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

    if agent in ("claude-code", "roo"):
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{proxy_port}"
        # Pin the model.  Without this Claude Code falls back to its built-in
        # default (currently claude-sonnet-4-6) which won't exist on project
        # gateways like PNNL's ai-incubator-api and the proxy will 400 every
        # request.
        env["ANTHROPIC_MODEL"] = config["llm"]["model"]
    if agent == "cline":
        # Cline ignores ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY.  Provider
        # config is stored in CLINE_DIR/globalState.json (written by
        # `cline auth`, bootstrapped in launch_agent).  CLINE_DIR is
        # project-scoped so MCP config + auth stay isolated per project.
        env["CLINE_DIR"] = str(Path(pdir) / ".cline-data")
    if agent == "goose":
        env["OPENAI_HOST"] = f"http://localhost:{proxy_port}"
        # Override any global ~/.config/goose/config.yaml so the project's
        # configured model is what actually runs.  Without these, a user's
        # global GOOSE_MODEL (e.g. a model their upstream doesn't offer)
        # silently wins over dsagt_config.yaml.
        env["GOOSE_PROVIDER"] = "openai"
        env["GOOSE_MODEL"] = config["llm"]["model"]
    if agent == "codex":
        # Codex reads model + provider + base_url + MCP servers from
        # $CODEX_HOME/config.toml (no per-workspace file).  Per-project
        # CODEX_HOME isolates state so multiple dsagt projects don't collide
        # on global config.  config.toml is written by _bootstrap_codex.
        env["CODEX_HOME"] = str(Path(pdir) / ".codex-data")

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
    if agent in ("claude-code", "roo"):
        env["ANTHROPIC_API_KEY"] = sentinel_key
    if agent == "goose":
        env["OPENAI_API_KEY"] = sentinel_key
    if agent == "codex":
        # Codex's [model_providers.dsagt-proxy] config.toml entry sets
        # env_key = "OPENAI_API_KEY", so the auth header sent to our proxy
        # is read from this var.  Sentinel value plays the same role as
        # for goose: any direct call bypassing the proxy 401s loudly at
        # the real upstream instead of silently succeeding.
        env["OPENAI_API_KEY"] = sentinel_key
    # Cline's sentinel goes into CLINE_DIR/globalState.json via
    # `cline auth -k`; same direct-call protection, different injection
    # point — see _bootstrap_cline_auth.

    # Embedding routing: every subprocess (agent → MCP server child, or
    # MCP server inherited via parent env on Claude Code/goose) sees the
    # proxy URL for embeddings.  The real EMBEDDING_BASE_URL/API_KEY only
    # live in the dsagt-proxy subprocess's env (inherited from os.environ
    # before agent_env's override), where the proxy uses them to forward
    # upstream.  Sentinel API key on the agent side prevents accidental
    # direct calls to the real provider — same protection pattern as
    # ANTHROPIC_API_KEY/OPENAI_API_KEY for the LLM path.
    env["EMBEDDING_BASE_URL"] = f"http://localhost:{proxy_port}"
    env["EMBEDDING_API_KEY"] = sentinel_key
    env["OPENAI_BASE_URL"] = f"http://localhost:{proxy_port}"
    if config.get("embedding", {}).get("model"):
        env["EMBEDDING_MODEL"] = config["embedding"]["model"]

    # Roo uses --provider anthropic.  ANTHROPIC_BASE_URL set above points
    # roo at the proxy.  Roo rewrites our PNNL model name into its own
    # default (``claude-sonnet-4-5``) before sending — the proxy aliases
    # that name back to the upstream primary in _generate_config.

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
    env: dict,
    working_dir: str | Path,
    script_path: str | Path | None = None,
    max_turns: int = 30,
) -> int:
    """Launch the agent in the foreground.  Blocks until it exits.

    Returns the agent's exit code.  Caller is responsible for having run
    ``static_agent_record`` (at init time, or at start time when files
    are missing) and ``dynamic_agent_record`` (at start time, after
    services are up).  This function only dispatches to the agent's
    runner with the env dict in hand.

    When ``script_path`` is set, the agent runs in non-interactive batch
    mode and we dispatch to a per-agent runner — each agent has a
    different shape for "run a script":

      goose:        single ``goose run --instructions FILE`` call
      claude-code:  single ``claude -p SCRIPT`` call
      cline:        single ``cline -y SCRIPT`` call (--continue rejects new
                    prompts, so we can't loop turns)
      roo:          single ``roo --print --oneshot --prompt-file FILE`` call
      codex:        single ``codex exec --yolo --skip-git-repo-check`` call

    Other agents raise — add a runner here when adding agent support.
    """
    working_dir = Path(working_dir)

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
    """Single ``claude -p`` call with the entire script as one prompt.

    Mirrors goose/cline/roo: hand the agent the whole script in one
    invocation and let its internal tool-call loop drive multi-turn
    behavior.  The prior loop shape (one ``claude -p`` per paragraph,
    chained with ``--continue``) inflated completion-call counts and token
    usage 5–20× without changing what the agent actually accomplished.

    ``max_turns`` is not applicable in single-shot mode (Claude Code has no
    flag exposing its internal turn cap).  The smoke wrapper's wall-clock
    timeout is the safety net.
    """
    del max_turns  # see docstring
    text = script_path.read_text().strip()
    if not text:
        logger.error("Script is empty: %s", script_path)
        return 1

    cmd = ["claude", "--dangerously-skip-permissions", "-p", text]
    logger.info("Launching claude (single-prompt batch)")
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error("Command not found: claude. Install Claude Code first.")
        return 1
    except KeyboardInterrupt:
        return 0


def _run_cline_script(
    config: dict, env: dict, working_dir: Path, script_path: Path, max_turns: int,
) -> int:
    """Single ``cline -y`` call with the entire script as one prompt.

    Cline's ``--continue`` resumes a task but rejects a new prompt ("Use
    --continue without a prompt"), so we can't loop discrete turns the way
    claude-code does.  Instead, hand cline the whole script in one task —
    same shape as goose's ``--instructions FILE``.  Multi-turn behavior
    happens inside cline's autonomous tool-call loop, not across processes.

    ``max_turns`` is not applicable in single-shot mode (cline has no
    internal turn cap exposed via flags).  The smoke wrapper's wall-clock
    timeout is the safety net.
    """
    del max_turns  # see docstring
    text = script_path.read_text().strip()
    if not text:
        logger.error("Script is empty: %s", script_path)
        return 1

    # -v shows model reasoning + tool calls inline; without it, cline emits
    # only the task id and final summary, which is opaque during the smoke
    # test and makes MCP/tool-call failures invisible.
    cmd = ["cline", "-v", "-y", text]
    logger.info("Launching cline (single-prompt batch, verbose)")
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error("Command not found: cline. Install Cline CLI first.")
        return 1
    except KeyboardInterrupt:
        return 0


def _dynamic_cline(config, env, working_dir, pdir, proxy_port) -> list[str]:
    """Cline's runtime-dependent setup: auth + MCP registration + env file.

    Three side-effects, all of which need the launch-time env dict:

    1. ``cline auth -p openai -b URL`` writes provider config (proxy URL
       + sentinel key + model) into ``$CLINE_DIR/globalState.json``.
       Cline ignores ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` env
       vars; the auth subcommand is the only path.  Provider is
       ``openai`` because ``cline auth -p anthropic -b URL`` errors out:
       "Base URL is only supported for OpenAI and OpenAI-compatible
       providers".  Cline-as-openai sends ``/chat/completions`` to our
       proxy — same path goose uses; proxy translates upstream regardless.

    2. ``cline mcp add`` per server registers the dsagt MCP servers into
       cline's loaded state.  Hand-writing
       ``cline_mcp_settings.json`` is silently ignored — only the
       subcommand path works.

    3. Patch the JSON cline wrote with our env block.  Cline doesn't
       inherit parent process env into MCP server subprocesses (unlike
       claude-code and goose), so the dsagt servers' env vars
       (``MLFLOW_TRACKING_URI``, ``EMBEDDING_*``, ``DSAGT_SESSION_ID``,
       etc.) must live in the JSON.  An earlier attempt added
       ``disabled`` and ``alwaysAllow`` keys here too and cline silently
       rejected the whole config — adding only ``env`` keeps the schema
       close enough to what ``cline mcp add`` wrote that the rest of
       the entry still parses.

    Plus the ``.dsagt_env`` shell file (used by manual ``source`` workflows).
    """
    actions = []
    cline_dir = env.get("CLINE_DIR") or str(working_dir / ".cline-data")
    Path(cline_dir).mkdir(parents=True, exist_ok=True)
    sentinel = _PROXY_FORWARDED_SENTINEL
    model = config["llm"]["model"]

    auth_cmd = [
        "cline", "auth",
        "-p", "openai",
        "-k", sentinel,
        "-m", model,
        "-b", f"http://localhost:{proxy_port}",
    ]
    logger.info("Running: cline auth (CLINE_DIR=%s)", cline_dir)
    try:
        result = subprocess.run(
            auth_cmd, env=env, cwd=str(working_dir),
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "cline command not found. Install with `npm i -g cline`."
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"cline auth failed (exit {result.returncode}): {detail}"
        )
    actions.append(f"Configured cline auth at {cline_dir}")

    for server in ("registry", "knowledge"):
        add_cmd = [
            "cline", "mcp", "add",
            "--config", cline_dir,
            f"dsagt-{server}",
            "--",
            "uv", "run", f"dsagt-{server}-server",
        ]
        result = subprocess.run(
            add_cmd, env=env, cwd=str(working_dir),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"cline mcp add dsagt-{server} failed (exit {result.returncode}): {detail}"
            )

    mcp_path = Path(cline_dir) / "data" / "settings" / "cline_mcp_settings.json"
    if not mcp_path.exists():
        raise RuntimeError(
            f"cline mcp add succeeded but {mcp_path} was not created — "
            "cline may have changed its config layout."
        )
    settings = json.loads(mcp_path.read_text())
    mcp_env = _mcp_subprocess_env(env)
    for entry in settings.get("mcpServers", {}).values():
        entry["env"] = mcp_env
    mcp_path.write_text(json.dumps(settings, indent=2) + "\n")
    actions.append(
        f"Registered MCP servers and patched {len(mcp_env)} env vars into {mcp_path}"
    )

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "CLINE_DIR": str(cline_dir),
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")
    return actions


def _mcp_subprocess_env(parent_env: dict) -> dict:
    """Pick the env vars a dsagt MCP server child needs from the launch env.

    Cline and roo strip parent env when spawning MCP server subprocesses
    (Claude Code and goose inherit, so they don't need this), so everything
    the server needs to initialize must be listed explicitly in the JSON
    env block.  The ``${LLM_*}``/``${EMBEDDING_*}`` placeholders in
    ``dsagt_config.yaml`` are resolved at server startup via
    ``resolve_env_vars``; if any are missing, the placeholder leaks through
    and the knowledge server's ``api_key.startswith("${")`` validation
    kills the process — manifesting agent-side as a silent connection
    failure.  Pass them all through alongside the dsagt-set vars.
    """
    keys = (
        "DSAGT_PROJECT_DIR", "DSAGT_PROJECT", "DSAGT_SESSION_ID", "DSAGT_AGENT",
        "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL",
        "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
        "OPENAI_BASE_URL",
        "MLFLOW_TRACKING_URI",
    )
    return {k: parent_env[k] for k in keys if k in parent_env}


#: Tools each dsagt MCP server exposes — listed in ``alwaysAllow`` so roo
#: auto-approves them without a human-in-the-loop prompt that ``--print``
#: mode can't answer.  Keep in sync with ``commands/registry_server.py``
#: and ``commands/knowledge_server.py`` tool registrations.  Adding a tool
#: there but forgetting it here means roo will hang on its first call.
_DSAGT_MCP_ALWAYS_ALLOW = {
    "registry": [
        "get_registry", "http_request", "install_dependencies", "read_file",
        "reconstruct_pipeline", "run_command", "save_tool_spec",
        "search_registry", "search_skills",
    ],
    "knowledge": [
        "kb_add_vector_db", "kb_append", "kb_dismiss_suggestion",
        "kb_get_memories", "kb_get_suggestions", "kb_ingest", "kb_job_status",
        "kb_list_collections", "kb_remember", "kb_search",
    ],
}


def _build_mcp_servers_dict(env_block: dict | None) -> dict:
    """Build the standard ``{"mcpServers": {...}}`` dict for the dsagt servers.

    Used by agents that load MCP config from a JSON file (roo via
    ``.roo/mcp.json``).  Claude Code uses the same shape via ``.mcp.json``
    but builds it inline in ``_generate_claude_code``.  Cline doesn't use
    this — it requires ``cline mcp add`` to register servers.
    """
    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {
            "command": "uv",
            "args": _mcp_server_args(server),
            "disabled": False,
            "alwaysAllow": _DSAGT_MCP_ALWAYS_ALLOW[server],
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry
    return mcp_config


def _dynamic_roo(config, env, working_dir, pdir, proxy_port) -> list[str]:
    """Roo's runtime-dependent setup: ``.roo/mcp.json`` + ``.dsagt_env``.

    Roo doesn't inherit parent env into MCP server children — every var
    the dsagt servers need (``${EMBEDDING_*}`` substitution placeholders,
    ``MLFLOW_TRACKING_URI``, ``DSAGT_SESSION_ID``) must be explicit in
    the JSON env block.

    ``proxy_port`` is unused — the proxy URL goes onto the agent's
    command line at launch (``_run_roo_script`` passes ``--api-key`` /
    ``--model`` directly), and the env block carries everything the MCP
    children need.
    """
    del proxy_port
    actions = []
    mcp_path = working_dir / ".roo" / "mcp.json"
    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mcp_env = _mcp_subprocess_env(env)
    mcp_config = _build_mcp_servers_dict(mcp_env)
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path} ({len(mcp_env)} env vars)")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")
    return actions


def _run_roo_script(
    config: dict, env: dict, working_dir: Path, script_path: Path, max_turns: int,
) -> int:
    """Single ``roo --print --oneshot --prompt-file FILE`` call.

    Roo CLI exposes ``--prompt-file`` for batch input — much cleaner than
    cline (no need to argv-encode multi-line) or claude-code (no need to
    loop with ``--continue``).  ``--print --oneshot`` is roo's
    non-interactive single-task shape.

    Provider is ``anthropic``.  Roo's anthropic provider rewrites
    unrecognized model names (lab-gateway aliases like PNNL's
    ``claude-haiku-4-5-20251001-v1-project``) into its current default
    (``claude-sonnet-4-5`` as of v0.1.x).  We sidestep that by aliasing
    that name in the proxy config (see ``_generate_config`` in
    ``commands/proxy_server.py``) so it routes to the same upstream as
    the configured primary.  Avoids the openai-native path entirely,
    which hits ``/responses`` and tickles a LiteLLM proxy streaming bug
    (``TypeError: ResponsesAPIResponse not async-iterable``).

    ``max_turns`` is unused — roo has its own consecutive-mistake cap
    (``--consecutive-mistake-limit``) but no overall turn cap.
    """
    del max_turns
    sentinel = "dsagt-proxy-forwarded-disable-direct-calls"
    # MCP tool calls otherwise hang on an "approve?" prompt that --print
    # mode can't answer.  Roo doesn't have a global yolo flag — auto-
    # approval is per-tool, listed in each MCP entry's ``alwaysAllow``
    # array (see _DSAGT_MCP_ALWAYS_ALLOW + _build_mcp_servers_dict).
    # `--mode dsagt` activates the custom mode we defined in .roomodes
    # (slug: "dsagt", see _format_roomodes).  Without this, roo runs in
    # its default "code" mode and our entire customInstructions body —
    # CRITICAL CONSTRAINTS, kb_remember rule, dsagt-run rule, everything
    # — is dropped from the system prompt.  The model only sees the
    # roleDefinition (first paragraph of dsagt_instructions.md).  This
    # was the root cause of the hallucinations and rule-bypassing we
    # spent a lot of debugging time on; the instructions weren't being
    # ignored, they weren't reaching the model at all.
    cmd = [
        "roo",
        "--print", "--oneshot",
        "--mode", "dsagt",
        "--prompt-file", str(script_path),
        "--provider", "anthropic",
        "--api-key", env.get("ANTHROPIC_API_KEY", sentinel),
        "--model", config["llm"]["model"],
        "--workspace", str(working_dir),
        "--debug",
    ]
    logger.info("Launching roo (single-prompt batch, debug)")
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error(
            "Command not found: roo. Install via "
            "https://github.com/RooCodeInc/Roo-Code/blob/main/apps/cli/install.sh"
        )
        return 1
    except KeyboardInterrupt:
        return 0


def _toml_quote(value: str) -> str:
    """TOML-quote a string: escape backslashes and double-quotes only.

    Codex config.toml is regular TOML — basic strings need backslash and
    quote escaping but not control chars (which we don't have in any of
    the values we emit: paths, URLs, model names).  Avoids pulling in a
    TOML writer dep just for a few lines.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_codex_config(config: dict, env: dict, mcp_env: dict) -> str:
    """Render the per-project ``$CODEX_HOME/config.toml`` body.

    Pinned shape (see module docstring for the why):
      - ``model_provider = "dsagt-proxy"`` routes every call through our
        local LiteLLM proxy.  Built-in ``openai`` provider is left intact
        so a future debug session can flip back via ``-c
        model_provider=openai`` without rewriting config.
      - ``wire_api = "responses"`` is Codex's only supported value.
        Proxy already exercises this path for claude-code and roo.
      - ``requires_openai_auth = false`` skips the ``codex login``
        OAuth flow — the sentinel ``OPENAI_API_KEY`` env var is enough.
      - ``disable_response_storage = true`` because the upstream behind
        our proxy translates ``/v1/responses`` → ``/v1/chat/completions``
        and can't honor ``previous_response_id`` state.  Without this,
        the second turn 400s with "previous_response_id not found".
      - ``approval_policy = "never"`` and ``sandbox_mode =
        "danger-full-access"`` so batch mode runs without approval
        prompts.  ``codex exec --yolo`` sets the same at the CLI but
        also writing to config makes interactive ``dsagt start``
        sessions behave consistently.
    """
    proxy_port = config["proxy"]["port"]
    model = config["llm"]["model"]
    base_url = f"http://localhost:{proxy_port}/v1"

    lines = [
        f"model = {_toml_quote(model)}",
        'model_provider = "dsagt-proxy"',
        'approval_policy = "never"',
        'sandbox_mode = "danger-full-access"',
        "disable_response_storage = true",
        "",
        "[model_providers.dsagt-proxy]",
        'name = "DSAgt Proxy"',
        f"base_url = {_toml_quote(base_url)}",
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "responses"',
        "requires_openai_auth = false",
        "",
    ]
    for server in ("registry", "knowledge"):
        lines.append(f"[mcp_servers.dsagt-{server}]")
        lines.append('command = "uv"')
        args = _mcp_server_args(server)
        args_toml = ", ".join(_toml_quote(a) for a in args)
        lines.append(f"args = [{args_toml}]")
        if mcp_env:
            lines.append(f"[mcp_servers.dsagt-{server}.env]")
            for k, v in mcp_env.items():
                lines.append(f"{k} = {_toml_quote(v)}")
        lines.append("")
    return "\n".join(lines)


def _dynamic_codex(config, env, working_dir, pdir, proxy_port) -> list[str]:
    """Codex's runtime-dependent setup: ``$CODEX_HOME/config.toml`` + env file.

    Codex MCP children, like cline's and roo's, don't inherit parent
    process env — every var the dsagt servers need has to be explicit
    in the per-server ``[mcp_servers.*.env]`` table.

    ``proxy_port`` is read from ``config`` inside ``_render_codex_config``;
    we accept it as a positional for symmetry with the other dynamic
    writers but don't use it directly.
    """
    del proxy_port
    actions = []
    codex_home = env.get("CODEX_HOME") or str(working_dir / ".codex-data")
    Path(codex_home).mkdir(parents=True, exist_ok=True)
    config_path = Path(codex_home) / "config.toml"
    mcp_env = _mcp_subprocess_env(env)
    config_path.write_text(_render_codex_config(config, env, mcp_env) + "\n")
    actions.append(f"Wrote {config_path} ({len(mcp_env)} MCP env vars)")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "CODEX_HOME": str(codex_home),
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")
    return actions


def _dynamic_claude_code(config, env, working_dir, pdir, proxy_port) -> list[str]:
    """Claude Code's runtime-dependent setup: ``.mcp.json`` + ``.dsagt_env``.

    Claude Code DOES inherit parent process env into MCP server children,
    so the explicit env block in ``.mcp.json`` is partly redundant — but
    we include it anyway to keep behavior identical when the agent is
    launched outside our ``dsagt start`` flow (e.g. someone runs
    ``claude`` directly in the project dir after sourcing
    ``.dsagt_env``).
    """
    del env  # claude-code's MCP env block is config-derived, not env-derived
    actions = []
    env_block = _mcp_env_block(config, proxy_port)

    mcp_config = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry = {"command": "uv", "args": _mcp_server_args(server)}
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry

    mcp_path = working_dir / ".mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    actions.append(f"Wrote {mcp_path}")

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")
    return actions


def _dynamic_goose(config, env, working_dir, pdir, proxy_port) -> list[str]:
    """Goose's runtime-dependent setup: ``goose.yaml`` + ``.dsagt_env``.

    Goose inherits parent env into MCP children, so the per-extension
    block doesn't need an explicit env list — the proxy URL flows in
    via ``OPENAI_HOST`` from the agent's environment.
    """
    del env  # goose inherits parent env; nothing to embed in goose.yaml
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

    env_path = working_dir / ".dsagt_env"
    _write_env_file(env_path, {
        "OPENAI_HOST": f"http://localhost:{proxy_port}",
        "DSAGT_PROJECT": config["project"],
        "DSAGT_PROJECT_DIR": str(pdir),
    })
    actions.append(f"Wrote {env_path}")
    return actions


def dynamic_agent_record(
    config: dict, env: dict, working_dir: str | Path,
) -> list[str]:
    """Write the agent's runtime-dependent files: MCP config + ``.dsagt_env``,
    and (for cline) run ``cline auth`` + ``cline mcp add`` subprocesses.

    Caller must have already:
      - Resolved the agent and stored it in ``config["agent"]``
      - Started services and updated ``config["proxy"]["port"]`` /
        ``config["mlflow"]["port"]`` to the actually-bound ports
      - Built ``env`` via ``agent_env(config)``

    Returns a list of one-line descriptions of what was written.
    """
    agent = config["agent"]
    working_dir = Path(working_dir)
    pdir = Path(config["project_dir"])
    proxy_port = config["proxy"]["port"]

    dynamics = {
        "claude-code": _dynamic_claude_code,
        "goose": _dynamic_goose,
        "roo": _dynamic_roo,
        "cline": _dynamic_cline,
        "codex": _dynamic_codex,
    }
    return dynamics[agent](config, env, working_dir, pdir, proxy_port)


def _run_codex_script(
    config: dict, env: dict, working_dir: Path, script_path: Path, max_turns: int,
) -> int:
    """Single ``codex exec --yolo --skip-git-repo-check -C <wd> <prompt>`` call.

    ``codex exec`` is Codex's non-interactive mode — runs the prompt to
    completion and exits with the assistant's final message on stdout.
    ``--yolo`` is Codex's alias for ``--dangerously-bypass-approvals-and-
    sandbox`` (skips both shell-command approval prompts and sandboxing,
    which batch mode can't answer).  ``--skip-git-repo-check`` allows
    running outside a git repo (smoke-test project dirs aren't git
    repos).

    ``max_turns`` is unused — Codex has no exposed turn cap.  Wall-clock
    cap in the smoke wrapper is the safety net.
    """
    del max_turns
    text = script_path.read_text().strip()
    if not text:
        logger.error("Script is empty: %s", script_path)
        return 1

    cmd = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-C", str(working_dir),
        text,
    ]
    logger.info("Launching codex exec (single-prompt batch)")
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error(
            "Command not found: codex. Install with `npm i -g @openai/codex` "
            "or `brew install --cask codex`."
        )
        return 1
    except KeyboardInterrupt:
        return 0


_SCRIPT_RUNNERS = {
    "goose": _run_goose_script,
    "claude-code": _run_claude_script,
    "cline": _run_cline_script,
    "roo": _run_roo_script,
    "codex": _run_codex_script,
}

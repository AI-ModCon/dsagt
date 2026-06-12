"""
Tests for DSAGT config loading, project init, and agent config generation.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from dsagt.session import (
    _deep_merge,
    resolve_env_vars,
    default_config_content,
    load_config,
    project_dir,
)
from dsagt.agents import (
    agent_env,
    dynamic_agent_record,
    static_agent_record,
    static_agent_files_present,
)
from dsagt.session import init_project, persist_agent_choice


@pytest.fixture(autouse=True)
def _use_tmp_registry(tmp_path):
    """Redirect project registry and default location to tmp_path for all tests.

    The fake registry auto-discovers any project dir that exists under tmp_path,
    so tests that create dirs manually (without init_project) still work.
    """
    registry = {}

    def fake_load():
        # Auto-discover: any subdir of tmp_path with dsagt_config.yaml counts
        discovered = dict(registry)
        for child in tmp_path.iterdir():
            if child.is_dir() and (child / "dsagt_config.yaml").exists():
                discovered.setdefault(child.name, str(child))
        return discovered

    def fake_save(reg):
        registry.clear()
        registry.update(reg)

    def fake_register(name, path):
        registry[name] = str(Path(path).resolve())

    with patch("dsagt.session._load_registry", fake_load):
        with patch("dsagt.session._save_registry", fake_save):
            with patch("dsagt.session.register_project", fake_register):
                with patch("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path):
                    with patch("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path):
                        yield


# ---------------------------------------------------------------------------
# Config: env var resolution
# ---------------------------------------------------------------------------


class TestResolveEnvVars:

    def test_resolves_set_var(self):
        with patch.dict(os.environ, {"MY_KEY": "secret"}):
            assert resolve_env_vars("${MY_KEY}") == "secret"

    def test_unset_var_left_as_is(self):
        os.environ.pop("NOPE", None)
        assert resolve_env_vars("${NOPE}") == "${NOPE}"

    def test_nested_dicts(self):
        with patch.dict(os.environ, {"K": "v"}):
            result = resolve_env_vars({"a": {"b": "${K}"}})
            assert result == {"a": {"b": "v"}}

    def test_non_string_passthrough(self):
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(True) is True


# ---------------------------------------------------------------------------
# Config: deep merge
# ---------------------------------------------------------------------------


class TestDeepMerge:

    def test_override_leaf(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        result = _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 99}})
        assert result == {"a": {"b": 99, "c": 2}}

    def test_new_keys_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Config: load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:

    def _write_config(self, tmp_path, name, content: dict):
        pdir = tmp_path / name
        pdir.mkdir(exist_ok=True)
        (pdir / "dsagt_config.yaml").write_text(
            yaml.dump(content, default_flow_style=False)
        )
        return name

    def test_loads_minimal_config(self, tmp_path):
        name = self._write_config(
            tmp_path,
            "myproject",
            {
                "project": "myproject",
                "agent": "goose",
                "llm": {"provider": "openai"},
            },
        )

        config = load_config(name)

        assert config["project"] == "myproject"
        assert config["agent"] == "goose"
        # Ports are no longer defaulted in YAML — start_services picks them
        # at runtime via socket.bind(("", 0)) and writes them to .runtime.
        assert "proxy" not in config or "port" not in config.get("proxy", {})
        # User-supplied keys win on the merge; DEFAULTS fills in missing
        # llm.* fields with ``${VAR}`` placeholders (resolved by
        # ``resolve_env_vars`` against the user's shell, or filtered by
        # ``_real()`` if env unset).  In BYOA the agent_env gate prevents
        # env_overrides from acting on these — they only matter when
        # --enable-proxy populates ``config["proxy"]["port"]``.
        assert config["llm"]["provider"] == "openai"  # user value preserved
        assert config["mlflow"]["backend"] == "sqlite"  # non-llm defaults kept

    def test_overrides_defaults(self, tmp_path):
        name = self._write_config(
            tmp_path,
            "myproject",
            {
                "project": "myproject",
                "agent": "claude",
                "llm": {"provider": "openai"},
                "proxy": {"port": 9000},  # explicit override stays honored
            },
        )

        config = load_config(name)
        assert config["proxy"]["port"] == 9000

    def test_missing_project_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {"agent": "goose"})
        with pytest.raises(ValueError, match="project"):
            load_config(name)

    def test_skills_block_backfilled_for_old_config(self, tmp_path):
        """A config written before the skills block still gets the default."""
        name = self._write_config(
            tmp_path,
            "myproject",
            {"project": "myproject", "agent": "claude", "llm": {"provider": "openai"}},
        )
        config = load_config(name)
        sources = config["skills"]["sources"]
        assert sources[0]["name"] == "scientific"
        assert "K-Dense-AI" in sources[0]["url"]
        assert config["skills"]["populate_native"] is True
        assert config["skills"]["populate_catalog"] is True

    def test_missing_agent_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {"project": "myproject"})
        with pytest.raises(ValueError, match="agent"):
            load_config(name)

    def test_invalid_agent_raises(self, tmp_path):
        name = self._write_config(
            tmp_path, "myproject", {"project": "myproject", "agent": "copilot"}
        )
        with pytest.raises(ValueError, match="copilot"):
            load_config(name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent")

    def test_project_dir_injected(self, tmp_path):
        name = self._write_config(
            tmp_path,
            "myproject",
            {
                "project": "myproject",
                "agent": "goose",
                "llm": {"provider": "openai"},
            },
        )
        config = load_config(name)
        assert config["project_dir"] == str(tmp_path / "myproject")


class TestSkillsDefaults:

    def test_defaults_has_skills(self):
        from dsagt.session import DEFAULTS

        assert DEFAULTS["skills"]["sources"][0]["name"] == "scientific"

    def test_default_config_content_includes_skills(self):
        body = yaml.safe_load(default_config_content("p", "claude", 5001))
        assert body["skills"]["sources"][0]["name"] == "scientific"


# ---------------------------------------------------------------------------
# Config: helpers
# ---------------------------------------------------------------------------


class TestProjectDir:

    def test_registered_project_resolves(self, tmp_path):
        from dsagt.session import register_project

        register_project("myproj", tmp_path / "myproj")
        result = project_dir("myproj")
        assert result == tmp_path / "myproj"

    def test_unregistered_project_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            project_dir("nonexistent")


class TestDefaultConfigContent:

    def test_roundtrips_as_valid_yaml(self):
        content = default_config_content("test", "goose", mlflow_port=5000)
        parsed = yaml.safe_load(content)
        assert parsed["project"] == "test"
        assert parsed["agent"] == "goose"

    def test_no_user_facing_llm_block(self):
        """BYOA: project YAML is internal-only — no llm: block, no
        ${VAR} placeholders.  User credentials live in their shell."""
        content = default_config_content("test", "claude", mlflow_port=5000)
        parsed = yaml.safe_load(content)
        assert "llm" not in parsed
        assert "${" not in content

    def test_pinned_mlflow_port_lands_in_yaml(self):
        content = default_config_content("test", "claude", mlflow_port=12345)
        parsed = yaml.safe_load(content)
        assert parsed["mlflow"]["port"] == 12345


# ---------------------------------------------------------------------------
# CLI: init_project
# ---------------------------------------------------------------------------


class TestInitProject:

    def test_creates_directory_structure(self):
        pdir, _port = init_project("myproj", "goose")

        assert pdir.exists()
        assert (pdir / "dsagt_config.yaml").exists()
        assert (pdir / "trace_archive").is_dir()
        assert (pdir / "mlflow").is_dir()
        assert (pdir / "skills").is_dir()
        assert (pdir / "kb_index").is_dir()
        assert (pdir / ".dsagt").is_dir()
        # `tools/` is intentionally NOT created by init_project — ToolRegistry
        # creates it on first server startup so bundled tools get copied in.
        assert not (pdir / "tools").exists()

    def test_config_is_valid(self):
        init_project("myproj", "claude")
        config = load_config("myproj")
        assert config["project"] == "myproj"
        assert config["agent"] == "claude"

    def test_returns_pdir_and_port(self):
        """BYOA: init_project returns (pdir, mlflow_port)."""
        pdir, port = init_project("myproj", "goose")
        assert isinstance(port, int) and port > 0
        # Port is persisted in the YAML so MCP servers can read it.
        config = load_config("myproj")
        assert config["mlflow"]["port"] == port

    def test_explicit_mlflow_port_honored(self):
        from dsagt.session import pick_free_port

        chosen = pick_free_port()
        _pdir, port = init_project("myproj", "goose", mlflow_port=chosen)
        assert port == chosen

    def test_duplicate_raises(self):
        init_project("myproj", "goose")
        with pytest.raises(FileExistsError):
            init_project("myproj", "goose")

    def test_invalid_agent_raises(self):
        with pytest.raises(ValueError):
            init_project("myproj", "invalid-agent")

    def test_static_record_written_eagerly(self, tmp_path):
        """BYOA flow writes static + dynamic records at init time so the
        user can edit instructions and inspect the MCP config artifact
        before launching their agent.  Static record is now driven by
        the CLI command, not init_project itself — but the agent dir
        layout exists post-init.
        """
        pdir, _ = init_project("myproj", "claude")
        config = load_config("myproj")
        assert config["agent"] == "claude"


# ---------------------------------------------------------------------------
# persist_agent_choice: first-start agent selection writes back to YAML
# ---------------------------------------------------------------------------


class TestPersistAgentChoice:
    """``persist_agent_choice`` is called from ``_cmd_start`` when the
    YAML doesn't already pin an agent — covers the case where the user
    deferred ``--agent`` from init time and supplied it on first start.
    """

    def test_overwrites_existing_field(self, tmp_path):
        init_project("myproj", "goose")
        persist_agent_choice("myproj", "claude")

        config = load_config("myproj")
        assert config["agent"] == "claude"

    def test_invalid_agent_raises(self, tmp_path):
        init_project("myproj", "goose")
        with pytest.raises(ValueError):
            persist_agent_choice("myproj", "not-an-agent")


# ---------------------------------------------------------------------------
# pick_free_port: kernel-assigned via socket.bind(("", 0))
# ---------------------------------------------------------------------------


class TestPickFreePort:

    def test_returns_a_usable_port(self):
        """The kernel hands back a positive port number we can bind."""
        from dsagt.session import pick_free_port
        import socket as _socket

        port = pick_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535
        # And we can actually bind it (no leak, no half-stuck socket).
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("", port))

    def test_consecutive_calls_pick_different_ports(self):
        """No memoization, no preferred-port bias — consecutive calls just
        reflect whatever the kernel hands out (almost always different on
        a quiet box, very rarely the same; this test tolerates that)."""
        from dsagt.session import pick_free_port

        ports = {pick_free_port() for _ in range(10)}
        # Not strictly guaranteed all 10 differ, but >1 distinct is expected.
        assert len(ports) > 1


# ---------------------------------------------------------------------------
# reap_runtime: SIGTERM PIDs in <project>/.runtime, grace, SIGKILL stragglers
# ---------------------------------------------------------------------------


class TestReapRuntime:

    def test_no_runtime_file_returns_empty(self, tmp_path):
        from dsagt.session import reap_runtime

        assert reap_runtime(tmp_path / ".runtime") == []

    def test_skips_pids_whose_cmdline_doesnt_match(self, tmp_path, monkeypatch):
        """PID-recycling guard: cmdline doesn't name our service → leave alone."""
        from dsagt import session

        runtime = tmp_path / ".runtime"
        runtime.write_text(
            json.dumps(
                {
                    "pids": {"mlflow": 99999, "proxy": 99998},
                    "ports": {"mlflow": 12345, "proxy": 12346},
                }
            )
        )
        # Pretend both PIDs are recycled — cmdline doesn't contain our name.
        monkeypatch.setattr(session, "_process_command", lambda pid: "/bin/zsh")
        killed = session.reap_runtime(runtime)
        assert killed == []
        # File still gets cleaned up so a stale .runtime doesn't linger.
        assert not runtime.exists()

    def test_signals_pids_whose_cmdline_matches(self, tmp_path, monkeypatch):
        """When the cmdline still names our service, SIGTERM is sent."""
        from dsagt import session

        runtime = tmp_path / ".runtime"
        runtime.write_text(
            json.dumps(
                {
                    "pids": {"mlflow": 11111, "proxy": 22222},
                    "ports": {"mlflow": 12345, "proxy": 12346},
                }
            )
        )
        monkeypatch.setattr(
            session,
            "_process_command",
            lambda pid: f"python -m {'mlflow' if pid == 11111 else 'dsagt.commands.proxy_server'}",
        )
        signals_sent: list[tuple[int, int]] = []

        def fake_killpg(pgid, sig):
            signals_sent.append((pgid, sig))

        monkeypatch.setattr(session.os, "killpg", fake_killpg)
        monkeypatch.setattr(session.os, "getpgid", lambda pid: pid)
        # Make grace-wait short and the process appear to exit immediately.
        monkeypatch.setattr(session, "_STOP_GRACE_SECONDS", 0)

        killed = session.reap_runtime(runtime)
        assert len(killed) == 2
        # Both PIDs got SIGTERM.
        import signal as _signal

        sigterm_pgids = {pgid for pgid, sig in signals_sent if sig == _signal.SIGTERM}
        assert sigterm_pgids == {11111, 22222}


# ---------------------------------------------------------------------------
# Per-agent record writers: static_agent_record + dynamic_agent_record
#
# Each test runs both writers against a project init'd with --agent,
# mirroring what `dsagt start` does in production.  Cline's dynamic
# writer would shell out to `cline auth` here, so cline gets exercised
# by ``test_cline_mcp_config_shape`` (mocked subprocess) instead of
# the full dynamic writer.
# ---------------------------------------------------------------------------


class TestAgentRecord:

    def _init_and_load(self, agent):
        init_project("testproj", agent)
        return load_config("testproj")

    def _write_both(self, config, working_dir):
        """Run static then dynamic — what dsagt start does, minus services.

        ``start_services`` populates ``config["mlflow"]["port"]`` in
        production; we stub it here so ``agent_env`` (which reads it to
        build the OTLP endpoint) doesn't KeyError.
        """
        config.setdefault("mlflow", {})["port"] = 5001
        static_agent_record(config, config["agent"], working_dir)
        env = agent_env(config)
        dynamic_agent_record(config, env, working_dir)

    def test_claude_writes_mcp_json(self, tmp_path):
        config = self._init_and_load("claude")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        mcp_path = working_dir / ".mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "dsagt-registry" in mcp["mcpServers"]
        assert "dsagt-knowledge" in mcp["mcpServers"]
        assert (working_dir / "CLAUDE.md").exists()
        # BYOA: .dsagt_env is no longer written; user manages shell env.
        assert not (working_dir / ".dsagt_env").exists()

    def test_goose_writes_goose_yaml(self, tmp_path):
        config = self._init_and_load("goose")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        goose_path = working_dir / "goose.yaml"
        assert goose_path.exists()
        goose = yaml.safe_load(goose_path.read_text())
        assert "registry" in goose["extensions"]
        assert "knowledge" in goose["extensions"]
        assert (working_dir / ".goosehints").exists()

    def test_roo_writes_static_and_dynamic(self, tmp_path):
        config = self._init_and_load("roo")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        assert (working_dir / ".roo").is_dir()
        assert (working_dir / ".roomodes").exists()
        assert (working_dir / ".roo" / "mcp.json").exists()
        # BYOA: project routing (project name, project_dir, mlflow port)
        # comes from dsagt_config.yaml via cwd-walk; the MCP env block
        # only carries EMBEDDING_* settings.
        mcp = json.loads((working_dir / ".roo" / "mcp.json").read_text())
        assert "EMBEDDING_BACKEND" in mcp["mcpServers"]["dsagt-registry"]["env"]

    def test_cline_writes_static_only_in_split_test(self, tmp_path):
        # Cline's dynamic writer shells out to `cline auth` and `cline mcp
        # add`, which would fail without cline installed.  Test only the
        # static half here; ``test_cline_mcp_config_shape`` covers the
        # dynamic half with mocked subprocess.
        config = self._init_and_load("cline")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        static_agent_record(config, config["agent"], working_dir)

        instructions = working_dir / ".clinerules" / "dsagt_instructions.md"
        assert instructions.exists()
        assert (working_dir / ".cline-data").is_dir()

    def test_codex_writes_static_and_dynamic(self, tmp_path):
        # Codex's .codex-data is rooted at working_dir.
        config = self._init_and_load("codex")
        working_dir = Path(config["project_dir"])

        self._write_both(config, working_dir)

        assert (working_dir / "AGENTS.md").exists()
        assert (working_dir / ".codex-data").is_dir()
        toml = (working_dir / ".codex-data" / "config.toml").read_text()
        assert "[mcp_servers.dsagt-registry.env]" in toml
        # Project routing comes from dsagt_config.yaml via cwd-walk; the
        # MCP env block only carries EMBEDDING_* settings.
        assert "EMBEDDING_BACKEND" in toml

    def test_static_is_idempotent(self, tmp_path):
        # Running static twice doesn't duplicate or destroy content —
        # the marker check skips the second write.  This is what lets
        # users edit CLAUDE.md / AGENTS.md between init and start
        # without losing edits.
        config = self._init_and_load("claude")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        static_agent_record(config, "claude", working_dir)
        first = (working_dir / "CLAUDE.md").read_text()
        # Simulate a user edit
        (working_dir / "CLAUDE.md").write_text(first + "\n\n## My project notes\nfoo")
        edited = (working_dir / "CLAUDE.md").read_text()
        # Re-run static — should be no-op since marker is present
        static_agent_record(config, "claude", working_dir)
        assert (working_dir / "CLAUDE.md").read_text() == edited

    def test_static_files_present_check(self, tmp_path):
        # Used by `dsagt start` to decide whether to call static_agent_record.
        config = self._init_and_load("codex")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        assert not static_agent_files_present("codex", working_dir)
        static_agent_record(config, "codex", working_dir)
        assert static_agent_files_present("codex", working_dir)

    def test_codex_config_toml_shape(self, tmp_path):
        """``_render_codex_config`` emits ``[mcp_servers.*]`` sections
        plus an ``[otel]`` block (opting codex's partial-OTel emission
        into MLflow via the OTLP exporter).  No top-level keys — those
        come from the user's ``~/.codex/config.toml`` which
        ``write_dynamic`` copies as a base.
        """
        from dsagt.agents import _render_codex_config

        mcp_env = {
            "DSAGT_PROJECT_DIR": "/proj",
            "MLFLOW_TRACKING_URI": "http://localhost:5001",
        }
        toml = _render_codex_config(mcp_env)

        assert "[mcp_servers.dsagt-registry]" in toml
        assert "[mcp_servers.dsagt-knowledge]" in toml
        assert "[mcp_servers.dsagt-registry.env]" in toml
        assert 'MLFLOW_TRACKING_URI = "http://localhost:5001"' in toml
        # OTel opt-in: enables OTLP-HTTP export + un-redacts user prompt.
        assert "[otel]" in toml
        assert 'trace_exporter = "otlp-http"' in toml
        assert "log_user_prompt = true" in toml
        # No top-level approval/sandbox keys — those come from user's
        # config.toml or the codex exec CLI flag.
        assert "approval_policy" not in toml
        assert "sandbox_mode" not in toml

    def test_opencode_config_json_shape(self):
        """``_render_opencode_config`` produces opencode.json with MCP
        servers + provider blocks using ``{env:VAR}`` interpolation —
        no actual creds on disk.  Provider blocks are emitted only for
        providers whose API key the user has set.
        """
        from dsagt.agents.opencode import _render_opencode_config

        mcp_env = {
            "DSAGT_PROJECT_DIR": "/proj",
            "MLFLOW_TRACKING_URI": "http://localhost:5001",
        }
        body = _render_opencode_config(
            mcp_env,
            present_creds={
                "OPENAI_API_KEY": True,
                "OPENAI_BASE_URL": True,
                "ANTHROPIC_API_KEY": False,
                "ANTHROPIC_BASE_URL": False,
            },
        )
        parsed = json.loads(body)

        assert parsed["$schema"] == "https://opencode.ai/config.json"
        assert set(parsed["mcp"]) == {"dsagt-registry", "dsagt-knowledge"}
        reg = parsed["mcp"]["dsagt-registry"]
        assert reg["type"] == "local"
        assert reg["command"] == ["uv", "run", "dsagt-registry-server"]
        assert reg["environment"]["DSAGT_PROJECT_DIR"] == "/proj"
        # Provider block uses {env:VAR} reference, never the resolved value.
        assert (
            parsed["provider"]["openai"]["options"]["apiKey"] == "{env:OPENAI_API_KEY}"
        )
        assert (
            parsed["provider"]["openai"]["options"]["baseURL"]
            == "{env:OPENAI_BASE_URL}"
        )
        # Anthropic block omitted because user didn't have the key set.
        assert "anthropic" not in parsed["provider"]

    def test_opencode_config_omits_provider_when_no_creds(self):
        """If the user has no provider creds set, opencode.json gets no
        provider block — opencode falls back to its own auth flow
        (``opencode auth login``)."""
        from dsagt.agents.opencode import _render_opencode_config

        body = _render_opencode_config({}, present_creds={})
        parsed = json.loads(body)
        assert "provider" not in parsed

    def test_opencode_registers_custom_model_under_provider(self):
        """Lab-gateway-aliased models like
        ``claude-haiku-4-5-20251001-v1-project`` aren't in models.dev's
        catalog, so opencode rejects them under standard providers
        unless declared explicitly in ``provider.<id>.models``.  At
        init we parse OPENCODE_MODEL and register the model there, plus
        set the top-level ``model`` for interactive convenience.
        """
        from dsagt.agents.opencode import _render_opencode_config

        body = _render_opencode_config(
            {},
            present_creds={"OPENAI_API_KEY": True, "OPENAI_BASE_URL": True},
            opencode_model="openai/claude-haiku-4-5-20251001-v1-project",
        )
        parsed = json.loads(body)
        assert parsed["model"] == "openai/claude-haiku-4-5-20251001-v1-project"
        assert (
            parsed["provider"]["openai"]["models"][
                "claude-haiku-4-5-20251001-v1-project"
            ]["name"]
            == "claude-haiku-4-5-20251001-v1-project"
        )

    def test_opencode_skips_model_registration_when_provider_absent(self):
        """If OPENCODE_MODEL names a provider whose API key isn't set,
        we don't emit a provider block to attach the model to.  Top-level
        ``model`` is also skipped so opencode doesn't error at startup
        on a model with no provider config."""
        from dsagt.agents.opencode import _render_opencode_config

        body = _render_opencode_config(
            {},
            present_creds={"OPENAI_API_KEY": True},  # only openai creds
            opencode_model="anthropic/claude-sonnet-4-5",  # but model is anthropic
        )
        parsed = json.loads(body)
        assert "model" not in parsed
        assert "anthropic" not in parsed.get("provider", {})

    def test_mcp_servers_dict_shape(self):
        from dsagt.agents import _build_mcp_servers_dict

        env_block = {
            "DSAGT_PROJECT_DIR": "/tmp/x",
            "MLFLOW_TRACKING_URI": "http://localhost:5001",
        }
        mcp = _build_mcp_servers_dict(env_block)

        assert set(mcp["mcpServers"]) == {"dsagt-registry", "dsagt-knowledge"}
        assert mcp["mcpServers"]["dsagt-knowledge"]["disabled"] is False
        # Env block plumbs through so the MCP server children have what they need.
        assert (
            mcp["mcpServers"]["dsagt-registry"]["env"]["MLFLOW_TRACKING_URI"]
            == "http://localhost:5001"
        )

    def test_mcp_config_omits_redundant_dsagt_env(self, tmp_path):
        """Single source of truth: project routing (project, project_dir,
        mlflow port) lives in dsagt_config.yaml and is read via cwd-walk
        by every dsagt service.  The MCP env block must NOT duplicate
        those values — that's the contract being asserted here."""
        config = self._init_and_load("claude")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        mcp = json.loads((working_dir / ".mcp.json").read_text())
        for server in ("dsagt-registry", "dsagt-knowledge"):
            env = mcp["mcpServers"][server].get("env", {})
            assert "DSAGT_PROJECT" not in env
            assert "DSAGT_PROJECT_DIR" not in env
            assert "MLFLOW_TRACKING_URI" not in env


# ---------------------------------------------------------------------------
# run.py: _resolve_records_dir with DSAGT_PROJECT_DIR
# ---------------------------------------------------------------------------


class TestResolveRecordsDirProjectAware:
    """``_resolve_records_dir`` reads the project's ``dsagt_config.yaml``
    from cwd as the single source of truth — no env-var chain."""

    def test_explicit_overrides_cwd(self, tmp_path):
        from dsagt.provenance import _resolve_records_dir

        result = _resolve_records_dir("/custom")
        assert result == Path("/custom")

    def test_cwd_with_config(self, tmp_path, monkeypatch):
        """When cwd contains ``dsagt_config.yaml``, records dir is
        ``<cwd>/trace_archive``.  Even if env vars are set to point
        elsewhere, the config-in-cwd rule wins (env is ignored)."""
        from dsagt.provenance import _resolve_records_dir

        (tmp_path / "dsagt_config.yaml").write_text("project: t\n")
        monkeypatch.chdir(tmp_path)
        # Stale env vars must not be consulted.
        monkeypatch.setenv("DSAGT_PROJECT_DIR", "/stale/proj/dir")
        monkeypatch.setenv("DSAGT_RECORDS_DIR", "/stale/records/dir")
        assert _resolve_records_dir(None) == tmp_path / "trace_archive"


# ---------------------------------------------------------------------------
# CLI: agent_env
# ---------------------------------------------------------------------------


class TestAgentEnv:

    def _make_config(self, agent, project_dir="/proj"):
        return {
            "project": "test",
            "agent": agent,
            "project_dir": project_dir,
            "mlflow": {"port": 5001},
            "llm": {"model": "test-model"},
            "embedding": {"api_key": "test-key"},
        }

    def test_dsagt_vars_set(self):
        from dsagt.agents import agent_env

        env = agent_env(self._make_config("claude"))
        assert env["DSAGT_PROJECT"] == "test"
        assert env["DSAGT_AGENT"] == "claude"
        assert env["DSAGT_PROJECT_DIR"] == "/proj"

    def test_claude_byoa_skips_otel_routing(self):
        """Claude in BYOA mode (no proxy) uses ``mlflow autolog claude``
        for agent-side traces — its Stop hook produces richer
        transcript-based traces than native OTel.  agent_env should NOT
        export OTEL_EXPORTER_OTLP_* vars for Claude in BYOA, to avoid
        duplicate (and inferior) trace shapes.
        """
        from dsagt.agents import agent_env

        env = agent_env(self._make_config("claude"))
        assert env["MLFLOW_TRACKING_URI"] == "http://localhost:5001"
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in env
        assert "OTEL_RESOURCE_ATTRIBUTES" not in env

    def test_non_claude_byoa_keeps_otel_routing(self):
        """Goose has no ``mlflow autolog goose`` analog — it still gets
        OTel routing so its native OTel emission lands in MLflow.
        """
        from dsagt.agents import agent_env

        env = agent_env(self._make_config("goose"))
        assert env["MLFLOW_TRACKING_URI"] == "http://localhost:5001"
        assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5001/v1/traces"
        assert "service.name=goose" in env["OTEL_RESOURCE_ATTRIBUTES"]

    def test_claude_telemetry_verbosity_flags(self):
        """Claude Code needs OTEL_LOG_TOOL_DETAILS + OTEL_LOG_USER_PROMPTS
        to emit unredacted tool_use payloads + user prompts memory
        extraction depends on.  OTEL_LOG_RAW_API_BODIES is intentionally
        NOT in the static env block — its value is a per-project file
        path rendered dynamically by ``_cmd_mlflow`` (see claude.py for
        why ``=1`` mode would lose bodies to MLflow's missing /v1/logs
        endpoint)."""
        from dsagt.agents import agent_env

        env = agent_env(self._make_config("claude"))
        assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
        assert env["OTEL_LOG_TOOL_DETAILS"] == "1"
        assert env["OTEL_LOG_USER_PROMPTS"] == "1"
        assert "OTEL_LOG_RAW_API_BODIES" not in env


@pytest.mark.skip(
    reason=(
        "Old-code-shape env_overrides: assertions describe the pre-Phase-1 "
        "design where env_overrides translated llm.* into ANTHROPIC_*/OPENAI_* "
        "credential env vars across the board.  Phase 2's env_overrides is "
        "narrower — it pins the agent's MODEL env var only; provider creds "
        "and base URLs are set by proxy_env_overrides (sentinel + proxy URL) "
        "or live in agent config files (cline auth state, codex config.toml, "
        "opencode.json).  See ``TestPhase2EnvOverrides`` below for the new "
        "contract.  Keeping these tests around for the LoC reference; "
        "delete when Phase 2 is stable."
    )
)
class TestProviderEnvInjection:
    """``llm.{provider, model, api_key, base_url}`` from dsagt_config must
    flow into the agent's expected env-var names — otherwise the agent
    falls back to its own user-config and the project YAML's settings
    don't actually drive the agent.  This was a real bug for goose."""

    def _config_with_llm(self, **llm) -> dict:
        return {
            "project": "test",
            "agent": "goose",
            "project_dir": "/proj",
            "mlflow": {"port": 5001},
            "llm": llm,
            "embedding": {},
        }

    def test_openai_provider_sets_openai_env(self):
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="openai",
                model="gpt-4o",
                api_key="sk-real",
                base_url="https://gateway.example.com",
            )
        )
        assert env["OPENAI_API_KEY"] == "sk-real"
        assert env["OPENAI_BASE_URL"] == "https://gateway.example.com"
        assert env["GOOSE_PROVIDER"] == "openai"
        assert env["GOOSE_MODEL"] == "gpt-4o"
        # Goose's Rust openai client reads OPENAI_HOST, not OPENAI_BASE_URL.
        # Without this, goose silently hits api.openai.com regardless of
        # the project's gateway.
        assert env["OPENAI_HOST"] == "https://gateway.example.com"

    def test_goose_anthropic_sets_anthropic_host(self):
        """Same HOST-vs-BASE_URL issue for goose's anthropic provider."""
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="anthropic",
                model="claude-sonnet",
                api_key="anthropic-key",
                base_url="https://api.anthropic.com",
            )
        )
        assert env["ANTHROPIC_HOST"] == "https://api.anthropic.com"

    @pytest.mark.skip(reason="proxy mode deferred to Phase 2")
    def test_goose_proxy_sets_host_at_proxy(self):
        """Proxy mode: OPENAI_HOST / ANTHROPIC_HOST must point at the proxy
        too, not just the standard BASE_URL slots."""
        from dsagt.agents import agent_env

        config = self._config_with_llm(
            provider="openai",
            api_key="sk-real",
            base_url="https://upstream.example.com",
        )
        config["proxy"] = {"port": 9999}
        env = agent_env(config)
        assert env["OPENAI_HOST"] == "http://localhost:9999"
        assert env["ANTHROPIC_HOST"] == "http://localhost:9999"

    def test_openai_like_provider_treated_as_openai(self):
        """openai_like (lab gateways speaking OpenAI wire protocol) maps
        to OPENAI_* env vars too — that's what every OpenAI-compat client
        reads."""
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="openai_like",
                api_key="sk-lab",
                base_url="https://lab.example.com",
            )
        )
        assert env["OPENAI_API_KEY"] == "sk-lab"
        assert env["OPENAI_BASE_URL"] == "https://lab.example.com"

    def test_anthropic_provider_sets_anthropic_env(self):
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="anthropic",
                model="claude-sonnet-4-5",
                api_key="anthropic-key",
                base_url="https://api.anthropic.com",
            )
        )
        assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
        assert env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
        # ANTHROPIC_MODEL pins the model so claude code doesn't fall back
        # to its built-in default (which won't exist on lab gateways).
        assert env["ANTHROPIC_MODEL"] == "claude-sonnet-4-5"

    def test_cline_gets_provider_env(self):
        """Cline's CLI reads OPENAI_*/ANTHROPIC_* — must flow from llm.*."""
        from dsagt.agents import agent_env

        cfg = self._config_with_llm(
            provider="openai",
            api_key="sk-cline",
            base_url="https://gateway.example.com",
        )
        cfg["agent"] = "cline"
        env = agent_env(cfg)
        assert env["OPENAI_API_KEY"] == "sk-cline"
        assert env["OPENAI_BASE_URL"] == "https://gateway.example.com"

    def test_roo_gets_provider_env(self):
        """Roo's CLI reads ANTHROPIC_* — must flow from llm.*."""
        from dsagt.agents import agent_env

        cfg = self._config_with_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key="anthropic-roo-key",
            base_url="https://api.anthropic.com",
        )
        cfg["agent"] = "roo"
        env = agent_env(cfg)
        assert env["ANTHROPIC_API_KEY"] == "anthropic-roo-key"
        assert env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
        assert env["ANTHROPIC_MODEL"] == "claude-haiku-4-5"

    def test_unresolved_placeholder_skipped(self):
        """``${VAR}`` (unresolved interpolation) must NOT be propagated —
        we only inject real values."""
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="openai",
                api_key="${LLM_API_KEY}",
                base_url="${LLM_BASE_URL}",
            )
        )
        # Provider still set, but bogus key/url not injected.
        assert env["GOOSE_PROVIDER"] == "openai"
        assert env.get("OPENAI_API_KEY") != "${LLM_API_KEY}"
        assert env.get("OPENAI_BASE_URL") != "${LLM_BASE_URL}"

    def test_blank_values_skipped(self):
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="openai",
                api_key="",
                base_url="   ",
            )
        )
        # Blank values shouldn't override the user's shell-level env vars
        # (which the os.environ copy in agent_env preserves).
        assert env.get("OPENAI_API_KEY") != ""
        assert env.get("OPENAI_BASE_URL") != "   "

    def test_unknown_provider_still_sets_goose_vars(self):
        """Goose can dispatch to unknown providers via its own config
        fallback; we still set GOOSE_PROVIDER / GOOSE_MODEL so the agent
        knows which provider to look up."""
        from dsagt.agents import agent_env

        env = agent_env(
            self._config_with_llm(
                provider="bedrock",
                model="anthropic.claude-3-sonnet",
            )
        )
        assert env["GOOSE_PROVIDER"] == "bedrock"
        assert env["GOOSE_MODEL"] == "anthropic.claude-3-sonnet"
        # No OPENAI_* / ANTHROPIC_* injection for unknown providers — user
        # supplies those via their shell env (e.g. AWS_ACCESS_KEY_ID).
        assert "OPENAI_API_KEY" not in env or env["OPENAI_API_KEY"] != ""

    @pytest.mark.skip(reason="proxy mode deferred to Phase 2")
    def test_proxy_overrides_provider_injection(self):
        """When --enable-proxy is set, the proxy block runs AFTER provider
        injection and overrides OPENAI_BASE_URL / OPENAI_API_KEY with the
        proxy URL + sentinel.  Provider injection must not stomp the proxy."""
        from dsagt.agents import agent_env

        config = self._config_with_llm(
            provider="openai",
            api_key="sk-real",
            base_url="https://upstream.example.com",
        )
        config["proxy"] = {"port": 9999}
        env = agent_env(config)
        assert env["OPENAI_BASE_URL"] == "http://localhost:9999"
        assert env["OPENAI_API_KEY"].startswith("dsagt-proxy-forwarded")

    def test_claude_does_not_get_openai_env(self):
        """Protocol isolation: claude is anthropic-native.  When the user
        configures provider=openai, claude's env_overrides must NOT
        propagate OPENAI_* — claude's runtime ignores them and the cross-
        protocol mismatch is what the proxy is for.  Setting them anyway
        muddies the contract."""
        from dsagt.agents import agent_env

        cfg = self._config_with_llm(
            provider="openai",
            api_key="sk-real",
            base_url="https://gateway.example.com",
        )
        cfg["agent"] = "claude"
        env = agent_env(cfg)
        # Claude is openai-blind; project openai creds shouldn't show up
        # in claude's process env (the user's shell may set them, in
        # which case os.environ wins — that's a user-controlled choice).
        # We assert ours specifically isn't injected.
        assert env.get("OPENAI_BASE_URL") != "https://gateway.example.com"

    def test_codex_does_not_get_anthropic_env(self):
        """Protocol isolation: codex is openai-native.  Anthropic upstream
        requires --enable-proxy for translation."""
        from dsagt.agents import agent_env

        cfg = self._config_with_llm(
            provider="anthropic",
            model="claude-sonnet",
            api_key="anthropic-key",
            base_url="https://api.anthropic.com",
        )
        cfg["agent"] = "codex"
        env = agent_env(cfg)
        assert env.get("ANTHROPIC_BASE_URL") != "https://api.anthropic.com"


class TestPhase2EnvOverrides:
    """Phase 2 ``env_overrides`` is narrower than old code: it pins the
    agent's MODEL env var only.  Provider creds / base URLs are set by
    ``proxy_env_overrides`` (sentinel + proxy URL) or live in agent
    config files (cline auth state, codex config.toml, opencode.json).
    All gated by ``config["proxy"]["port"]`` — BYOA mode doesn't fire it.
    """

    def _proxy_config(self, agent: str, model: str = "test-model") -> dict:
        return {
            "project": "test",
            "agent": agent,
            "project_dir": "/proj",
            "mlflow": {"port": 5001},
            "llm": {
                "provider": "openai",
                "model": model,
                "base_url": "https://up",
                "api_key": "sk-up",
            },
            "embedding": {},
            "proxy": {"port": 9999},
        }

    def test_claude_pins_anthropic_model_in_proxy_mode(self):
        from dsagt.agents import agent_env

        env = agent_env(self._proxy_config("claude", model="claude-test"))
        assert env["ANTHROPIC_MODEL"] == "claude-test"

    def test_goose_pins_provider_and_model_in_proxy_mode(self):
        from dsagt.agents import agent_env

        env = agent_env(self._proxy_config("goose", model="my-model"))
        assert env["GOOSE_PROVIDER"] == "openai"
        assert env["GOOSE_MODEL"] == "my-model"

    def test_roo_pins_anthropic_model_in_proxy_mode(self):
        from dsagt.agents import agent_env

        env = agent_env(self._proxy_config("roo", model="my-roo-model"))
        assert env["ANTHROPIC_MODEL"] == "my-roo-model"

    def test_byoa_mode_does_not_fire_env_overrides(self):
        """Without proxy_port in config, env_overrides is not called —
        the user's shell ANTHROPIC_MODEL / GOOSE_MODEL etc. are not
        overwritten by config["llm"] values.  This is the gate that
        prevents the Phase-1 GOOSE_MODEL bug from coming back.
        """
        from dsagt.agents import agent_env

        cfg = self._proxy_config("goose", model="config-model")
        cfg.pop("proxy")  # BYOA: no proxy block
        env = agent_env(cfg)
        # GOOSE_MODEL only set if user's shell has it; we don't override.
        assert env.get("GOOSE_MODEL") != "config-model"

    def test_unresolved_placeholder_filtered(self):
        """A ${VAR} that didn't resolve at load_config time must not
        leak into env vars (would be an obvious bug surface)."""
        from dsagt.agents import agent_env

        cfg = self._proxy_config("goose", model="${LLM_MODEL}")
        env = agent_env(cfg)
        assert "GOOSE_MODEL" not in env or env["GOOSE_MODEL"] != "${LLM_MODEL}"


class TestProxyEnvOverrides:
    """``--enable-proxy`` plants the same proxy-routing env vars on
    every agent.  AgentSetup.proxy_env_overrides centralizes the
    contract; agents inherit the default unless they have a non-
    standard env-var convention (goose's OPENAI_HOST/ANTHROPIC_HOST)."""

    def _make_config(self, agent: str) -> dict:
        return {
            "project": "test",
            "agent": agent,
            "project_dir": "/proj",
            "mlflow": {"port": 5001},
            "llm": {"provider": "openai", "api_key": "sk-up", "base_url": "https://up"},
            "embedding": {},
            "proxy": {"port": 9999},
        }

    def test_default_sets_proxy_url_for_both_protocols(self):
        from dsagt.agents.base import AgentSetup

        # Spin up any concrete subclass to call the inherited default.
        from dsagt.agents.goose import GooseSetup

        env = GooseSetup().proxy_env_overrides(9999)
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:9999"
        assert env["OPENAI_BASE_URL"] == "http://localhost:9999"
        assert env["EMBEDDING_BASE_URL"] == "http://localhost:9999"

    def test_default_plants_sentinel_keys(self):
        from dsagt.agents.goose import GooseSetup

        env = GooseSetup().proxy_env_overrides(9999)
        assert env["ANTHROPIC_API_KEY"].startswith("dsagt-proxy-forwarded")
        assert env["OPENAI_API_KEY"].startswith("dsagt-proxy-forwarded")
        assert env["EMBEDDING_API_KEY"].startswith("dsagt-proxy-forwarded")

    def test_proxy_applied_uniformly_across_agents(self):
        """Every agent gets identical proxy env from the inherited
        default — that's the whole point of putting it on the base class."""
        from dsagt.agents import agent_env

        envs = {
            a: agent_env(self._make_config(a))
            for a in ("claude", "goose", "cline", "roo", "codex")
        }
        for agent, env in envs.items():
            assert env["ANTHROPIC_BASE_URL"] == "http://localhost:9999", agent
            assert env["OPENAI_BASE_URL"] == "http://localhost:9999", agent
            assert env["OPENAI_API_KEY"].startswith("dsagt-proxy-forwarded"), agent


class TestPreconfiguredCredsWarning:
    """When the project YAML has no llm credentials, ``agent_env`` warns
    that the agent will fall back to the user's shell env, listing which
    of the agent's credential env vars are actually present."""

    def _config(self, agent: str = "claude", **llm) -> dict:
        return {
            "project": "test",
            "agent": agent,
            "project_dir": "/proj",
            "mlflow": {"port": 5001},
            "llm": llm,
            "embedding": {},
        }

    def test_warns_when_project_has_no_creds_but_shell_does(self, caplog, monkeypatch):
        from dsagt.agents import agent_env

        monkeypatch.setenv("ANTHROPIC_API_KEY", "shell-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

        with caplog.at_level("WARNING", logger="dsagt.agents"):
            agent_env(self._config(agent="claude"))

        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "preconfigured env vars" in msgs
        assert "ANTHROPIC_API_KEY" in msgs
        assert "ANTHROPIC_BASE_URL" in msgs
        # Var names only, never values.
        assert "shell-key" not in msgs

    def test_no_warning_when_project_supplies_creds(self, caplog):
        from dsagt.agents import agent_env

        cfg = self._config(
            agent="claude",
            provider="anthropic",
            api_key="project-key",
            base_url="https://api.anthropic.com",
            model="claude-haiku",
        )

        with caplog.at_level("WARNING", logger="dsagt.agents"):
            agent_env(cfg)

        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "preconfigured env vars" not in msgs

    def test_no_warning_when_proxy_enabled(self, caplog):
        """Proxy mode plants its own creds; transparency warning is moot."""
        from dsagt.agents import agent_env

        cfg = self._config(agent="claude")
        cfg["proxy"] = {"port": 9999}

        with caplog.at_level("WARNING", logger="dsagt.agents"):
            agent_env(cfg)

        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "preconfigured env vars" not in msgs

    def test_warns_with_no_shell_either_lists_expected_vars(self, caplog, monkeypatch):
        """When neither project nor shell has creds, surface the var names
        the agent's runtime expects so the user can fix the gap."""
        from dsagt.agents import agent_env

        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"):
            monkeypatch.delenv(var, raising=False)

        with caplog.at_level("WARNING", logger="dsagt.agents"):
            agent_env(self._config(agent="claude"))

        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "fall back to its own auth flow" in msgs
        assert "ANTHROPIC_API_KEY" in msgs


# ---------------------------------------------------------------------------
# CLI: agent_command
# ---------------------------------------------------------------------------


class TestAgentCommand:

    def test_claude(self):
        from dsagt.agents import agent_command

        assert agent_command({"agent": "claude"}) == ["claude"]

    def test_goose(self):
        from dsagt.agents import agent_command

        assert agent_command({"agent": "goose"}) == [
            "goose",
            "session",
            "--with-extension",
            "uv run dsagt-registry-server",
            "--with-extension",
            "uv run dsagt-knowledge-server",
        ]

    def test_roo(self):
        from dsagt.agents import agent_command

        assert agent_command({"agent": "roo"}) == ["roo"]

    def test_cline(self):
        from dsagt.agents import agent_command

        assert agent_command({"agent": "cline"}) == ["cline"]

    def test_codex(self):
        from dsagt.agents import agent_command

        assert agent_command({"agent": "codex"}) == ["codex"]


# ---------------------------------------------------------------------------
# Config flow: embedding config propagation
# ---------------------------------------------------------------------------


class TestConfigFlow:

    def test_default_config_has_embedding_block(self):
        """default_config_content() includes embedding (KB needs it)."""
        content = default_config_content("test", "claude", mlflow_port=5000)
        parsed = yaml.safe_load(content)
        assert "embedding" in parsed
        assert "backend" in parsed["embedding"]

    def test_mcp_env_block_carries_embedding_routing(self):
        """_mcp_env_block plumbs embedding routing (model + base_url)
        through to MCP server children.  EMBEDDING_API_KEY is NOT baked
        in — it lives in the user's shell env (set when launching the
        agent) so credentials never land in on-disk artifacts."""
        from dsagt.agents import _mcp_env_block

        config = {
            "project": "test",
            "project_dir": "/p",
            "mlflow": {"port": 5000},
            "embedding": {
                "backend": "api",
                "base_url": "https://api.test/v1",
                "model": "m",
            },
        }
        env = _mcp_env_block(config)
        assert env["EMBEDDING_BASE_URL"] == "https://api.test/v1"
        assert env["EMBEDDING_BACKEND"] == "api"
        assert env["EMBEDDING_MODEL"] == "m"
        # Credentials never land in artifacts; user sets in shell.
        assert "EMBEDDING_API_KEY" not in env

    def test_mcp_env_block_omits_project_routing(self):
        """Single source of truth: MLflow URI / project name / project_dir
        come from dsagt_config.yaml via cwd-walk, not from a duplicated
        env block.  The MCP env block carries only embedding-backend
        settings (and EMBEDDING_API_KEY from the user's shell)."""
        from dsagt.agents import _mcp_env_block

        config = {
            "project": "test",
            "project_dir": "/p",
            "mlflow": {"port": 12345},
            "embedding": {"backend": "local"},
        }
        env = _mcp_env_block(config)
        assert "MLFLOW_TRACKING_URI" not in env
        assert "DSAGT_PROJECT" not in env
        assert "DSAGT_PROJECT_DIR" not in env
        assert env["EMBEDDING_BACKEND"] == "local"

    def test_mcp_env_block_omits_empty_embedding_keys(self):
        """Local-backend embedding has no base_url / model — those keys
        should be absent from the env block, not present-but-blank."""
        from dsagt.agents import _mcp_env_block

        config = {
            "project": "test",
            "project_dir": "/p",
            "mlflow": {"port": 5000},
            "embedding": {"backend": "local"},
        }
        env = _mcp_env_block(config)
        assert env["EMBEDDING_BACKEND"] == "local"
        assert "EMBEDDING_BASE_URL" not in env
        assert "EMBEDDING_MODEL" not in env

    def test_mcp_server_args_are_just_command(self):
        """MCP server args are just ["run", "dsagt-<name>-server"].
        All configuration flows through env vars and dsagt_config.yaml.
        """
        from dsagt.agents import _mcp_server_args

        assert _mcp_server_args("knowledge") == ["run", "dsagt-knowledge-server"]
        assert _mcp_server_args("registry") == ["run", "dsagt-registry-server"]

    def test_mcp_env_block_carries_only_embedding_settings(self):
        from dsagt.agents import _mcp_env_block

        config = {
            "project": "test",
            "project_dir": "/home/user/dsagt-projects/test",
            "mlflow": {"port": 5000},
            "embedding": {"model": "m", "base_url": "u"},
        }
        env = _mcp_env_block(config)
        # Project routing comes from dsagt_config.yaml via cwd-walk —
        # never duplicated into the MCP env block.
        assert "DSAGT_PROJECT_DIR" not in env
        assert "DSAGT_PROJECT" not in env
        assert env["EMBEDDING_MODEL"] == "m"
        assert env["EMBEDDING_BASE_URL"] == "u"


# ---------------------------------------------------------------------------
# BYOA: per-agent env hints + launch one-liners surfaced by `dsagt init`
# ---------------------------------------------------------------------------


class TestByoaEnvHints:
    """``dsagt init`` prints provider credentials only.  Internal env
    (DSAGT_*, MLFLOW_*, OTEL_*, telemetry verbosity flags) goes into
    the per-project launch shim — the user's shell stays clean."""

    @pytest.mark.parametrize(
        "agent_name", ["claude", "goose", "cline", "roo", "codex", "opencode"]
    )
    def test_returns_only_credential_hints(self, agent_name, tmp_path):
        from dsagt.agents import AGENTS

        setup = AGENTS[agent_name]()
        hints = setup.byoa_env_hints(
            mlflow_port=5001, project="p", project_dir=tmp_path
        )
        # Only credential hints come back; no DSAGT/MLflow/OTel routing.
        names = [n for n, _ in hints]
        assert "DSAGT_PROJECT" not in names
        assert "MLFLOW_TRACKING_URI" not in names
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in names

    @pytest.mark.parametrize(
        "agent_name", ["claude", "goose", "cline", "roo", "codex", "opencode"]
    )
    def test_credentials_match_credential_hints(self, agent_name, tmp_path):
        from dsagt.agents import AGENTS

        setup = AGENTS[agent_name]()
        hints = setup.byoa_env_hints(5001, "p", tmp_path)
        assert hints == list(setup.credential_hints)

    def test_goose_credentials_include_host_not_base_url(self, tmp_path):
        """Goose's Rust client reads OPENAI_HOST / ANTHROPIC_HOST, NOT
        OPENAI_BASE_URL — surfacing this gotcha to the user up front
        avoids the silent 'agent hits api.openai.com regardless of
        gateway' bug."""
        from dsagt.agents import AGENTS

        names = [n for n, _ in AGENTS["goose"]().byoa_env_hints(5001, "p", tmp_path)]
        assert "OPENAI_HOST" in names
        assert "ANTHROPIC_HOST" in names

    @pytest.mark.parametrize(
        "agent_name,gateway_var",
        [
            ("claude", "ANTHROPIC_BASE_URL"),
            ("codex", "OPENAI_BASE_URL"),
            # Cline / roo: ``cline auth -b`` is openai-only and openai-native
            # path needs a non-standard model env var, so we standardize on
            # the anthropic path for both — same env conventions as the rest
            # of the BYOA story.  See cline.py / roo.py credential_hints.
            ("cline", "ANTHROPIC_BASE_URL"),
            ("roo", "ANTHROPIC_BASE_URL"),
            # opencode emits both — its provider config supports both wire
            # protocols via {env:VAR} interpolation in opencode.json.
            ("opencode", "OPENAI_BASE_URL"),
            ("opencode", "ANTHROPIC_BASE_URL"),
        ],
    )
    def test_gateway_url_in_credential_hints(self, agent_name, gateway_var, tmp_path):
        """Lab gateway / proxy URL hint is surfaced for every agent that
        speaks the standard BASE_URL convention."""
        from dsagt.agents import AGENTS

        names = [n for n, _ in AGENTS[agent_name]().byoa_env_hints(5001, "p", tmp_path)]
        assert gateway_var in names


class TestLaunchOneliner:
    """``launch_oneliner`` returns the literal agent command (no shim).
    The shim is a separate file (``dsagt-launch.sh``) written by
    ``dynamic_agent_record`` — see TestLaunchShim below."""

    @pytest.mark.parametrize(
        "agent_name", ["claude", "goose", "cline", "roo", "codex", "opencode"]
    )
    def test_oneliner_does_not_invoke_shim(self, agent_name, tmp_path):
        from dsagt.agents import AGENTS

        cmd = AGENTS[agent_name]().launch_oneliner("myproj", tmp_path)
        assert "dsagt-launch.sh" not in cmd

    @pytest.mark.parametrize(
        "agent_name,expected_cmd",
        [
            ("claude", "claude"),
            ("goose", "goose session"),
            ("roo", "roo"),
            ("opencode", "opencode"),
        ],
    )
    def test_oneliner_runs_agent_directly(self, agent_name, expected_cmd, tmp_path):
        from dsagt.agents import AGENTS

        cmd = AGENTS[agent_name]().launch_oneliner("myproj", tmp_path)
        assert expected_cmd in cmd
        assert f"cd {tmp_path}" in cmd

    def test_cline_oneliner_includes_config_flag(self, tmp_path):
        """Cline's ``--config <pdir>/.cline-data`` is the only way to
        pick up the per-project MCP-server registrations."""
        from dsagt.agents import AGENTS

        cmd = AGENTS["cline"]().launch_oneliner("myproj", tmp_path)
        assert "--config" in cmd
        assert ".cline-data" in cmd

    def test_codex_oneliner_sets_codex_home(self, tmp_path):
        """Codex has no ``--config`` flag — must export ``CODEX_HOME``
        before launching so codex finds the per-project config.toml."""
        from dsagt.agents import AGENTS

        cmd = AGENTS["codex"]().launch_oneliner("myproj", tmp_path)
        assert "CODEX_HOME=" in cmd
        assert ".codex-data" in cmd


# ---------------------------------------------------------------------------
# dsagt memory: high-water-mark extraction state
# ---------------------------------------------------------------------------


class TestMemoryWatermark:
    """``dsagt memory --project X`` tracks which sessions have been
    extracted in ``<pdir>/.dsagt/extracted_at.json`` so re-runs only
    process new traces."""

    def test_first_run_creates_watermark_on_success(self, tmp_path, monkeypatch):
        from argparse import Namespace
        from dsagt.commands.cli import _cmd_memory

        pdir, _ = init_project("mem-test", "claude")
        monkeypatch.setattr(
            "dsagt.commands.cli.run_extraction",
            lambda _project: {"status": "ok", "total_entries": 3},
        )
        _cmd_memory(Namespace(project="mem-test"))

        state_path = pdir / ".dsagt" / "extracted_at.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "last_extracted_at" in state
        assert state["previous"] is None

    def test_subsequent_run_records_previous_watermark(self, tmp_path, monkeypatch):
        from argparse import Namespace
        from dsagt.commands.cli import _cmd_memory

        pdir, _ = init_project("mem-test", "claude")
        monkeypatch.setattr(
            "dsagt.commands.cli.run_extraction",
            lambda _project: {"status": "ok", "total_entries": 1},
        )
        _cmd_memory(Namespace(project="mem-test"))
        first_state = json.loads((pdir / ".dsagt" / "extracted_at.json").read_text())

        _cmd_memory(Namespace(project="mem-test"))
        second_state = json.loads((pdir / ".dsagt" / "extracted_at.json").read_text())

        # The previous run's mark moves into the "previous" slot.
        assert second_state["previous"] == first_state["last_extracted_at"]
        assert second_state["last_extracted_at"] != first_state["last_extracted_at"]

    def test_empty_extraction_does_not_advance_watermark(self, tmp_path, monkeypatch):
        """When run_extraction returns status=empty, the state file is
        not written — re-running later still picks up new traces."""
        from argparse import Namespace
        from dsagt.commands.cli import _cmd_memory

        pdir, _ = init_project("mem-test", "claude")
        monkeypatch.setattr(
            "dsagt.commands.cli.run_extraction",
            lambda _project: {"status": "empty"},
        )
        _cmd_memory(Namespace(project="mem-test"))

        assert not (pdir / ".dsagt" / "extracted_at.json").exists()


class TestLaunchShim:
    """``dynamic_agent_record`` writes ``dsagt-launch.sh`` for BYOA-mode
    projects.  The shim starts MLflow, exports env, and prints how to
    launch the agent (it does NOT exec the agent — user picks)."""

    def _make_config(self, agent_name: str, pdir):
        return {
            "project": "test",
            "project_dir": str(pdir),
            "agent": agent_name,
            "mlflow": {"port": 5099},
            "embedding": {},
            "llm": {},
            "session_id": "sess-xyz",
        }

    def test_shim_written_for_byoa(self, tmp_path):
        """`dynamic_agent_record` writes dsagt-launch.sh in BYOA mode."""
        from dsagt.agents import dynamic_agent_record

        config = self._make_config("goose", tmp_path)
        dynamic_agent_record(config, env={}, working_dir=tmp_path)

        shim = tmp_path / "dsagt-launch.sh"
        assert shim.exists()
        assert (shim.stat().st_mode & 0o111) != 0  # executable

    def test_shim_does_not_exec_agent(self, tmp_path):
        """Shim prints launch options instead of execing — user picks."""
        from dsagt.agents import dynamic_agent_record

        config = self._make_config("goose", tmp_path)
        dynamic_agent_record(config, env={}, working_dir=tmp_path)

        shim_text = (tmp_path / "dsagt-launch.sh").read_text()
        assert "exec " not in shim_text
        assert "Environment ready. Launch the agent" in shim_text
        assert "goose session" in shim_text  # CLI option printed

    def test_shim_starts_mlflow_in_background(self, tmp_path):
        from dsagt.agents import dynamic_agent_record

        config = self._make_config("goose", tmp_path)
        dynamic_agent_record(config, env={}, working_dir=tmp_path)

        shim_text = (tmp_path / "dsagt-launch.sh").read_text()
        assert "dsagt mlflow test --background-only" in shim_text

    def test_shim_for_claude_skips_otel_routing(self, tmp_path):
        """Claude shim omits OTEL_EXPORTER_OTLP_* exports — agent-side
        traces come from `mlflow autolog claude`'s Stop hook."""
        from dsagt.agents import dynamic_agent_record

        config = self._make_config("claude", tmp_path)
        dynamic_agent_record(config, env={}, working_dir=tmp_path)

        shim_text = (tmp_path / "dsagt-launch.sh").read_text()
        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in shim_text
        assert "MLFLOW_TRACKING_URI" in shim_text  # still exported

    def test_shim_for_goose_includes_otel_routing(self, tmp_path):
        """Non-Claude agents need OTel routing (no autolog analog)."""
        from dsagt.agents import dynamic_agent_record

        config = self._make_config("goose", tmp_path)
        dynamic_agent_record(config, env={}, working_dir=tmp_path)

        shim_text = (tmp_path / "dsagt-launch.sh").read_text()
        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" in shim_text
        assert "service.name=goose" in shim_text


class TestClaudeAutologSetup:
    """`dsagt init --agent claude` configures `mlflow autolog claude`
    by writing `.claude/settings.json` with the Stop hook + tracking
    env vars."""

    def test_settings_file_written(self, tmp_path):
        from dsagt.agents.claude import ClaudeSetup

        config = {
            "project": "myproj",
            "project_dir": str(tmp_path),
            "agent": "claude",
            "mlflow": {"port": 5099},
            "embedding": {},
            "llm": {},
        }
        actions = ClaudeSetup().write_dynamic(
            config,
            env={},
            working_dir=tmp_path,
            pdir=tmp_path,
        )

        settings_file = tmp_path / ".claude" / "settings.json"
        assert settings_file.exists()
        assert any("mlflow autolog claude" in a for a in actions)

        import json as _json

        settings = _json.loads(settings_file.read_text())
        env_block = settings.get("env") or {}
        assert env_block.get("MLFLOW_TRACKING_URI") == "http://localhost:5099"
        assert env_block.get("MLFLOW_EXPERIMENT_NAME") == "myproj"
        assert env_block.get("MLFLOW_CLAUDE_TRACING_ENABLED") == "true"

        hooks = settings.get("hooks") or {}
        stop_hooks = hooks.get("Stop") or []
        # Stop hook should reference mlflow autolog claude.
        all_commands = []
        for group in stop_hooks:
            for h in group.get("hooks") or []:
                if h.get("command"):
                    all_commands.append(h["command"])
        assert any("mlflow autolog claude" in c for c in all_commands)

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
        name = self._write_config(tmp_path, "myproject", {
            "project": "myproject",
            "agent": "goose",
            "llm": {"provider": "openai"},
        })

        config = load_config(name)

        assert config["project"] == "myproject"
        assert config["agent"] == "goose"
        assert config["proxy"]["port"] == 4000  # default
        assert config["mlflow"]["port"] == 5001  # default
        assert config["llm"]["model"] == "claude-sonnet-4-20250514"

    def test_overrides_defaults(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {
            "project": "myproject",
            "agent": "claude-code",
            "llm": {"provider": "openai"},
            "proxy": {"port": 9000},
        })

        config = load_config(name)
        assert config["proxy"]["port"] == 9000

    def test_missing_project_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {"agent": "goose"})
        with pytest.raises(ValueError, match="project"):
            load_config(name)

    def test_missing_agent_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {"project": "myproject"})
        with pytest.raises(ValueError, match="agent"):
            load_config(name)

    def test_invalid_agent_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {"project": "myproject", "agent": "copilot"})
        with pytest.raises(ValueError, match="copilot"):
            load_config(name)

    def test_missing_provider_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {
            "project": "myproject",
            "agent": "goose",
        })
        with pytest.raises(ValueError, match="llm.provider"):
            load_config(name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent")

    def test_project_dir_injected(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {
            "project": "myproject",
            "agent": "goose",
            "llm": {"provider": "openai"},
        })
        config = load_config(name)
        assert config["project_dir"] == str(tmp_path / "myproject")


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
        content = default_config_content("test", "goose")
        parsed = yaml.safe_load(content)
        assert parsed["project"] == "test"
        assert parsed["agent"] == "goose"


# ---------------------------------------------------------------------------
# CLI: init_project
# ---------------------------------------------------------------------------

class TestInitProject:

    def test_creates_directory_structure(self):
        pdir = init_project("myproj", "goose")

        assert pdir.exists()
        assert (pdir / "dsagt_config.yaml").exists()
        assert (pdir / "trace_archive").is_dir()
        assert (pdir / "mlflow").is_dir()
        assert (pdir / "skills").is_dir()
        assert (pdir / "kb_index").is_dir()
        # `tools/` is intentionally NOT created by init_project — ToolRegistry
        # creates it on first server startup so bundled tools get copied in.
        assert not (pdir / "tools").exists()

    def test_config_is_valid(self):
        init_project("myproj", "claude-code")
        config = load_config("myproj")
        assert config["project"] == "myproj"
        assert config["agent"] == "claude-code"

    def test_duplicate_raises(self):
        init_project("myproj", "goose")
        with pytest.raises(FileExistsError):
            init_project("myproj", "goose")

    def test_invalid_agent_raises(self):
        with pytest.raises(ValueError):
            init_project("myproj", "invalid-agent")

    def test_init_without_agent_omits_field_in_yaml(self, tmp_path):
        """Optional --agent: when omitted at init, the YAML has no 'agent:'
        field.  This is what enables `dsagt start --agent X` to set the
        default on first start.
        """
        pdir = init_project("myproj")
        config = yaml.safe_load((pdir / "dsagt_config.yaml").read_text())
        assert "agent" not in config

    def test_init_without_agent_skips_static_record(self, tmp_path):
        """Without --agent at init, no instructions file is written —
        we don't know which one to write.  Static record happens at
        first start instead.
        """
        pdir = init_project("myproj")
        # None of the per-agent marker files should exist
        assert not (pdir / "CLAUDE.md").exists()
        assert not (pdir / "AGENTS.md").exists()
        assert not (pdir / ".goosehints").exists()
        assert not (pdir / ".roomodes").exists()

    def test_init_with_agent_writes_static_record_eagerly(self, tmp_path):
        """With --agent at init, the static record is written immediately
        so the user can edit instructions before first start.
        """
        pdir = init_project("myproj", "claude-code")
        assert (pdir / "CLAUDE.md").exists()
        config = yaml.safe_load((pdir / "dsagt_config.yaml").read_text())
        assert config["agent"] == "claude-code"


# ---------------------------------------------------------------------------
# persist_agent_choice: first-start agent selection writes back to YAML
# ---------------------------------------------------------------------------

class TestPersistAgentChoice:

    def test_adds_field_when_absent(self, tmp_path):
        """First start without YAML default: --agent X persists into YAML."""
        init_project("myproj")
        persist_agent_choice("myproj", "codex")

        config = load_config("myproj")
        assert config["agent"] == "codex"

    def test_overwrites_existing_field(self, tmp_path):
        """If somehow called when YAML already has an agent (e.g., the
        user manually edited the YAML between starts), the new value
        wins.  Belt-and-suspenders: _cmd_start only calls this when the
        YAML had no agent, so overwrite is rare in practice.
        """
        init_project("myproj", "goose")
        persist_agent_choice("myproj", "claude-code")

        config = load_config("myproj")
        assert config["agent"] == "claude-code"

    def test_invalid_agent_raises(self, tmp_path):
        init_project("myproj")
        with pytest.raises(ValueError):
            persist_agent_choice("myproj", "not-an-agent")


# ---------------------------------------------------------------------------
# pick_free_port: tries preferred, falls back to next free in range
# ---------------------------------------------------------------------------

class TestPickFreePort:

    def test_returns_preferred_when_free(self, monkeypatch):
        from dsagt import session
        monkeypatch.setattr(session, "port_in_use", lambda port: False)
        port, warn = session.pick_free_port(4000)
        assert port == 4000
        assert warn is None

    def test_falls_back_when_preferred_taken(self, monkeypatch):
        from dsagt import session
        # 4000 is taken, 4001 is free
        monkeypatch.setattr(
            session, "port_in_use", lambda port: port == 4000,
        )
        monkeypatch.setattr(session, "port_held_by_foreign_process", lambda port: False)
        port, warn = session.pick_free_port(4000)
        assert port == 4001
        assert warn is not None
        assert "4000" in warn
        assert "4001" in warn

    def test_warning_mentions_dsagt_stop_for_orphan(self, monkeypatch):
        """When the preferred port looks like a stuck dsagt service, the
        warning suggests `dsagt stop` so users can reclaim it."""
        from dsagt import session
        monkeypatch.setattr(session, "port_in_use", lambda port: port == 4000)
        monkeypatch.setattr(session, "port_held_by_foreign_process", lambda port: False)
        _, warn = session.pick_free_port(4000)
        assert "dsagt stop" in warn

    def test_warning_mentions_lsof_for_foreign(self, monkeypatch):
        """When the preferred port is held by a non-dsagt process, the
        warning points at lsof so the user can identify the squatter."""
        from dsagt import session
        monkeypatch.setattr(session, "port_in_use", lambda port: port == 4000)
        monkeypatch.setattr(session, "port_held_by_foreign_process", lambda port: True)
        _, warn = session.pick_free_port(4000)
        assert "lsof" in warn

    def test_raises_when_all_in_range_taken(self, monkeypatch):
        from dsagt import session
        monkeypatch.setattr(session, "port_in_use", lambda port: True)
        monkeypatch.setattr(session, "port_held_by_foreign_process", lambda port: True)
        with pytest.raises(RuntimeError, match="All ports"):
            session.pick_free_port(4000, max_offset=3)


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
        """Run static then dynamic — what dsagt start does, minus services."""
        static_agent_record(config, config["agent"], working_dir)
        env = agent_env(config)
        dynamic_agent_record(config, env, working_dir)

    def test_claude_code_writes_mcp_json(self, tmp_path):
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        mcp_path = working_dir / ".mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "dsagt-registry" in mcp["mcpServers"]
        assert "dsagt-knowledge" in mcp["mcpServers"]
        assert (working_dir / ".dsagt_env").exists()
        assert (working_dir / "CLAUDE.md").exists()

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
        assert goose["GOOSE_PROVIDER"] == "openai"
        assert (working_dir / ".goosehints").exists()

    def test_roo_writes_static_and_env_dynamic(self, tmp_path):
        # Roo's static record creates .roo/ + .roomodes; dynamic writes
        # .roo/mcp.json (env-block-baked) and .dsagt_env.  Splitting was
        # the whole point of the refactor — pin both halves here.
        config = self._init_and_load("roo")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        assert (working_dir / ".roo").is_dir()
        assert (working_dir / ".roomodes").exists()
        assert (working_dir / ".roo" / "mcp.json").exists()

        env_content = (working_dir / ".dsagt_env").read_text()
        assert "DSAGT_PROJECT" in env_content

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
        # Codex's CODEX_HOME comes from agent_env, which derives it from
        # project_dir (not working_dir).  In production these are the
        # same path; mirror that in the test by using project_dir as
        # working_dir.
        config = self._init_and_load("codex")
        working_dir = Path(config["project_dir"])

        self._write_both(config, working_dir)

        assert (working_dir / "AGENTS.md").exists()
        assert (working_dir / ".codex-data").is_dir()
        assert (working_dir / ".codex-data" / "config.toml").exists()

        env_content = (working_dir / ".dsagt_env").read_text()
        assert "CODEX_HOME" in env_content
        assert str(working_dir / ".codex-data") in env_content

    def test_static_is_idempotent(self, tmp_path):
        # Running static twice doesn't duplicate or destroy content —
        # the marker check skips the second write.  This is what lets
        # users edit CLAUDE.md / AGENTS.md between init and start
        # without losing edits.
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        static_agent_record(config, "claude-code", working_dir)
        first = (working_dir / "CLAUDE.md").read_text()
        # Simulate a user edit
        (working_dir / "CLAUDE.md").write_text(first + "\n\n## My project notes\nfoo")
        edited = (working_dir / "CLAUDE.md").read_text()
        # Re-run static — should be no-op since marker is present
        static_agent_record(config, "claude-code", working_dir)
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
        """_render_codex_config produces a config.toml that routes Codex
        through the dsagt proxy with explicit MCP env injection.

        Pins the four invariants: model_provider name, base_url shape,
        wire_api=responses (Codex's only supported value), and
        disable_response_storage=true (required because the upstream
        behind our proxy translates /v1/responses -> /v1/chat/completions
        and can't honor previous_response_id state).
        """
        from dsagt.agents import _render_codex_config

        config = {
            "agent": "codex",
            "project": "p",
            "project_dir": "/proj",
            "proxy": {"port": 4242},
            "llm": {"model": "test-model"},
        }
        env = {}
        mcp_env = {"DSAGT_PROJECT_DIR": "/proj", "MLFLOW_TRACKING_URI": "http://localhost:5001"}

        toml = _render_codex_config(config, env, mcp_env)

        assert 'model = "test-model"' in toml
        assert 'model_provider = "dsagt-proxy"' in toml
        assert "[model_providers.dsagt-proxy]" in toml
        assert 'base_url = "http://localhost:4242/v1"' in toml
        assert 'wire_api = "responses"' in toml
        assert "disable_response_storage = true" in toml
        assert "requires_openai_auth = false" in toml
        assert 'env_key = "OPENAI_API_KEY"' in toml
        assert "[mcp_servers.dsagt-registry]" in toml
        assert "[mcp_servers.dsagt-knowledge]" in toml
        assert "[mcp_servers.dsagt-registry.env]" in toml
        assert 'MLFLOW_TRACKING_URI = "http://localhost:5001"' in toml

    def test_mcp_servers_dict_shape(self):
        from dsagt.agents import _build_mcp_servers_dict

        env_block = {"DSAGT_PROJECT_DIR": "/tmp/x", "MLFLOW_TRACKING_URI": "http://localhost:5001"}
        mcp = _build_mcp_servers_dict(env_block)

        assert set(mcp["mcpServers"]) == {"dsagt-registry", "dsagt-knowledge"}
        assert mcp["mcpServers"]["dsagt-knowledge"]["disabled"] is False
        # Env block plumbs through so the MCP server children have what they need.
        assert mcp["mcpServers"]["dsagt-registry"]["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5001"

    def test_mcp_config_has_project_dir_in_env(self, tmp_path):
        """MCP server entries should have DSAGT_PROJECT_DIR in their env
        block so the servers know where to find their config and data."""
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        mcp = json.loads((working_dir / ".mcp.json").read_text())
        reg_env = mcp["mcpServers"]["dsagt-registry"].get("env", {})
        assert "DSAGT_PROJECT_DIR" in reg_env

        kb_env = mcp["mcpServers"]["dsagt-knowledge"].get("env", {})
        assert "DSAGT_PROJECT_DIR" in kb_env

    def test_env_file_has_proxy_url(self, tmp_path):
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        env_content = (working_dir / ".dsagt_env").read_text()
        assert "ANTHROPIC_BASE_URL" in env_content
        assert "localhost:4000" in env_content
        assert "DSAGT_PROJECT" in env_content

    def test_goose_env_uses_openai_host(self, tmp_path):
        config = self._init_and_load("goose")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        env_content = (working_dir / ".dsagt_env").read_text()
        assert "OPENAI_HOST" in env_content


# ---------------------------------------------------------------------------
# run.py: _resolve_records_dir with DSAGT_PROJECT_DIR
# ---------------------------------------------------------------------------

class TestResolveRecordsDirProjectAware:

    def test_project_dir_env(self):
        from dsagt.provenance import _resolve_records_dir
        with patch.dict(os.environ, {"DSAGT_PROJECT_DIR": "/proj/dir"}, clear=False):
            os.environ.pop("DSAGT_RECORDS_DIR", None)
            result = _resolve_records_dir(None)
            assert result == Path("/proj/dir/trace_archive")

    def test_explicit_overrides_project_dir(self):
        from dsagt.provenance import _resolve_records_dir
        with patch.dict(os.environ, {"DSAGT_PROJECT_DIR": "/proj/dir"}):
            result = _resolve_records_dir("/custom")
            assert result == Path("/custom")


# ---------------------------------------------------------------------------
# CLI: agent_env
# ---------------------------------------------------------------------------

class TestAgentEnv:

    def _make_config(self, agent, project_dir="/proj"):
        return {
            "project": "test",
            "agent": agent,
            "project_dir": project_dir,
            "proxy": {"port": 4000},
            "llm": {"model": "test-model"},
            "embedding": {"api_key": "test-key"},
        }

    def test_claude_code_sets_anthropic_base_url(self):
        from dsagt.agents import agent_env
        env = agent_env(self._make_config("claude-code"))
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["DSAGT_PROJECT"] == "test"

    def test_goose_sets_openai_host(self):
        from dsagt.agents import agent_env
        env = agent_env(self._make_config("goose"))
        assert env["OPENAI_HOST"] == "http://localhost:4000"
        assert "ANTHROPIC_BASE_URL" not in env

    def test_embedding_routed_through_proxy(self):
        """agent_env points the embedding endpoint at our local proxy with
        a sentinel key.  The real EMBEDDING_API_KEY only lives in the
        dsagt-proxy subprocess (inherited from os.environ before the
        agent_env override) — agent and MCP children only see the sentinel.
        """
        from dsagt.agents import agent_env, _PROXY_FORWARDED_SENTINEL
        env = agent_env(self._make_config("claude-code"))
        assert env["EMBEDDING_BASE_URL"] == "http://localhost:4000"
        assert env["EMBEDDING_API_KEY"] == _PROXY_FORWARDED_SENTINEL
        assert env["OPENAI_BASE_URL"] == "http://localhost:4000"


# ---------------------------------------------------------------------------
# CLI: agent_command
# ---------------------------------------------------------------------------

class TestAgentCommand:

    def test_claude_code(self):
        from dsagt.agents import agent_command
        assert agent_command({"agent": "claude-code"}) == ["claude"]

    def test_goose(self):
        from dsagt.agents import agent_command
        assert agent_command({"agent": "goose"}) == [
            "goose", "session",
            "--with-extension", "uv run dsagt-registry-server",
            "--with-extension", "uv run dsagt-knowledge-server",
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

    def test_default_config_has_embedding_base_url(self):
        """default_config_content() includes embedding.base_url."""
        content = default_config_content("test", "claude-code")
        parsed = yaml.safe_load(content)
        assert "base_url" in parsed["embedding"]

    def test_mcp_env_block_routes_embedding_through_proxy(self):
        """_mcp_env_block points the MCP server's EMBEDDING_BASE_URL at the
        local proxy with a sentinel key — embeddings go through the same
        translation/observability pipeline as LLM calls."""
        from dsagt.agents import _mcp_env_block, _PROXY_FORWARDED_SENTINEL
        config = {"project_dir": "/p", "embedding": {"api_key": "k", "base_url": "https://api.test/v1", "model": "m"}}
        env = _mcp_env_block(config, proxy_port=4000)
        assert env["EMBEDDING_BASE_URL"] == "http://localhost:4000"
        assert env["EMBEDDING_API_KEY"] == _PROXY_FORWARDED_SENTINEL
        assert env["OPENAI_BASE_URL"] == "http://localhost:4000"

    def test_mcp_env_block_includes_model(self):
        """_mcp_env_block passes EMBEDDING_MODEL through so MCP-server-side
        litellm.embedding(model=...) matches the proxy's model_list entry."""
        from dsagt.agents import _mcp_env_block
        config = {"project_dir": "/p", "embedding": {"api_key": "k", "base_url": "u", "model": "my-model"}}
        env = _mcp_env_block(config, proxy_port=4000)
        assert env["EMBEDDING_MODEL"] == "my-model"

    def test_agent_env_routes_embedding_through_proxy(self):
        """agent_env() points the embedding endpoint at the local proxy."""
        from dsagt.agents import agent_env, _PROXY_FORWARDED_SENTINEL
        config = {
            "project": "test",
            "agent": "claude-code",
            "project_dir": "/proj",
            "proxy": {"port": 4000},
            "llm": {"model": "test-model"},
            "embedding": {"api_key": "k", "base_url": "https://api.test/v1", "model": "m"},
        }
        env = agent_env(config)
        assert env["EMBEDDING_BASE_URL"] == "http://localhost:4000"
        assert env["EMBEDDING_API_KEY"] == _PROXY_FORWARDED_SENTINEL
        assert env["OPENAI_BASE_URL"] == "http://localhost:4000"
        assert env["EMBEDDING_MODEL"] == "m"

    def test_mcp_server_args_are_just_command(self):
        """MCP server args are just ["run", "dsagt-<name>-server"].

        All configuration flows through env vars (DSAGT_PROJECT_DIR,
        EMBEDDING_BASE_URL/API_KEY pointing at proxy, EMBEDDING_MODEL) and
        dsagt_config.yaml.  No CLI flags needed.
        """
        from dsagt.agents import _mcp_server_args
        assert _mcp_server_args("knowledge") == ["run", "dsagt-knowledge-server"]
        assert _mcp_server_args("registry") == ["run", "dsagt-registry-server"]

    def test_mcp_env_block_includes_project_dir(self):
        """_mcp_env_block must include DSAGT_PROJECT_DIR so MCP servers
        know where to find their project directory and config."""
        from dsagt.agents import _mcp_env_block
        config = {
            "project_dir": "/home/user/dsagt-projects/test",
            "embedding": {"model": "m", "base_url": "u", "api_key": "k"},
        }
        env = _mcp_env_block(config, proxy_port=4000)
        assert env["DSAGT_PROJECT_DIR"] == "/home/user/dsagt-projects/test"
        assert env["EMBEDDING_MODEL"] == "m"

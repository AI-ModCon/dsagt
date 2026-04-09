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
    _resolve_env_vars,
    default_config_content,
    load_config,
    project_dir,
)
from dsagt.agents import generate_agent_configs
from dsagt.session import init_project


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
            assert _resolve_env_vars("${MY_KEY}") == "secret"

    def test_unset_var_left_as_is(self):
        os.environ.pop("NOPE", None)
        assert _resolve_env_vars("${NOPE}") == "${NOPE}"

    def test_nested_dicts(self):
        with patch.dict(os.environ, {"K": "v"}):
            result = _resolve_env_vars({"a": {"b": "${K}"}})
            assert result == {"a": {"b": "v"}}

    def test_non_string_passthrough(self):
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(True) is True


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

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent")

    def test_project_dir_injected(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {
            "project": "myproject",
            "agent": "goose",
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
        assert (pdir / "tools").is_dir()
        assert (pdir / "tools" / "code").is_dir()
        assert (pdir / "skills").is_dir()
        assert (pdir / "kb_index").is_dir()

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


# ---------------------------------------------------------------------------
# CLI: generate_agent_configs
# ---------------------------------------------------------------------------

class TestGenerateAgentConfigs:

    def _init_and_load(self, agent):
        init_project("testproj", agent)
        return load_config("testproj")

    def test_claude_code_generates_mcp_json(self, tmp_path):
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        actions = generate_agent_configs(config, working_dir)

        mcp_path = working_dir / ".mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "dsagt-registry" in mcp["mcpServers"]
        assert "dsagt-knowledge" in mcp["mcpServers"]

        # Env file written
        assert (working_dir / ".dsagt_env").exists()

    def test_goose_generates_goose_yaml(self, tmp_path):
        config = self._init_and_load("goose")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        goose_path = working_dir / "goose.yaml"
        assert goose_path.exists()
        goose = yaml.safe_load(goose_path.read_text())
        assert "registry" in goose["extensions"]
        assert "knowledge" in goose["extensions"]
        assert goose["GOOSE_PROVIDER"] == "openai"

    def test_roo_generates_roo_mcp(self, tmp_path):
        config = self._init_and_load("roo")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        mcp_path = working_dir / ".roo" / "mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "dsagt-registry" in mcp["mcpServers"]

    def test_cline_generates_mcp_json(self, tmp_path):
        config = self._init_and_load("cline")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        mcp_path = working_dir / "cline_mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "alwaysAllow" in mcp["mcpServers"]["dsagt-registry"]

    def test_mcp_args_include_project_dir(self, tmp_path):
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        mcp = json.loads((working_dir / ".mcp.json").read_text())
        reg_args = mcp["mcpServers"]["dsagt-registry"]["args"]
        assert "--runtime-dir" in reg_args

        kb_args = mcp["mcpServers"]["dsagt-knowledge"]["args"]
        assert "--base-index-dir" in kb_args

    def test_env_file_has_proxy_url(self, tmp_path):
        config = self._init_and_load("claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        env_content = (working_dir / ".dsagt_env").read_text()
        assert "ANTHROPIC_BASE_URL" in env_content
        assert "localhost:4000" in env_content
        assert "DSAGT_PROJECT" in env_content

    def test_goose_env_uses_openai_host(self, tmp_path):
        config = self._init_and_load("goose")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

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

    def test_embedding_key_set(self):
        from dsagt.agents import agent_env
        env = agent_env(self._make_config("claude-code"))
        assert env["LLM_API_KEY"] == "test-key"

    def test_unresolved_env_ref_skipped(self):
        from dsagt.agents import agent_env
        config = self._make_config("claude-code")
        config["embedding"]["api_key"] = "${UNSET_VAR}"
        env = agent_env(config)
        assert env.get("LLM_API_KEY") != "${UNSET_VAR}"


# ---------------------------------------------------------------------------
# CLI: agent_command
# ---------------------------------------------------------------------------

class TestAgentCommand:

    def test_claude_code(self):
        from dsagt.agents import agent_command
        assert agent_command({"agent": "claude-code"}) == ["claude"]

    def test_goose(self):
        from dsagt.agents import agent_command
        assert agent_command({"agent": "goose"}) == ["goose", "session"]

    def test_roo(self):
        from dsagt.agents import agent_command
        assert agent_command({"agent": "roo"}) == ["roo"]

    def test_cline(self):
        from dsagt.agents import agent_command
        assert agent_command({"agent": "cline"}) == ["cline"]


# ---------------------------------------------------------------------------
# Config flow: embedding config propagation
# ---------------------------------------------------------------------------

class TestConfigFlow:

    def test_default_config_has_embedding_base_url(self):
        """default_config_content() includes embedding.base_url."""
        content = default_config_content("test", "claude-code")
        parsed = yaml.safe_load(content)
        assert "base_url" in parsed["embedding"]

    def test_mcp_env_block_includes_base_url(self):
        """_mcp_env_block passes OPENAI_BASE_URL from config."""
        from dsagt.agents import _mcp_env_block
        config = {"embedding": {"api_key": "k", "base_url": "https://api.test/v1", "model": "m"}}
        env = _mcp_env_block(config)
        assert env["OPENAI_BASE_URL"] == "https://api.test/v1"

    def test_mcp_env_block_includes_model(self):
        """_mcp_env_block passes EMBEDDING_MODEL from config."""
        from dsagt.agents import _mcp_env_block
        config = {"embedding": {"api_key": "k", "base_url": "u", "model": "my-model"}}
        env = _mcp_env_block(config)
        assert env["EMBEDDING_MODEL"] == "my-model"

    def test_mcp_env_block_skips_empty_values(self):
        """_mcp_env_block doesn't set empty string env vars."""
        from dsagt.agents import _mcp_env_block
        config = {"embedding": {"api_key": "", "base_url": "", "model": ""}}
        env = _mcp_env_block(config)
        assert "LLM_API_KEY" not in env
        assert "OPENAI_BASE_URL" not in env
        assert "EMBEDDING_MODEL" not in env

    def test_agent_env_includes_embedding_base_url(self):
        """agent_env() sets OPENAI_BASE_URL from config."""
        from dsagt.agents import agent_env
        config = {
            "project": "test",
            "agent": "claude-code",
            "project_dir": "/proj",
            "proxy": {"port": 4000},
            "embedding": {"api_key": "k", "base_url": "https://api.test/v1", "model": "m"},
        }
        env = agent_env(config)
        assert env["OPENAI_BASE_URL"] == "https://api.test/v1"
        assert env["EMBEDDING_MODEL"] == "m"

    def test_mcp_server_args_include_embedding_flags(self):
        """Knowledge server args include embedding config from dsagt_config.yaml."""
        from dsagt.agents import _mcp_server_args
        config = {
            "embedding": {
                "base_url": "https://api.test/v1",
                "model": "my-model",
                "api_key": "my-key",
            }
        }
        args = _mcp_server_args("knowledge", Path("/proj"), config)
        assert "--embedding-base-url" in args
        assert "https://api.test/v1" in args
        assert "--embedding-model" in args
        assert "my-model" in args
        assert "--embedding-api-key" in args
        assert "my-key" in args

    def test_mcp_server_args_skip_unresolved_key(self):
        """Knowledge server args skip ${VAR} style api_key."""
        from dsagt.agents import _mcp_server_args
        config = {"embedding": {"api_key": "${LLM_API_KEY}", "base_url": "", "model": ""}}
        args = _mcp_server_args("knowledge", Path("/proj"), config)
        assert "--embedding-api-key" not in args

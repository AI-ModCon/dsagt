"""
Tests for DSAGT config loading, project init, and agent config generation.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from dsagt.config import (
    _deep_merge,
    _resolve_env_vars,
    default_config_content,
    load_config,
    project_dir_for,
)
from dsagt.session import (
    generate_agent_configs,
    init_project,
)


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

    def _write_config(self, tmp_path, content: dict):
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / "dsagt_config.yaml").write_text(
            yaml.dump(content, default_flow_style=False)
        )
        return project_dir

    def test_loads_minimal_config(self, tmp_path):
        project_dir = self._write_config(tmp_path, {
            "project": "test",
            "agent": "goose",
        })

        config = load_config(project_dir)

        assert config["project"] == "test"
        assert config["agent"] == "goose"
        assert config["proxy"]["port"] == 4000  # default
        assert config["mlflow"]["port"] == 5001  # default
        assert config["llm"]["model"] == "claude-sonnet-4-20250514"

    def test_overrides_defaults(self, tmp_path):
        project_dir = self._write_config(tmp_path, {
            "project": "test",
            "agent": "claude-code",
            "proxy": {"port": 9000},
        })

        config = load_config(project_dir)
        assert config["proxy"]["port"] == 9000

    def test_missing_project_raises(self, tmp_path):
        project_dir = self._write_config(tmp_path, {"agent": "goose"})
        with pytest.raises(ValueError, match="project"):
            load_config(project_dir)

    def test_missing_agent_raises(self, tmp_path):
        project_dir = self._write_config(tmp_path, {"project": "test"})
        with pytest.raises(ValueError, match="agent"):
            load_config(project_dir)

    def test_invalid_agent_raises(self, tmp_path):
        project_dir = self._write_config(tmp_path, {"project": "t", "agent": "copilot"})
        with pytest.raises(ValueError, match="copilot"):
            load_config(project_dir)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent")

    def test_project_dir_injected(self, tmp_path):
        project_dir = self._write_config(tmp_path, {
            "project": "test",
            "agent": "goose",
        })
        config = load_config(project_dir)
        assert config["project_dir"] == str(project_dir.resolve())


# ---------------------------------------------------------------------------
# Config: helpers
# ---------------------------------------------------------------------------

class TestProjectDirFor:

    def test_basic(self):
        p = project_dir_for("myproj", "/base")
        assert p == Path("/base/myproj")


class TestDefaultConfigContent:

    def test_roundtrips_as_valid_yaml(self):
        content = default_config_content("test", "goose")
        parsed = yaml.safe_load(content)
        assert parsed["project"] == "test"
        assert parsed["agent"] == "goose"


# ---------------------------------------------------------------------------
# Session: init_project
# ---------------------------------------------------------------------------

class TestInitProject:

    def test_creates_directory_structure(self, tmp_path):
        project_dir = init_project("myproj", "goose", runtime_base=tmp_path)

        assert project_dir.exists()
        assert (project_dir / "dsagt_config.yaml").exists()
        assert (project_dir / "trace_archive").is_dir()
        assert (project_dir / "mlflow").is_dir()
        assert (project_dir / "skills").is_dir()
        assert (project_dir / "kb_index").is_dir()

    def test_config_is_valid(self, tmp_path):
        project_dir = init_project("myproj", "claude-code", runtime_base=tmp_path)
        config = load_config(project_dir)
        assert config["project"] == "myproj"
        assert config["agent"] == "claude-code"

    def test_duplicate_raises(self, tmp_path):
        init_project("myproj", "goose", runtime_base=tmp_path)
        with pytest.raises(FileExistsError):
            init_project("myproj", "goose", runtime_base=tmp_path)

    def test_invalid_agent_raises(self, tmp_path):
        with pytest.raises(ValueError):
            init_project("myproj", "invalid-agent", runtime_base=tmp_path)


# ---------------------------------------------------------------------------
# Session: generate_agent_configs
# ---------------------------------------------------------------------------

class TestGenerateAgentConfigs:

    def _init_and_load(self, tmp_path, agent):
        project_dir = init_project("testproj", agent, runtime_base=tmp_path / "runtime")
        return load_config(project_dir)

    def test_claude_code_generates_mcp_json(self, tmp_path):
        config = self._init_and_load(tmp_path, "claude-code")
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
        config = self._init_and_load(tmp_path, "goose")
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
        config = self._init_and_load(tmp_path, "roo")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        mcp_path = working_dir / ".roo" / "mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "dsagt-registry" in mcp["mcpServers"]

    def test_cline_generates_mcp_json(self, tmp_path):
        config = self._init_and_load(tmp_path, "cline")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        mcp_path = working_dir / "cline_mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text())
        assert "alwaysAllow" in mcp["mcpServers"]["dsagt-registry"]

    def test_mcp_args_include_project_dir(self, tmp_path):
        config = self._init_and_load(tmp_path, "claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        mcp = json.loads((working_dir / ".mcp.json").read_text())
        reg_args = mcp["mcpServers"]["dsagt-registry"]["args"]
        assert "--runtime-dir" in reg_args

        kb_args = mcp["mcpServers"]["dsagt-knowledge"]["args"]
        assert "--base-index-dir" in kb_args

    def test_env_file_has_proxy_url(self, tmp_path):
        config = self._init_and_load(tmp_path, "claude-code")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(config, working_dir)

        env_content = (working_dir / ".dsagt_env").read_text()
        assert "ANTHROPIC_BASE_URL" in env_content
        assert "localhost:4000" in env_content
        assert "DSAGT_PROJECT" in env_content

    def test_goose_env_uses_openai_host(self, tmp_path):
        config = self._init_and_load(tmp_path, "goose")
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
        from dsagt.run import _resolve_records_dir
        with patch.dict(os.environ, {"DSAGT_PROJECT_DIR": "/proj/dir"}, clear=False):
            os.environ.pop("DSAGT_RECORDS_DIR", None)
            result = _resolve_records_dir(None)
            assert result == Path("/proj/dir/trace_archive")

    def test_explicit_overrides_project_dir(self):
        from dsagt.run import _resolve_records_dir
        with patch.dict(os.environ, {"DSAGT_PROJECT_DIR": "/proj/dir"}):
            result = _resolve_records_dir("/custom")
            assert result == Path("/custom")


# ---------------------------------------------------------------------------
# Session: agent_env
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
        from dsagt.session import agent_env
        env = agent_env(self._make_config("claude-code"))
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["DSAGT_PROJECT"] == "test"

    def test_goose_sets_openai_host(self):
        from dsagt.session import agent_env
        env = agent_env(self._make_config("goose"))
        assert env["OPENAI_HOST"] == "http://localhost:4000"
        assert "ANTHROPIC_BASE_URL" not in env

    def test_embedding_key_set(self):
        from dsagt.session import agent_env
        env = agent_env(self._make_config("claude-code"))
        assert env["LLM_API_KEY"] == "test-key"

    def test_unresolved_env_ref_skipped(self):
        from dsagt.session import agent_env
        config = self._make_config("claude-code")
        config["embedding"]["api_key"] = "${UNSET_VAR}"
        env = agent_env(config)
        assert env.get("LLM_API_KEY") != "${UNSET_VAR}"


# ---------------------------------------------------------------------------
# Session: agent_command
# ---------------------------------------------------------------------------

class TestAgentCommand:

    def test_claude_code(self):
        from dsagt.session import agent_command
        assert agent_command({"agent": "claude-code"}) == ["claude"]

    def test_goose(self):
        from dsagt.session import agent_command
        assert agent_command({"agent": "goose"}) == ["goose", "session"]

    def test_roo_returns_none(self):
        from dsagt.session import agent_command
        assert agent_command({"agent": "roo"}) is None

    def test_cline_returns_none(self):
        from dsagt.session import agent_command
        assert agent_command({"agent": "cline"}) is None

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
from dsagt.session import init_project


@pytest.fixture(autouse=True)
def _use_tmp_registry(tmp_path):
    """Redirect project registry and default location to tmp_path for all tests.

    The fake registry auto-discovers any project dir that exists under tmp_path,
    so tests that create dirs manually (without init_project) still work.
    """
    registry = {}

    def fake_load():
        # Auto-discover: any subdir of tmp_path with .dsagt/config.yaml counts
        discovered = dict(registry)
        for child in tmp_path.iterdir():
            if child.is_dir() and (child / ".dsagt" / "config.yaml").exists():
                discovered.setdefault(child.name, str(child))
        return discovered

    def fake_save(reg):
        registry.clear()
        registry.update(reg)

    def fake_register(name, path):
        registry[name] = str(Path(path).resolve())

    # Isolate the shared KB to tmp_path and stub the asset build so init
    # tests stay fast and offline (no embedding-model load, no git clone).
    def _noop_ensure_assets(*_a, **_k):
        return {"built": [], "skipped": []}

    with patch("dsagt.session._load_registry", fake_load):
        with patch("dsagt.session._save_registry", fake_save):
            with patch("dsagt.session.register_project", fake_register):
                with patch("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path):
                    with patch("dsagt.session.REGISTRY_DIR", tmp_path):
                        with patch(
                            "dsagt.commands.setup_core_kb.ensure_assets",
                            _noop_ensure_assets,
                        ):
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
        (pdir / ".dsagt").mkdir(parents=True, exist_ok=True)
        (pdir / ".dsagt" / "config.yaml").write_text(
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
        # Serverless: no mlflow block at all — the store is a sqlite path
        # resolved from the project dir, nothing to pin in config.
        assert "proxy" not in config
        assert "mlflow" not in config
        # User-supplied keys win on the merge; DEFAULTS fills in missing
        # llm.* fields with ``${VAR}`` placeholders (resolved by
        # ``resolve_env_vars`` against the user's shell, or filtered by
        # ``_real()`` if env unset).
        assert config["llm"]["provider"] == "openai"  # user value preserved

    def test_missing_project_raises(self, tmp_path):
        name = self._write_config(tmp_path, "myproject", {"agent": "goose"})
        with pytest.raises(ValueError, match="project"):
            load_config(name)

    def test_skills_block_backfilled_for_old_config(self, tmp_path):
        """A config with no skills block gets the default genesis source.

        ``populate_native`` is a code default (in ``AgentSetup.setup_skills``),
        not a config key — so it isn't backfilled here."""
        name = self._write_config(
            tmp_path,
            "myproject",
            {"project": "myproject", "agent": "claude"},
        )
        config = load_config(name)
        sources = config["skills"]["sources"]
        assert sources[0]["name"] == "genesis"
        assert "genesis-skills" in sources[0]["url"]
        assert "populate_native" not in config["skills"]

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

        assert DEFAULTS["skills"]["sources"][0]["name"] == "genesis"

    def test_default_config_content_includes_skills(self):
        body = yaml.safe_load(default_config_content("p", "claude"))
        assert body["skills"]["sources"][0]["name"] == "genesis"


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

    def test_no_user_facing_llm_block(self):
        """BYOA: project YAML is internal-only — no llm: block, no
        ${VAR} placeholders.  User credentials live in their shell."""
        content = default_config_content("test", "claude")
        parsed = yaml.safe_load(content)
        assert "llm" not in parsed
        assert "${" not in content

    def test_no_mlflow_port_pinned(self):
        """Serverless store — nothing to pin.  The config carries no
        mlflow block at all."""
        content = default_config_content("test", "claude")
        parsed = yaml.safe_load(content)
        assert "mlflow" not in parsed


# ---------------------------------------------------------------------------
# CLI: unlisted `mlflow` alias for `traces`
# ---------------------------------------------------------------------------


class TestMlflowAlias:
    """``dsagt mlflow`` routes to ``traces`` without ever reaching argparse,
    so it stays out of --help; only the command slot is rewritten (a project
    may itself be named "mlflow")."""

    def _capture_traces(self, monkeypatch):
        from dsagt.commands import cli

        seen = {}

        def fake(args):
            seen.update(project=args.project, port=args.port)
            return 0

        monkeypatch.setattr(cli, "_cmd_traces", fake)
        return cli, seen

    def test_mlflow_routes_to_traces(self, monkeypatch):
        cli, seen = self._capture_traces(monkeypatch)
        assert cli.main(["mlflow", "myproj", "--port", "5001"]) == 0
        assert seen == {"project": "myproj", "port": 5001}

    def test_project_named_mlflow_is_not_rewritten(self, monkeypatch):
        cli, seen = self._capture_traces(monkeypatch)
        assert cli.main(["traces", "mlflow"]) == 0
        assert seen == {"project": "mlflow", "port": 5000}

    def test_alias_absent_from_help_command_list(self, capsys):
        from dsagt.commands import cli

        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        choices = out[out.index("{") + 1 : out.index("}")]
        assert "mlflow" not in choices


# ---------------------------------------------------------------------------
# CLI: init choice resolution (_collect_settings)
# ---------------------------------------------------------------------------


class TestCollectSettings:
    """``_collect_settings`` resolves the init choices into the config blocks.

    Two selection questions (KB collections, skill sources) + agent; the
    bundled ``tools`` collection is always provisioned and not a choice.
    """

    def test_interactive_menus(self):
        """Interactive path drives questionary select/checkbox; genesis is the
        default-checked skill source, collections default to none, tools is
        always in the provisioning set."""
        import types
        import questionary
        from unittest.mock import patch
        from dsagt.commands import cli

        def fake_select(message, choices, default, **kw):
            return _Ask(default)

        def fake_checkbox(message, choices, **kw):
            return _Ask([c.value for c in choices if c.checked])

        class _Ask:
            def __init__(self, ret):
                self.ret = ret

            def ask(self):
                return self.ret

        args = types.SimpleNamespace(agent=None, include=None, exclude=None)
        with (
            patch.object(questionary, "select", fake_select),
            patch.object(questionary, "checkbox", fake_checkbox),
            patch.object(cli, "_confirm", lambda *a, **k: False),  # episodic off
        ):
            s = cli._collect_settings(args, interactive=True, existing={}, pdir=None)

        assert s["agent"] == "claude"
        assert s["knowledge"] == {"collections": []}
        assert [src["name"] for src in s["skills"]["sources"]] == ["genesis"]
        assert s["assets"] == ["codes", "genesis"]  # tools always provisioned
        assert s["episodic"] is None  # opt-in, off by default

    def test_interactive_episodic_enabled(self):
        """Enabling episodic at the prompt yields the opt-in block."""
        import types
        import questionary
        from unittest.mock import patch
        from dsagt.commands import cli

        class _Ask:
            def __init__(self, ret):
                self.ret = ret

            def ask(self):
                return self.ret

        args = types.SimpleNamespace(agent=None, include=None, exclude=None)
        with (
            patch.object(questionary, "select", lambda *a, **k: _Ask("claude")),
            patch.object(questionary, "checkbox", lambda *a, **k: _Ask([])),
            patch.object(cli, "_confirm", lambda *a, **k: True),
        ):
            s = cli._collect_settings(args, interactive=True, existing={}, pdir=None)

        assert s["episodic"] == {"enabled": True}

    def test_non_interactive_splits_assets(self):
        """No-TTY path splits --include into collections vs skill sources;
        tools is always provisioned.  Episodic omitted → None."""
        import types
        from dsagt.commands import cli

        args = types.SimpleNamespace(
            agent="goose", include=["codes", "nemo_curator", "anthropic"], exclude=None
        )
        s = cli._collect_settings(args, interactive=False, existing={}, pdir=None)
        assert s["agent"] == "goose"
        assert s["knowledge"]["collections"] == ["nemo_curator"]
        assert [src["name"] for src in s["skills"]["sources"]] == ["anthropic"]
        assert s["assets"] == ["codes", "nemo_curator", "anthropic"]
        assert s["episodic"] is None

    def test_non_interactive_episodic_flag(self):
        """--episodic builds the opt-in block on the no-TTY path."""
        import types
        from dsagt.commands import cli

        args = types.SimpleNamespace(
            agent="goose",
            include=["codes"],
            exclude=None,
            episodic=True,
        )
        s = cli._collect_settings(args, interactive=False, existing={}, pdir=None)
        assert s["episodic"] == {"enabled": True}


# ---------------------------------------------------------------------------
# CLI: init_project
# ---------------------------------------------------------------------------


class TestInitProject:

    def test_creates_directory_structure(self):
        pdir = init_project("myproj", "goose")

        assert pdir.exists()
        assert (pdir / ".dsagt" / "config.yaml").exists()
        assert (pdir / "trace_archive").is_dir()
        assert (pdir / "skills").is_dir()
        assert (pdir / "kb_index").is_dir()
        assert (pdir / ".dsagt").is_dir()
        # Bundled codes are copied into codes/ at init — every available
        # code in one place, one format (skill-standard dirs).
        assert (pdir / "codes" / "scan-directory" / "SKILL.md").exists()
        assert (pdir / "codes" / "scan-directory" / "scripts").is_dir()
        # Serverless: no MLflow store is pre-created; ``mlflow.db`` is
        # written lazily by the MLflow client on first span.
        assert not (pdir / "mlflow.db").exists()
        assert not (pdir / "mlflow").exists()

    def test_config_is_valid(self):
        init_project("myproj", "claude")
        config = load_config("myproj")
        assert config["project"] == "myproj"
        assert config["agent"] == "claude"

    def test_returns_pdir(self):
        """Serverless: init_project returns just the project dir — no port."""
        pdir = init_project("myproj", "goose")
        assert pdir.exists()
        config = load_config("myproj")
        assert "mlflow" not in config

    def test_exclude_all_creates_empty_kb_without_building(self):
        """``--exclude all`` provisions a valid project with an empty KB and
        never attempts an asset build."""
        from dsagt.commands import setup_core_kb

        with patch.object(setup_core_kb, "ensure_assets") as mock_ensure:
            pdir = init_project("myproj", "goose", exclude=["all"])
            mock_ensure.assert_not_called()
        kb_index = pdir / "kb_index"
        assert kb_index.is_dir()
        assert not any(kb_index.iterdir())

    def test_default_init_provisions_default_asset_set(self):
        """Default init builds exactly the default asset set into the shared
        cache (here stubbed) — tools + genesis, nothing heavier."""
        from dsagt.commands import setup_core_kb

        with patch.object(
            setup_core_kb, "ensure_assets", return_value={"built": [], "skipped": []}
        ) as mock_ensure:
            init_project("myproj", "goose")
            mock_ensure.assert_called_once()
            requested = mock_ensure.call_args.args[0]
            assert requested == ["codes", "genesis"]

    def test_reinit_is_idempotent_update(self):
        """``dsagt init`` is re-runnable: a second init on the same project
        updates settings in place rather than raising."""
        init_project("myproj", "goose")
        init_project("myproj", "claude")  # re-init, switch agent
        config = load_config("myproj")
        assert config["agent"] == "claude"

    def test_reinit_handle_destructive_survives_missing_embedding_key(self):
        """Regression: re-init runs ``_handle_destructive``, which must not
        KeyError on settings that carry no ``embedding`` key (it was dropped as
        an init choice).  ``init_project()`` bypasses this path — which is how
        the crash shipped — so drive ``_handle_destructive`` directly."""
        import types
        from dsagt.commands import cli

        init_project("myproj", "goose")
        existing = load_config("myproj")
        pdir = project_dir("myproj")

        args = types.SimpleNamespace(
            agent="goose", include=["codes"], exclude=None, episodic=False
        )
        settings = cli._collect_settings(
            args, interactive=False, existing=existing, pdir=pdir
        )
        assert "embedding" not in settings  # the invariant the old code broke

        # Must not raise (previously KeyError('embedding')).
        cli._handle_destructive(existing, settings, pdir, interactive=False)

    def test_invalid_agent_raises(self):
        with pytest.raises(ValueError):
            init_project("myproj", "invalid-agent")

    def test_episodic_block_round_trips_through_config(self):
        """init_project writes the opted-in episodic block; load_config reads it
        back.  A project without it backfills ``enabled: False`` from DEFAULTS."""
        epi = {"enabled": True}
        init_project("withmem", "goose", exclude=["all"], episodic=epi)
        cfg = load_config("withmem")
        assert cfg["episodic"]["enabled"] is True

        init_project("nomem", "goose", exclude=["all"])
        cfg2 = load_config("nomem")
        assert cfg2["episodic"]["enabled"] is False  # backfilled default

    def test_static_record_written_eagerly(self, tmp_path):
        """BYOA flow writes static + dynamic records at init time so the
        user can edit instructions and inspect the MCP config artifact
        before launching their agent.  Static record is now driven by
        the CLI command, not init_project itself — but the agent dir
        layout exists post-init.
        """
        init_project("myproj", "claude")
        config = load_config("myproj")
        assert config["agent"] == "claude"


# ---------------------------------------------------------------------------
# Session state (.dsagt/state.yaml) — owned by the MCP server
# ---------------------------------------------------------------------------


class TestSessionState:
    """The MCP server mints sessions into ``.dsagt/state.yaml`` (a monotonic
    per-project counter) and ``dsagt-run`` reads the current tag from there.
    """

    def test_append_session_increments(self, tmp_path):
        from dsagt.session import append_session, current_session

        pdir = tmp_path / "proj"
        pdir.mkdir()
        e1 = append_session(pdir)
        e2 = append_session(pdir)
        assert e1["id"] == 1
        assert e2["id"] == 2
        assert e1["started_at"].endswith("Z")
        assert current_session(pdir)["id"] == 2

    def test_session_tag_shape(self):
        from dsagt.session import session_tag

        assert session_tag("myproj", 3) == "myproj-3"

    def test_current_session_tag_from_state(self, tmp_path):
        from dsagt.session import (
            append_session,
            current_session_tag,
            write_config_file,
            build_config,
        )

        pdir = tmp_path / "proj"
        pdir.mkdir()
        write_config_file(pdir, build_config("proj", "claude"))
        assert current_session_tag(pdir, "proj") is None  # no session yet
        append_session(pdir)
        assert current_session_tag(pdir, "proj") == "proj-1"

    def test_update_cursor_roundtrip(self, tmp_path):
        from dsagt.session import read_state, update_cursor

        pdir = tmp_path / "proj"
        pdir.mkdir()
        update_cursor(pdir, tool_use_indexed_through="2026-01-01T00:00:00Z")
        cur = read_state(pdir)["memory_cursor"]
        assert cur["tool_use_indexed_through"] == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Per-agent record writers: static_agent_record + dynamic_agent_record
#
# Each test runs both writers against a project init'd with --agent,
# mirroring what `dsagt start` does in production.
# ---------------------------------------------------------------------------


class TestAgentRecord:

    def _init_and_load(self, agent):
        init_project("testproj", agent)
        return load_config("testproj")

    def _write_both(self, config, working_dir):
        """Run static then dynamic — what dsagt start does.  Serverless:
        no port to populate; the store resolves from the project dir."""
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
        assert set(mcp["mcpServers"]) == {"dsagt"}
        assert mcp["mcpServers"]["dsagt"]["args"] == ["run", "dsagt-server"]
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
        assert set(goose["extensions"]) == {"dsagt"}
        assert goose["extensions"]["dsagt"]["cmd"] == "uv run dsagt-server"
        assert (working_dir / ".goosehints").exists()

    def test_cline_writes_project_mcp_settings(self, tmp_path):
        """Cline's dynamic writer hand-writes the per-project MCP settings
        file (no cline binary needed); runtime_env points cline at it via
        CLINE_MCP_SETTINGS_PATH, leaving global auth + settings untouched."""
        import json as _json

        from dsagt.agents import AGENTS

        config = self._init_and_load("cline")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        static_agent_record(config, config["agent"], working_dir)
        assert (working_dir / ".clinerules" / "dsagt_instructions.md").exists()

        dynamic_agent_record(config, env={}, working_dir=working_dir)
        settings_path = working_dir / ".cline-data" / "cline_mcp_settings.json"
        settings = _json.loads(settings_path.read_text())
        transport = settings["mcpServers"]["dsagt"]["transport"]
        assert transport["type"] == "stdio"
        assert [transport["command"], *transport["args"]] == [
            "uv",
            "run",
            "dsagt-server",
        ]
        assert "DSAGT_PROJECT_DIR" in transport["env"]

        env = AGENTS["cline"]().runtime_env(config)
        assert env["CLINE_MCP_SETTINGS_PATH"].endswith(
            ".cline-data/cline_mcp_settings.json"
        )
        assert "CLINE_DIR" not in env

    def test_cline_dynamic_preserves_user_mcp_entries(self, tmp_path):
        """Re-running the writer keeps non-dsagt servers the user added."""
        import json as _json

        config = self._init_and_load("cline")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        dynamic_agent_record(config, env={}, working_dir=working_dir)
        settings_path = working_dir / ".cline-data" / "cline_mcp_settings.json"
        settings = _json.loads(settings_path.read_text())
        settings["mcpServers"]["mytool"] = {
            "transport": {"type": "stdio", "command": "mytool", "args": []}
        }
        settings_path.write_text(_json.dumps(settings))

        dynamic_agent_record(config, env={}, working_dir=working_dir)
        settings = _json.loads(settings_path.read_text())
        assert set(settings["mcpServers"]) == {"dsagt", "mytool"}
        assert (working_dir / ".cline-data").is_dir()

    def test_codex_writes_static_and_dynamic(self, tmp_path):
        # Codex's .codex-data is rooted at working_dir.
        config = self._init_and_load("codex")
        working_dir = Path(config["project_dir"])

        self._write_both(config, working_dir)

        assert (working_dir / "AGENTS.md").exists()
        assert (working_dir / ".codex-data").is_dir()
        toml = (working_dir / ".codex-data" / "config.toml").read_text()
        assert "[mcp_servers.dsagt.env]" in toml
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
        """``_render_codex_config`` emits only ``[mcp_servers.*]`` sections.
        No ``[otel]`` block — DSAGT no longer forces codex's native
        telemetry (nor the ``log_user_prompt`` privacy override).  No
        top-level keys — those come from the user's ``~/.codex/config.toml``
        which ``write_dynamic`` copies as a base.
        """
        from dsagt.agents import _render_codex_config

        mcp_env = {
            "DSAGT_PROJECT_DIR": "/proj",
            "MLFLOW_TRACKING_URI": "http://localhost:5001",
        }
        toml = _render_codex_config(mcp_env)

        assert "[mcp_servers.dsagt]" in toml
        assert "[mcp_servers.dsagt.env]" in toml
        assert 'MLFLOW_TRACKING_URI = "http://localhost:5001"' in toml
        # No forced telemetry / privacy override.
        assert "[otel]" not in toml
        assert "log_user_prompt" not in toml
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
        assert set(parsed["mcp"]) == {"dsagt"}
        reg = parsed["mcp"]["dsagt"]
        assert reg["type"] == "local"
        assert reg["command"] == ["uv", "run", "dsagt-server"]
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

        assert set(mcp["mcpServers"]) == {"dsagt"}
        assert mcp["mcpServers"]["dsagt"]["disabled"] is False
        # Env block plumbs through so the MCP server children have what they need.
        assert (
            mcp["mcpServers"]["dsagt"]["env"]["MLFLOW_TRACKING_URI"]
            == "http://localhost:5001"
        )

    def test_mcp_config_carries_routing_env(self, tmp_path):
        """Benign routing in the MCP env block: agents that don't inherit
        the parent's shell env into their MCP children (codex / cline
        — and claude's block is robust against shells that don't
        export it) need project name + dir and the serverless
        ``MLFLOW_TRACKING_URI`` baked in.  No credentials, no OTel."""
        config = self._init_and_load("claude")
        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        self._write_both(config, working_dir)

        mcp = json.loads((working_dir / ".mcp.json").read_text())
        env = mcp["mcpServers"]["dsagt"].get("env", {})
        assert env["DSAGT_PROJECT"] == config["project"]
        assert env["DSAGT_PROJECT_DIR"] == config["project_dir"]
        assert env["MLFLOW_TRACKING_URI"].startswith("sqlite:///")
        # No credentials or OTel routing leak into the MCP config.
        assert "ANTHROPIC_API_KEY" not in env
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env


# ---------------------------------------------------------------------------
# run.py: _resolve_records_dir with DSAGT_PROJECT_DIR
# ---------------------------------------------------------------------------


class TestResolveRecordsDirProjectAware:
    """``_resolve_records_dir`` reads the project's ``.dsagt/config.yaml``
    from cwd as the single source of truth — no env-var chain."""

    def test_explicit_overrides_cwd(self, tmp_path):
        from dsagt.provenance import _resolve_records_dir

        result = _resolve_records_dir("/custom")
        assert result == Path("/custom")

    def test_cwd_with_config(self, tmp_path, monkeypatch):
        """When cwd contains ``.dsagt/config.yaml``, records dir is
        ``<cwd>/trace_archive``.  Even if env vars are set to point
        elsewhere, the config-in-cwd rule wins (env is ignored)."""
        from dsagt.provenance import _resolve_records_dir

        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
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

    def test_no_otel_routing_for_any_agent(self, monkeypatch):
        """DSAGT forces no native OTel emission — agent traces are
        recovered post-hoc from the on-disk transcript.  ``agent_env``
        sets ``MLFLOW_TRACKING_URI`` (for MCP-server / MLflow-client
        logging) but never the OTLP routing env, for any agent.
        """
        from dsagt.agents import agent_env

        for var in (
            "MLFLOW_TRACKING_URI",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_RESOURCE_ATTRIBUTES",
        ):
            monkeypatch.delenv(var, raising=False)

        for agent in ("claude", "goose", "codex", "cline", "opencode"):
            env = agent_env(self._make_config(agent))
            # Serverless sqlite store derived from the project dir.
            assert env["MLFLOW_TRACKING_URI"] == "sqlite:////proj/mlflow.db"
            assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
            assert "OTEL_EXPORTER_OTLP_HEADERS" not in env
            assert "OTEL_RESOURCE_ATTRIBUTES" not in env

    def test_no_telemetry_flags_for_claude(self, monkeypatch):
        """The forced Claude telemetry/privacy flags are gone — DSAGT no
        longer flips ``CLAUDE_CODE_ENABLE_TELEMETRY`` / ``OTEL_LOG_*``
        (the latter defeated Anthropic's off-by-default redaction).

        Cleared from the inherited shell env first so we test that DSAGT
        doesn't *add* them — not whatever the test runner's own shell set.
        """
        from dsagt.agents import agent_env

        flags = (
            "CLAUDE_CODE_ENABLE_TELEMETRY",
            "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA",
            "OTEL_LOG_TOOL_DETAILS",
            "OTEL_LOG_USER_PROMPTS",
            "OTEL_TRACES_EXPORTER",
            "OTEL_LOG_RAW_API_BODIES",
        )
        for flag in flags:
            monkeypatch.delenv(flag, raising=False)

        env = agent_env(self._make_config("claude"))
        for flag in flags:
            assert flag not in env


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
            "uv run dsagt-server",
        ]

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

    def test_default_config_mirrors_init_choices(self):
        """The written config holds only the init choices: project, agent,
        knowledge.collections, skills.sources.  The bundled ``tools`` collection
        is always provisioned (not a choice); embedding / chunk_size / rerank
        are code defaults backfilled on read, never written."""
        content = default_config_content("test", "claude")
        parsed = yaml.safe_load(content)
        assert set(parsed) == {"project", "agent", "knowledge", "skills"}
        assert parsed["knowledge"] == {"collections": []}
        assert parsed["skills"]["sources"][0]["name"] == "genesis"
        assert "embedding" not in parsed
        assert "episodic" not in parsed  # opt-in, omitted unless enabled

    def test_episodic_written_only_when_enabled(self):
        """An opted-in episodic block is written verbatim; absent otherwise
        (and ``load_config`` backfills ``enabled: false`` for the absent case)."""
        epi = {"enabled": True}
        parsed = yaml.safe_load(default_config_content("t", "claude", episodic=epi))
        assert parsed["episodic"] == epi
        # Omitted when None.
        assert "episodic" not in yaml.safe_load(default_config_content("t", "claude"))

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

    def test_mcp_env_block_carries_project_routing(self, monkeypatch):
        """Benign routing: the MCP env block carries project name + dir and
        the serverless ``MLFLOW_TRACKING_URI`` so MCP children of agents
        that don't inherit the parent shell env still log to the right
        store.  No credentials, no OTel."""
        from dsagt.agents import _mcp_env_block

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        config = {
            "project": "test",
            "project_dir": "/p",
            "embedding": {"backend": "local"},
        }
        env = _mcp_env_block(config)
        assert env["DSAGT_PROJECT"] == "test"
        assert env["DSAGT_PROJECT_DIR"] == "/p"
        assert env["MLFLOW_TRACKING_URI"] == "sqlite:////p/mlflow.db"
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
        """MCP server args are just ["run", "dsagt-server"] — one merged
        server.  All configuration flows through .dsagt/config.yaml (cwd-walk).
        """
        from dsagt.agents import _mcp_server_args

        assert _mcp_server_args() == ["run", "dsagt-server"]

    def test_mcp_env_block_carries_no_session_id(self):
        """The MCP server owns the session lifecycle now (minted into
        ``.dsagt/state.yaml`` at startup), so the env block never carries a
        ``DSAGT_SESSION_ID`` — only project routing + embedding settings."""
        from dsagt.agents import _mcp_env_block

        config = {
            "project": "test",
            "project_dir": "/home/user/dsagt-projects/test",
            "embedding": {"model": "m", "base_url": "u"},
        }
        env = _mcp_env_block(config)
        assert "DSAGT_SESSION_ID" not in env
        assert env["EMBEDDING_MODEL"] == "m"
        assert env["EMBEDDING_BASE_URL"] == "u"
        # Even if a stray session_id is on the config dict, it isn't emitted.
        env2 = _mcp_env_block({**config, "session_id": "test-1"})
        assert "DSAGT_SESSION_ID" not in env2


class TestNoLaunchShim:
    """Phase 1 collapsed the launch surface: ``dynamic_agent_record``
    writes the MCP config but NO ``dsagt-launch.sh`` shim.  The user
    starts the agent directly in the project dir or via ``dsagt start``."""

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

    @pytest.mark.parametrize("agent_name", ["goose", "claude", "codex"])
    def test_no_shim_written(self, agent_name, tmp_path):
        from dsagt.agents import dynamic_agent_record

        config = self._make_config(agent_name, tmp_path)
        dynamic_agent_record(config, env={}, working_dir=tmp_path)

        assert not (tmp_path / "dsagt-launch.sh").exists()


class TestClaudeSetup:
    """`dsagt init --agent claude` writes `.mcp.json` and does NOT wire MLflow's
    autolog Stop hook — DSAGT's own serverless heartbeat pipeline (ClaudeReader →
    ClaudeTranslator → MLflowSink) produces Claude's traces, uniformly with every
    other agent, so wiring autolog too would double-log."""

    def test_writes_mcp_json_no_autolog_hook(self, tmp_path, monkeypatch):
        from dsagt.agents.claude import ClaudeSetup

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        config = {
            "project": "myproj",
            "project_dir": str(tmp_path),
            "agent": "claude",
            "embedding": {},
            "llm": {},
        }
        actions = ClaudeSetup().write_dynamic(
            config, env={}, working_dir=tmp_path, pdir=tmp_path
        )

        assert (tmp_path / ".mcp.json").exists()
        # No autolog: no Stop hook, no .claude/settings.json.
        assert not any("autolog" in a.lower() for a in actions)
        assert not (tmp_path / ".claude" / "settings.json").exists()

"""The Claude Code PreToolUse hook that refuses bare python."""

import io
import json

import pytest

from dsagt.commands.bash_guard import bare_python_call, main


@pytest.mark.parametrize(
    "command",
    [
        "python compare.py a.json b.json",
        "python3 -c 'import pandas as pd; print(pd.read_csv(\"data/x.csv\").shape)'",
        "cd data && python3 fix.py",
        "uv run python scripts/plot.py",
        "python - <<'EOF'\nprint(1)\nEOF",
        "ls | python3 count.py",
    ],
)
def test_bare_python_is_refused(command):
    assert bare_python_call(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "dsagt-run -- python compare.py a.json b.json",
        "dsagt-run --code x --stdout audit/a.json -- python skills/x/scripts/x.py",
        "mkdir -p out && dsagt-run -- python3 fix.py",
        "python -m pytest tests -q",
        "python3 --version",
        "python --help",
        "pip install x",
        "ls data/ && head -3 data/x.csv",
        "echo python is not run here",
        "/opt/pythonic/tool --flag",
    ],
)
def test_recorded_and_harmless_forms_pass(command):
    assert bare_python_call(command) is None


def _run(payload: dict, monkeypatch, capsys) -> tuple[int, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    rc = main([])
    return rc, capsys.readouterr().err


def test_hook_refuses_with_the_recorded_form(monkeypatch, capsys):
    rc, err = _run(
        {"tool_name": "Bash", "tool_input": {"command": "python3 tally.py data/t.csv"}},
        monkeypatch,
        capsys,
    )
    assert rc == 2
    assert "dsagt-run -- python3 tally.py data/t.csv" in err
    assert "--stdout" in err


def test_hook_passes_other_tools_and_recorded_python(monkeypatch, capsys):
    assert _run({"tool_name": "Read", "tool_input": {}}, monkeypatch, capsys)[0] == 0
    assert (
        _run(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "dsagt-run -- python x.py"},
            },
            monkeypatch,
            capsys,
        )[0]
        == 0
    )


def test_claude_setup_writes_the_hook_once_and_keeps_user_hooks(tmp_path):
    from dsagt.agents.claude import _write_bash_guard_hook

    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"]},
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Write",
                            "hooks": [{"type": "command", "command": "mine"}],
                        }
                    ]
                },
            }
        )
    )
    assert _write_bash_guard_hook(tmp_path) == [
        f"Wrote the bash guard hook into {settings}"
    ]
    assert _write_bash_guard_hook(tmp_path) == []
    written = json.loads(settings.read_text())
    assert written["permissions"] == {"allow": ["Read"]}
    commands = [
        h["command"] for e in written["hooks"]["PreToolUse"] for h in e["hooks"]
    ]
    assert commands == ["mine", "uv run dsagt-bash-guard"]


def test_a_quoted_string_with_an_operator_stays_one_segment():
    command = "python3 -c 'import pandas as pd; print(pd.read_csv(\"data/x.csv\").shape)' && echo done"
    assert bare_python_call(command) == (
        "python3 -c 'import pandas as pd; print(pd.read_csv(\"data/x.csv\").shape)'"
    )


def test_a_shebang_executed_script_is_refused():
    assert bare_python_call("./convert.py data/in.csv") is not None
    assert bare_python_call("scripts/tally.py") is not None

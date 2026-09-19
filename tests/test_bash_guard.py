"""The Claude Code PreToolUse hook that puts bare python under dsagt-run."""

import io
import json

import pytest

from dsagt.commands.bash_guard import bare_python_call, main, recorded_form


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
def test_bare_python_is_found(command):
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
    captured = capsys.readouterr()
    return rc, captured.err or captured.out


def test_hook_returns_the_recorded_form_as_the_updated_input(monkeypatch, capsys):
    rc, out = _run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "python3 tally.py data/t.csv", "timeout": 60000},
        },
        monkeypatch,
        capsys,
    )
    assert rc == 0
    # No permissionDecision: the user's permission rules judge the new command.
    assert json.loads(out) == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": {
                "command": "dsagt-run -- python3 tally.py data/t.csv",
                "timeout": 60000,
            },
        }
    }


@pytest.mark.parametrize(
    "command, rewritten",
    [
        (
            "cd data && python3 fix.py 2>&1 | tail -5",
            "cd data && dsagt-run -- python3 fix.py 2>&1 | tail -5",
        ),
        (
            "FOO=1 uv run python scripts/plot.py > out/plot.txt",
            "FOO=1 dsagt-run --stdout out/plot.txt -- uv run python scripts/plot.py",
        ),
        (
            "for f in data/*.csv; do python3 tally.py $f; done",
            "for f in data/*.csv; do dsagt-run -- python3 tally.py $f; done",
        ),
        ("x=$(python3 v.py)", "x=$(dsagt-run -- python3 v.py)"),
        (
            "python3 a.py && ./b.py x",
            "dsagt-run -- python3 a.py && dsagt-run -- ./b.py x",
        ),
    ],
)
def test_each_bare_call_is_wrapped_where_it_stands(command, rewritten):
    assert recorded_form(command) == rewritten


def test_a_heredoc_body_is_data():
    command = (
        "python3 - <<'EOF'\nimport os; x = 1 # note\npython nested.py\nEOF\necho done"
    )
    assert recorded_form(command) == "dsagt-run -- " + command


@pytest.mark.parametrize(
    "command",
    [
        'echo "$(python3 v.py)"',
        "ls *.csv | xargs python3 tally.py",
        "sh -c 'python x.py'",
        "python3 -c 'print(1",
    ],
)
def test_a_call_that_cannot_be_wrapped_in_place_is_refused(
    command, monkeypatch, capsys
):
    rc, err = _run(
        {"tool_name": "Bash", "tool_input": {"command": command}}, monkeypatch, capsys
    )
    assert rc == 2
    assert "dsagt-run -- python" in err


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
    assert commands[0] == "mine"
    # The guard's absolute path, beside the interpreter running dsagt.
    assert commands[1].startswith("/") and commands[1].endswith("/dsagt-bash-guard")
    # An entry from an earlier install is replaced, not duplicated.
    written["hooks"]["PreToolUse"][1]["hooks"][0][
        "command"
    ] = "/old/bin/dsagt-bash-guard"
    settings.write_text(json.dumps(written))
    assert _write_bash_guard_hook(tmp_path)
    again = json.loads(settings.read_text())
    commands = [h["command"] for e in again["hooks"]["PreToolUse"] for h in e["hooks"]]
    assert len(commands) == 2 and commands[1] != "/old/bin/dsagt-bash-guard"


def test_a_quoted_string_with_an_operator_stays_one_segment():
    command = "python3 -c 'import pandas as pd; print(pd.read_csv(\"data/x.csv\").shape)' && echo done"
    assert bare_python_call(command) == (
        "python3 -c 'import pandas as pd; print(pd.read_csv(\"data/x.csv\").shape)'"
    )


def test_a_shebang_executed_script_is_found():
    assert bare_python_call("./convert.py data/in.csv") is not None
    assert bare_python_call("scripts/tally.py") is not None


def test_the_config_opt_out_removes_the_hook_and_keeps_user_hooks(tmp_path):
    from dsagt.agents.claude import _write_bash_guard_hook

    settings = tmp_path / ".claude" / "settings.json"
    assert _write_bash_guard_hook(tmp_path, enabled=False) == []
    assert not settings.exists()
    _write_bash_guard_hook(tmp_path)
    written = json.loads(settings.read_text())
    written["hooks"]["PreToolUse"].insert(
        0, {"matcher": "Write", "hooks": [{"type": "command", "command": "mine"}]}
    )
    settings.write_text(json.dumps(written))
    assert _write_bash_guard_hook(tmp_path, enabled=False) == [
        f"Removed the bash guard hook from {settings}"
    ]
    commands = [
        h["command"]
        for e in json.loads(settings.read_text())["hooks"]["PreToolUse"]
        for h in e["hooks"]
    ]
    assert commands == ["mine"]


def test_a_recorded_run_asking_for_more_than_the_ceiling_is_refused(
    monkeypatch, capsys
):
    call = {"command": "dsagt-run -- bash assemble_all.sh", "timeout": 2400000}
    rc, err = _run({"tool_name": "Bash", "tool_input": call}, monkeypatch, capsys)
    assert rc == 2 and "one sample per call" in err
    call["timeout"] = 600000
    assert _run({"tool_name": "Bash", "tool_input": call}, monkeypatch, capsys)[0] == 0
    other = {"command": "sleep 5", "timeout": 2400000}
    assert _run({"tool_name": "Bash", "tool_input": other}, monkeypatch, capsys)[0] == 0


@pytest.mark.parametrize("keyword", ["until", "while", "if", "time"])
def test_a_call_after_a_loop_or_condition_keyword_is_wrapped(keyword):
    assert recorded_form(f"{keyword} python3 poll.py; do sleep 1; done").startswith(
        f"{keyword} dsagt-run -- python3 poll.py"
    )

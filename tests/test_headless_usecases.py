"""Tests for the headless walkthrough driver's prompt parsing and command shapes."""

import pytest

from tests import headless_usecases as hu

README = """\
# Demo

## Setup

```bash
dsagt init demo
```

## Execution

Prompt one:

```text
Build the pipeline.
```

```bash
not a prompt
```

```text
Show the records.
```

## Post-Conditions

```text
not a prompt either
```
"""


def test_prompts_are_the_text_fences_between_the_headings(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert hu.prompts_from(readme) == ["Build the pipeline.", "Show the records."]


def test_missing_heading_raises(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Demo\n\n## Execution\n\n```text\nx\n```\n")
    with pytest.raises(ValueError, match="Post-Conditions"):
        hu.prompts_from(readme)


def test_claude_first_prompt_starts_and_later_prompts_continue():
    first = hu.claude_command("hi", model=None, first=True)
    later = hu.claude_command("hi", model="claude-opus-5", first=False)
    assert first[:5] == ["claude", "-p", "hi", "--model", hu.CLAUDE_DEFAULT_MODEL]
    assert "--continue" not in first
    assert later[:6] == ["claude", "-p", "hi", "--model", "claude-opus-5", "--continue"]
    assert "mcp__dsagt" in later[later.index("--allowedTools") :]


def test_codex_first_prompt_execs_and_later_prompts_resume_last():
    first = hu.codex_command("hi", model=None, first=True)
    later = hu.codex_command("hi", model="gpt-5", first=False)
    assert first == [
        "codex", "exec",
        "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
        "hi",
    ]  # fmt: skip
    assert later == [
        "codex", "exec", "resume", "--last",
        "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
        "-m", "gpt-5",
        "hi",
    ]  # fmt: skip


def test_every_supported_agent_has_a_command_builder():
    assert set(hu.COMMANDS) == {"claude", "codex"}

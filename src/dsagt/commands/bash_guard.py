"""dsagt-bash-guard: the Claude Code PreToolUse hook that puts a bare python call under ``dsagt-run``.

Claude Code calls it before every Bash tool call with the call as JSON on
stdin.  A command word that runs ``python``, ``python3``, ``uv run python``
or a ``.py`` file outside ``dsagt-run`` gets ``dsagt-run --`` inserted in
front of it, and the hook returns the rewritten command as the call's
``updatedInput``; Claude Code then applies its own permission rules to the
rewritten command.  A trailing ``> path`` on the call becomes ``--stdout
path``, so the file is in the record.  The rest of the line (a ``cd``, a
pipe, a heredoc, a loop around the call) is kept as written: ``dsagt-run``
passes stdin and stdout through and keeps a copy of a heredoc's script.

A python call the scan cannot wrap in place (inside a quoted ``$(...)``,
after ``xargs``, in a ``sh -c`` string) is refused (exit 2, the reason on
stderr, which Claude Code shows the agent) with the recorded form to use.

The hook exists because the agent chooses a command at the moment it issues
one, without rereading the instructions.  In the headless runs a bare run
had four triggers: a skill's text naming a bare command, a usage line the
agent had written into its own skill, a quick comparison as a heredoc or
``python -c``, and a script written to the scratchpad after Claude Code's
shell check refused a multi-line ``python -c``.

A recorded run that asks for a timeout above Claude Code's ten-minute limit
on one shell command is refused with the two forms that finish: in the
headless runs the agent asked for 40 minutes, the harness ended the loop at
ten, and 3 of 11 samples were done.

Left as they are: a call already under ``dsagt-run``, ``--help`` and
``--version``, ``python -m pytest``, and ``pip``.
"""

from __future__ import annotations

import json
import re
import sys

# A command word that runs python, after shell keywords and VAR=value prefixes.
_PYTHON = re.compile(
    r"\s*(?:(?:do|then|else|elif|if|while|until|time|!|\{)\s+)*(?:\w+=\S*\s+)*"
    r"(?P<word>(?:uv\s+run\s+)?python3?(?=\s|$)|\.{0,2}/?[^\s;&|()<>]+\.py(?=\s|$))"
)
_ALLOWED = re.compile(r"python3?\s+(?:-m\s+pytest|-m\s+pip|--version|--help)\b")
_HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)(\w+)\2")
# ``> path`` closing a call: not ``>>``, ``2>``, ``>&``, or a quoted path.
_STDOUT_REDIRECT = re.compile(r"(?<![0-9&>])>\s*([^\s;&|<>'\"]+)\s*$")
# A python call where no command starts that the scan finds.
_UNWRAPPABLE = re.compile(
    r"(?:\$\(|`|\bxargs\s+(?:-\S+\s+)*|\bsh\s+-l?c\s+['\"]\s*)"
    r"(?:\w+=\S*\s+)*python3?(?=\s)(?!\s+(?:-m\s+pytest|-m\s+pip|--version|--help)\b)"
)


#: Claude Code's limit on one Bash call, in the unit of ``tool_input.timeout``.
CEILING_MS = 600_000


def command_spans(command: str) -> list[tuple[int, int]] | None:
    """The ``(start, end)`` offsets of each simple command in a shell line.

    Commands are separated by ``;``, ``&&``, ``||``, ``|``, ``&``, a newline
    and ``(``.  Quoted strings and ``#`` comments are passed over, a
    redirect's ``&`` (``2>&1``, ``&>``) separates nothing, and a heredoc's
    body is skipped to its delimiter line, so text that is data is never
    read as a command.  ``None`` when a quote is left open.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    pending: list[tuple[str, bool]] = []
    n = len(command)
    while i < n:
        c = command[i]
        if c == "\\":
            i += 2
            continue
        if c in "'\"":
            close = i + 1
            while close < n and command[close] != c:
                if c == '"' and command[close] == "\\":
                    close += 1
                close += 1
            if close >= n:
                return None
            i = close + 1
            continue
        if c == "#" and (i == 0 or command[i - 1].isspace()):
            while i < n and command[i] != "\n":
                i += 1
            continue
        if command.startswith("<<<", i):
            i += 3
            continue
        heredoc = _HEREDOC.match(command, i)
        if heredoc:
            pending.append((heredoc.group(3), heredoc.group(1) == "-"))
            i = heredoc.end()
            continue
        if c == "&" and (
            (i > 0 and command[i - 1] in "<>") or command.startswith("&>", i)
        ):
            i += 1
            continue
        if c in ";&|\n(":
            spans.append((start, i))
            i += 1
            if c == "\n":
                for delimiter, strip_tabs in pending:
                    while i < n:
                        line_end = command.find("\n", i)
                        line_end = n if line_end == -1 else line_end
                        line = command[i:line_end]
                        i = min(line_end + 1, n)
                        if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                            break
                pending = []
            start = i
            continue
        i += 1
    spans.append((start, n))
    return [(s, e) for s, e in spans if command[s:e].strip()]


def _bare_python_spans(command: str) -> list[tuple[int, int, int]] | None:
    """``(start, end, word)`` of each command in *command* that runs python
    outside ``dsagt-run``; *word* is the offset of the python command word."""
    spans = command_spans(command)
    if spans is None:
        return None
    found = []
    for start, end in spans:
        segment = command[start:end]
        match = _PYTHON.match(segment)
        if match is None or "dsagt-run" in segment or _ALLOWED.search(segment):
            continue
        found.append((start, end, start + match.start("word")))
    return found


def bare_python_call(command: str) -> str | None:
    """The first command in *command* that runs python outside ``dsagt-run``,
    or ``None``.  A line with an open quote is judged as one command."""
    found = _bare_python_spans(command)
    if found is None:
        bare = _PYTHON.match(command) and "dsagt-run" not in command
        return command.strip() if bare else None
    if not found:
        return None
    start, end, _ = found[0]
    return command[start:end].strip()


def recorded_form(command: str) -> str | None:
    """*command* with each bare python call under ``dsagt-run``, or ``None``
    when it holds none.

    Raises ``ValueError`` for a python call that cannot be wrapped in place.
    """
    found = _bare_python_spans(command)
    if found is None:
        if bare_python_call(command):
            raise ValueError("the command has an open quote")
        return None
    # Right to left, so the earlier offsets stay valid.
    for start, end, word in reversed(found):
        call = command[word:end]
        wrapper = "dsagt-run -- "
        redirect = _STDOUT_REDIRECT.search(call)
        if redirect and "<<" not in call:
            wrapper = f"dsagt-run --stdout {redirect.group(1)} -- "
            call = call[: redirect.start()].rstrip() + call[redirect.end() :]
        command = command[:word] + wrapper + call + command[end:]
    unwrappable = _UNWRAPPABLE.search(command)
    if unwrappable:
        raise ValueError(
            f"`{unwrappable.group(0).strip()}` runs inside another command"
        )
    return command if found else None


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command", "")
    if (tool_input.get("timeout") or 0) > CEILING_MS and (
        "dsagt-run" in command or bare_python_call(command)
    ):
        print(
            "dsagt: Claude Code ends one shell command at ten minutes whatever "
            "timeout is asked for, and a run it ends loses its remaining work. "
            "Split this run into calls that each finish under ten minutes (one "
            "sample per call, not a loop over all of them), or give it to a "
            "subagent and wait for its result.",
            file=sys.stderr,
        )
        return 2
    try:
        rewritten = recorded_form(tool_input.get("command", ""))
    except ValueError as err:
        print(
            f"dsagt: bare python leaves no execution record, and {err}. Run the "
            "python call on its own as `dsagt-run -- python ...` (or the "
            "registered code's stored command); use --stdout <path> for a "
            "report the command prints.",
            file=sys.stderr,
        )
        return 2
    if rewritten is None:
        return 0
    # No permissionDecision: Claude Code judges the rewritten command by the
    # user's own permission rules.
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": {**tool_input, "command": rewritten},
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

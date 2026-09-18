"""dsagt-bash-guard: the Claude Code PreToolUse hook that refuses a bare python call.

Claude Code calls it before every Bash tool call with the call as JSON on
stdin.  A command that runs ``python``, ``python3``, or ``uv run python``
outside ``dsagt-run`` is refused (exit 2, the reason on stderr, which Claude
Code shows the agent) with the recorded form to use instead.  The hook
exists because the instructions alone are not read at the moment that
matters: the agent's inspect-and-fix loop wrote scripts to its scratchpad
and ran them bare after the harness refused a ``python -c`` with ``#``
lines.  The hook acts at the point the command is issued.

Allowed as they are: a call already under ``dsagt-run``, ``--help`` and
``--version``, ``python -m pytest``, and ``pip``.
"""

from __future__ import annotations

import json
import re
import sys

# The segment's command word, after any VAR=value prefixes.
_PYTHON = re.compile(
    r"^\s*(?:\w+=\S*\s+)*(?:(?:uv\s+run\s+)?python3?(?=\s|$)|\.{0,2}/?\S+\.py(?=\s|$))"
)
_ALLOWED = re.compile(r"python3?\s+(?:-m\s+pytest|-m\s+pip|--version|--help)\b")


def _segments(command: str) -> list[str]:
    """The parts of a shell line between ``;``, ``&&``, ``||`` and ``|``,
    with quoted strings kept whole, so a ``python -c '...; ...'`` is one
    segment and the form the refusal suggests is the whole call."""
    import shlex

    lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|")
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    try:
        for token in lexer:
            if token in (";", "&&", "||", "|", "&", ";;"):
                segments.append([])
            else:
                segments[-1].append(token)
    except ValueError:
        # An unbalanced quote: one segment, judged as a whole.
        return [command]
    return [" ".join(seg) for seg in segments if seg]


def bare_python_call(command: str) -> str | None:
    """The segment of *command* that runs python outside ``dsagt-run``, or ``None``.

    A segment whose python is preceded by ``dsagt-run`` on the same segment
    is the recorded form and passes.
    """
    for segment in _segments(command):
        if "dsagt-run" in segment:
            continue
        if not _PYTHON.search(segment):
            continue
        if _ALLOWED.search(segment):
            continue
        return segment.strip()
    return None


#: Where Claude Code puts a session's temporary files.  A script written
#: there is outside the project, so a record naming it replays only on the
#: machine that ran it while the directory still exists.
_SCRATCHPAD = re.compile(r"/(?:private/)?tmp/claude-[^\s'\"]*/scratchpad/")


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    if _SCRATCHPAD.search(command):
        print(
            "dsagt: the scratchpad is outside the project, and a run that names a "
            "file there replays nowhere else. Write the script under "
            "skills/<name>/scripts/ and run it from the project directory.",
            file=sys.stderr,
        )
        return 2
    segment = bare_python_call(command)
    if segment is None:
        return 0
    print(
        "dsagt: bare python leaves no execution record. Run it as "
        f"`dsagt-run -- {segment}` (or the registered code's stored command); "
        "use --stdout <path> for a report the command prints.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

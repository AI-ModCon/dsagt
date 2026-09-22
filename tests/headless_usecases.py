"""Drive a use-case README's Execution prompts through a headless agent.

Usage::

    uv run --no-sync python tests/headless_usecases.py <use_case_dir> <project>
        [--from N] [--only N,M] [--subst KEY=VALUE ...] [--model M]
        [--log PATH] [--timeout SECONDS]

``<project>`` is a registered dsagt project (``dsagt init <name> --agent
claude|codex``); its config names the agent and the working directory, and
``agent_env`` supplies the per-project runtime env (``CODEX_HOME``), the same
way ``dsagt start`` does.  Prompts are the ```text fences under
``## Execution`` and before ``## Post-Conditions`` in the README, numbered
from 1.  The first prompt starts a session and every later one continues it
(``claude -p --continue``; ``codex exec resume --last``, which under the
per-project ``CODEX_HOME`` is this project's most recent session).  Each
response is appended to the log under a header so the run can be reviewed
afterwards; the driver stops at the first non-zero exit.

Claude runs under a scoped tool allowlist with ``acceptEdits``.  Codex runs
with ``--dangerously-bypass-approvals-and-sandbox``, the same flags
``CodexSetup.run_script`` uses, because the dsagt MCP server and the codes
write under ``~/dsagt-projects/`` outside the workspace sandbox.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dsagt.agents import agent_env
from dsagt.session import load_config

#: Tools Claude Code may use without a prompt.  ``mcp__dsagt`` is every
#: dsagt tool; the Bash entries are the commands the walkthroughs run.
CLAUDE_ALLOWED_TOOLS = [
    "mcp__dsagt",
    "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "LS",
    # Skill runs an installed skill; Task is the subagent a long code run may
    # wait on; TodoWrite is the planning list a multi-step prompt opens.
    "Skill", "Task", "TodoWrite",
    "Bash(./*)", "Bash(pip:*)", "Bash(.venv*/bin/*)",
    "Bash(python:*)", "Bash(python3:*)", "Bash(uv run:*)", "Bash(uv pip:*)",
    "Bash(dsagt-run:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)",
    "Bash(wc:*)", "Bash(mkdir:*)", "Bash(cp:*)", "Bash(mv:*)", "Bash(find:*)",
    "Bash(grep:*)", "Bash(diff:*)", "Bash(chmod:*)", "Bash(echo:*)", "Bash(pwd:*)",
    "Bash(bash:*)", "Bash(sh:*)", "Bash(file:*)", "Bash(du:*)", "Bash(md5:*)",
    "Bash(aidrin:*)", "Bash(fastp:*)", "Bash(megahit:*)", "Bash(git:*)",
    "Bash(h5dump:*)", "Bash(h5ls:*)", "Bash(gunzip:*)", "Bash(zcat:*)", "Bash(gzip:*)",
    "Bash(tar:*)", "Bash(sort:*)", "Bash(cut:*)", "Bash(awk:*)", "Bash(sed:*)",
    "Bash(tr:*)", "Bash(xargs:*)", "Bash(env:*)", "Bash(which:*)", "Bash(true:*)",
    "Bash(test:*)", "Bash(seq:*)", "Bash(date:*)", "Bash(basename:*)", "Bash(dirname:*)",
    "Bash(cd:*)", "Bash(export:*)", "Bash(sqlite3:*)", "Bash(jq:*)",
    "Bash(*/.tools/*)", "Bash(~/dsagt-projects/.tools/*)",
    # A script the agent wrote to Claude Code's session scratchpad and runs by
    # its absolute path; with a person present this is an approval prompt.
    "Bash(/private/tmp/claude-*)", "Bash(/tmp/claude-*)",
]  # fmt: skip

CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-5"


def prompts_from(readme: Path) -> list[str]:
    """The ```text fences between ``## Execution`` and ``## Post-Conditions``.

    Raises ``ValueError`` when either heading is missing, since a README
    without them has no prompt sequence to drive.
    """
    text = readme.read_text()
    try:
        start = text.index("\n## Execution")
        end = text.index("\n## Post-Conditions", start)
    except ValueError as err:
        raise ValueError(
            f"{readme} needs '## Execution' followed by '## Post-Conditions'"
        ) from err
    body = text[start:end]
    return [
        m.group(1).strip() for m in re.finditer(r"```text\n(.*?)```", body, flags=re.S)
    ]


def selected_prompts(count: int, from_n: int, only: set[int] | None) -> list[int]:
    """The prompt numbers to run, in order.

    The first of them starts the session; every later one continues it, so
    ``--only 2,3`` and ``--from 2`` continue a session the earlier prompts
    opened, while any selection that begins at prompt 1 starts a new one.
    """
    return [
        n for n in range(1, count + 1) if n >= from_n and (only is None or n in only)
    ]


def set_shell_timeouts(env: dict, timeout: int) -> dict:
    """Let a shell command run for the whole prompt timeout.

    Claude Code ends one by moving it to the background, and a headless turn
    that then ends takes the command with it.  Two limits decide when:
    ``BASH_DEFAULT_TIMEOUT_MS`` (two minutes) is what a call that asks for no
    timeout gets, which is most of them, and ``BASH_MAX_TIMEOUT_MS`` (ten
    minutes) caps what a call may ask for, so both have to rise.  A person's
    session stays open, so only the driver needs the larger limits.
    """
    for name in ("BASH_DEFAULT_TIMEOUT_MS", "BASH_MAX_TIMEOUT_MS"):
        env.setdefault(name, str(timeout * 1000))
    return env


def claude_command(prompt: str, *, model: str | None, first: bool) -> list[str]:
    cmd = ["claude", "-p", prompt, "--model", model or CLAUDE_DEFAULT_MODEL]
    if not first:
        cmd.append("--continue")
    cmd += [
        "--permission-mode", "acceptEdits",
        "--output-format", "text",
        "--allowedTools", *CLAUDE_ALLOWED_TOOLS,
    ]  # fmt: skip
    return cmd


def codex_command(prompt: str, *, model: str | None, first: bool) -> list[str]:
    cmd = ["codex", "exec"]
    if not first:
        cmd += ["resume", "--last"]
    cmd += ["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
    if model:
        cmd += ["-m", model]
    cmd.append(prompt)
    return cmd


#: Agent name (as in ``.dsagt/config.yaml``) to its headless command builder.
COMMANDS = {"claude": claude_command, "codex": codex_command}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("use_case_dir", help="directory holding the walkthrough README.md")
    ap.add_argument("project", help="registered dsagt project name")
    ap.add_argument(
        "--from", dest="from_n", type=int, default=1, help="first prompt number"
    )
    ap.add_argument("--only", default=None, help="comma-separated prompt numbers")
    ap.add_argument("--subst", action="append", default=[], metavar="KEY=VALUE",
                    help="literal replacement applied to every prompt")  # fmt: skip
    ap.add_argument(
        "--model",
        default=None,
        help="agent model (claude default: %s)" % CLAUDE_DEFAULT_MODEL,
    )
    ap.add_argument(
        "--log", default=None, help="default <project_dir>/headless_run.log"
    )
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per prompt")
    args = ap.parse_args()

    config = load_config(args.project)
    agent = config["agent"]
    if agent not in COMMANDS:
        raise SystemExit(
            f"project {args.project!r} uses agent {agent!r}; "
            f"the driver supports {sorted(COMMANDS)}"
        )
    build_command = COMMANDS[agent]
    env = agent_env(config)
    set_shell_timeouts(env, args.timeout)
    project_dir = Path(config["project_dir"])

    prompts = prompts_from(Path(args.use_case_dir) / "README.md")
    subs = dict(s.split("=", 1) for s in args.subst)
    only = {int(x) for x in args.only.split(",")} if args.only else None
    log = Path(args.log) if args.log else project_dir / "headless_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    selected = selected_prompts(len(prompts), args.from_n, only)
    first = bool(selected) and selected[0] == 1
    for n, prompt in enumerate(prompts, start=1):
        if n not in selected:
            continue
        for key, value in subs.items():
            prompt = prompt.replace(key, value)
        cmd = build_command(prompt, model=args.model, first=first)
        first = False
        t0 = time.time()
        with log.open("a") as fh:
            fh.write(
                f"\n\n===== PROMPT {n}/{len(prompts)}  {datetime.now():%H:%M:%S}"
                f"  agent={agent}\n{prompt}\n----- RESPONSE\n"
            )
            fh.flush()
            try:
                # codex exec reads a non-terminal stdin to the end before it
                # starts, so an inherited pipe that stays open blocks the run.
                rc = subprocess.run(
                    cmd, cwd=project_dir, env=env, stdin=subprocess.DEVNULL,
                    stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout,
                ).returncode  # fmt: skip
            except subprocess.TimeoutExpired:
                fh.write(f"\n[driver] TIMEOUT after {args.timeout}s\n")
                rc = 124
            fh.write(f"\n----- END PROMPT {n}  rc={rc}  {time.time() - t0:.0f}s\n")
        print(
            f"[driver] prompt {n}/{len(prompts)} rc={rc} {time.time() - t0:.0f}s",
            flush=True,
        )
        if rc != 0:
            print(f"[driver] stopping: prompt {n} exited {rc}; see {log}", flush=True)
            return rc
    print(f"[driver] done; log at {log}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

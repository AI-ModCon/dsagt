"""
DSAgt CLI — project initialization and session management.

``dsagt init`` is the single, interactive, re-runnable place a user expresses
every choice; the prompts mirror ``.dsagt/config.yaml`` 1:1 and it writes the
per-agent instructions + MCP config.  ``dsagt start <project>`` is pure
convenience — ``cd <project> && <agent>`` and nothing else (the MCP server
owns the session lifecycle, minting session ids into ``.dsagt/state.yaml`` and
catching up post-session extraction in the background at startup).

The agent talks to its provider directly — DSAGT never interposes on its
traffic.  Self-logging goes to a serverless ``sqlite:///<pdir>/mlflow.db``
store (no server to run).

Usage:
    dsagt init [<project>]              # interactive; re-run to reconfigure
    dsagt init <project> --agent <platform> [--location <path>]
                         [--include <asset>... | --exclude <asset>...]   # non-interactive
    dsagt start <project>
    dsagt info <project> [--json]
    dsagt smoke-test
    dsagt list
    dsagt mv <project> <location>
    dsagt rm <project> [-y] [--keep-files]
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dsagt.agents import (
    agent_env,
    dynamic_agent_record,
    launch_agent,
    static_agent_record,
    AGENTS,
)
from dsagt.session import (
    DEFAULT_PROJECTS_BASE,
    VALID_AGENTS,
    list_projects,
    load_config,
    init_project,
    move_project,
    project_dir,
    read_config_file,
    remove_collection,
    remove_project,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactive prompt helpers
#
# Selection-style prompts (agent, KB collections, skill sources) use
# ``questionary`` for arrow-key navigation + space-to-toggle checkboxes — no
# typing required.  Free-text (name / location) and y/N confirms stay plain.
# These run only on the interactive (TTY) path; automation drives init via
# flags and never reaches them.
# ---------------------------------------------------------------------------


def _prompt(text: str, default: str | None = None) -> str:
    """Free-text prompt; empty input returns *default*."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    resp = input(f"{text}{suffix}: ").strip()
    return resp or (default or "")


def _confirm(text: str, default: bool = False) -> bool:
    """Yes/no prompt; empty input returns *default*."""
    hint = "Y/n" if default else "y/N"
    resp = input(f"{text} [{hint}] ").strip().lower()
    if not resp:
        return default
    return resp in ("y", "yes")


def _select(message: str, options: list[str], default: str) -> str:
    """Single-select menu (arrow keys).  Aborts init on cancel (Ctrl-C)."""
    import questionary

    answer = questionary.select(message, choices=options, default=default).ask()
    if answer is None:
        raise SystemExit("dsagt init: cancelled.")
    return answer


def _checkbox(message: str, choices: list[tuple[str, str, bool]]) -> list[str]:
    """Multi-select checkbox menu (space toggles, enter confirms).

    *choices* is a list of ``(value, label, checked)``.  Returns the selected
    values.  Aborts init on cancel (Ctrl-C).
    """
    import questionary

    qchoices = [
        questionary.Choice(title=label, value=value, checked=checked)
        for value, label, checked in choices
    ]
    # ``instruction=""`` suppresses questionary's own hint line — our message
    # already spells out the controls.
    answer = questionary.checkbox(message, choices=qchoices, instruction="").ask()
    if answer is None:
        raise SystemExit("dsagt init: cancelled.")
    return answer


def _current_assets(pdir: Path) -> list[str]:
    """Asset names whose collection already exists in the project's kb_index."""
    from dsagt.commands.setup_core_kb import all_assets, asset_collection_name

    present = []
    for asset in all_assets():
        try:
            coll = asset_collection_name(asset)
        except ValueError:
            continue
        if (pdir / "kb_index" / coll).exists():
            present.append(asset)
    return present


def _skills_block_for(source_names: list[str]) -> dict:
    """Build the ``skills`` config block from the selected skill-source names."""
    from dsagt.skills import KNOWN_SOURCES

    sources = []
    for name in source_names:
        src = KNOWN_SOURCES[name]
        sources.append(
            {
                "name": name,
                "url": src["url"],
                "branch": src.get("branch", "main"),
                "subdir": src.get("subdir"),
            }
        )
    return {"sources": sources}


def _episodic_block(enabled: bool) -> dict | None:
    """The ``episodic`` config block, or ``None`` when the user didn't opt in.

    Enabling captures each completed turn into ``session_memory`` (mechanical
    chunk + tag + embed).  ``None`` keeps a disabled project's config minimal
    (``enabled: false`` is backfilled on read).
    """
    if not enabled:
        return None
    return {"enabled": True}


def _collect_settings(args, interactive: bool, existing: dict, pdir: Path | None):
    """Resolve the init choices (the 1:1 mirror of the config).

    Selection questions: agent platform, packaged KB document *collections*,
    skill-catalog *sources*, and the episodic-memory opt-in.  The bundled
    ``tools`` collection is always provisioned and is NOT a per-project choice.
    Project name + folder location are resolved by the caller.  Embedding /
    chunk_size / rerank are code defaults, not init choices.

    Interactive: questionary select/checkbox menus + y/N, pre-filled with the
    project's current choices on re-init.  Non-interactive (no TTY): drive from
    ``--include`` / ``--exclude`` / ``--episodic`` flags — the automation/test
    path.
    """
    from dsagt.commands.setup_core_kb import COLLECTIONS, resolve_assets
    from dsagt.skills import KNOWN_SOURCES

    coll_choices = list(COLLECTIONS)
    skill_choices = list(KNOWN_SOURCES)

    if interactive:
        agent = _select(
            "Agent platform",
            list(VALID_AGENTS),
            default=existing.get("agent") or args.agent or "claude",
        )

        # Knowledge collections (heavy doc collections; default none).
        # Labels are bare names — short enough to never wrap the terminal.
        cur_colls = set(existing.get("knowledge", {}).get("collections", []))
        collections = _checkbox(
            "Knowledge collections (space toggles, ↑/↓ to move, enter confirms)",
            [(c, c, c in cur_colls) for c in coll_choices],
        )

        # Skill-catalog sources (default genesis on a fresh project).
        cur_srcs = set(
            s["name"] for s in existing.get("skills", {}).get("sources", [])
        ) or {"genesis"}
        skill_names = _checkbox(
            "Skill catalog sources (space toggles, ↑/↓ to move, enter confirms)",
            [(s, s, s in cur_srcs) for s in skill_choices],
        )

        # Episodic memory (opt-in): captures session turns into session_memory.
        cur_epi = existing.get("episodic", {}) or {}
        enable_epi = _confirm(
            "Enable episodic memory? (captures session turns into searchable "
            "memory)",
            default=bool(cur_epi.get("enabled")),
        )
        episodic = _episodic_block(enable_epi)
    else:
        agent = args.agent or existing.get("agent")
        if not agent:
            raise SystemExit("dsagt init: --agent is required (non-interactive).")
        # --include / --exclude pick the full asset set; split it.
        full = resolve_assets(include=args.include, exclude=args.exclude)
        collections = [a for a in full if a in COLLECTIONS]
        skill_names = [a for a in full if a in KNOWN_SOURCES]
        # Episodic is flag-driven here (automation); omit --episodic to leave it
        # off.  Re-pass it on re-init — like --include/--exclude, flags are
        # authoritative on the non-interactive path.
        episodic = _episodic_block(getattr(args, "episodic", False))

    return {
        "agent": agent,
        # The bundled ``tools`` collection is always provisioned.
        "assets": ["codes", *collections, *skill_names],
        "knowledge": {"collections": collections},
        "skills": _skills_block_for(skill_names),
        "episodic": episodic,
    }


def _handle_destructive(
    existing: dict, settings: dict, pdir: Path, interactive: bool
) -> None:
    """Detect destructive deltas on re-init and prompt delete-or-keep.

    Non-interactive (no TTY): never deletes — warns and keeps, so automation
    can't lose data and ``input()`` is never called on a closed stdin.

    Never touches agent-populated data: ``tool_use`` / ``session_memory``
    collections, ``.dsagt/`` memory, ``trace_archive/``, ``skills/``.
    """
    from dsagt.commands.setup_core_kb import asset_collection_name

    protected = {"code_use", "session_memory"}

    # Agent switch → old platform's files are now stale.
    old_agent = existing.get("agent")
    new_agent = settings["agent"]
    if old_agent and old_agent != new_agent:
        setup = AGENTS[old_agent]()
        stale = [p for p in setup.owned_artifacts(pdir) if p.exists()]
        if stale:
            print(f"\n  Switching agent {old_agent} → {new_agent} leaves stale files:")
            for p in stale:
                print(f"    {p}")
            if interactive and _confirm("  Delete these stale files?", default=True):
                import shutil

                for p in stale:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                print("  Deleted.")
            else:
                print("  Kept (remove them manually if you want them gone).")

    # Removed KB collections → their dirs are now orphaned.
    new_colls = set()
    for a in settings["assets"]:
        try:
            new_colls.add(asset_collection_name(a))
        except ValueError:
            pass
    for a in _current_assets(pdir):
        try:
            coll = asset_collection_name(a)
        except ValueError:
            continue
        if coll in new_colls or coll in protected:
            continue
        if interactive and _confirm(
            f"\n  Collection '{coll}' was dropped from the asset set. Remove it?",
            default=False,
        ):
            if remove_collection(pdir, coll):
                print(f"  Removed kb_index/{coll}.")
        else:
            print(
                f"\n  Collection '{coll}' was dropped from the asset set "
                "(kept on disk)."
            )

    # Embedding backend/model change invalidates existing embeddings.
    old_emb = existing.get("embedding") or {}
    new_emb = settings["embedding"]
    changed = old_emb.get("backend") != new_emb.get("backend") or old_emb.get(
        "model"
    ) != new_emb.get("model")
    if changed and any((pdir / "kb_index").glob("*/")):
        print(
            "\n  WARNING: embedding backend/model changed.  Existing collections "
            "were embedded with the old settings and KB search will degrade until "
            "they are re-provisioned (delete + re-add the affected assets)."
        )


def _cmd_init(args):
    """Create or reconfigure a BYOA project — interactive and re-runnable.

    ``dsagt init`` is the single place a user expresses every choice; the
    prompts mirror ``.dsagt/config.yaml`` 1:1.  On an existing project it
    becomes a settings editor (prompts prefilled with current values) and
    prompts before any destructive change (agent switch, removed collection).
    Non-interactive (no TTY) drives from flags — the automation/test path.
    """
    interactive = sys.stdin.isatty()

    # Project name
    name = args.project
    if interactive and not name:
        name = _prompt("Project name")
    if not name:
        raise SystemExit("dsagt init: project name required.")

    # Existing project? → re-init (settings editor).
    try:
        existing_pdir = project_dir(name)
    except FileNotFoundError:
        existing_pdir = None
    existing = read_config_file(existing_pdir) if existing_pdir else {}
    reinit = bool(existing)

    # Location (first init only; re-init keeps the registered path).  The
    # prompt collects the full project directory and defaults to one that
    # already ends in the project name.  If the user types a path that ends
    # in the project name we take it as-is; otherwise we append the name —
    # so both "~/proj/myproj" and "~/proj" land at "~/proj/myproj".
    if reinit:
        location = existing_pdir.parent
        pdir_preview = existing_pdir
    else:
        if interactive:
            default_full = str(DEFAULT_PROJECTS_BASE / name)
            entered = Path(_prompt("Project location", default=default_full)).resolve()
            proj_dir = entered if entered.name == name else entered / name
            location = proj_dir.parent
        else:
            location = Path(args.location).resolve() if args.location else None
        pdir_preview = (location or DEFAULT_PROJECTS_BASE) / name

    settings = _collect_settings(args, interactive, existing, pdir_preview)

    if reinit:
        _handle_destructive(existing, settings, existing_pdir, interactive)

    include = settings["assets"] if settings["assets"] else None
    exclude = ["all"] if not settings["assets"] else None
    pdir = init_project(
        name,
        settings["agent"],
        location=location,
        include=include,
        exclude=exclude,
        knowledge=settings["knowledge"],
        skills=settings["skills"],
        episodic=settings["episodic"],
    )

    agent = settings["agent"]
    config = load_config(name)

    # 1. Actions first (Wrote … / Mirrored …).
    print()
    for action in static_agent_record(config, agent, pdir):
        print(action)
    # Pass the user's shell env so per-agent ``write_dynamic`` can read
    # provider creds (e.g., cline.write_dynamic invokes ``cline auth``).
    for action in dynamic_agent_record(config, env=dict(os.environ), working_dir=pdir):
        print(action)

    # 2. Project summary.
    print()
    print(f"Project directory:  {pdir}")
    print(f"Agent:              {agent}")
    print(f"Trace store:        sqlite:///{pdir}/mlflow.db")

    # 3. Startup instructions.
    print()
    print(f"Start {agent} in the project directory, or run:")
    print(f"  dsagt start {name}")
    if AGENTS[agent]().vscode_hint(pdir):
        print()
        print(
            f"Or open the project directory in VS Code and start the "
            f"{agent} extension"
        )


def _cmd_start(args):
    """Pure convenience: ``cd <project> && <agent>``.

    Nothing more — the agent is launched in the foreground at the project
    dir.  All configuration is owned by ``dsagt init``; the session
    lifecycle (session-id minting, post-session extraction catch-up) is
    owned by the MCP server at startup.  ``dsagt start`` has no behavior the
    user couldn't get by hand with ``cd <pdir> && <agent>``.

    The per-project runtime env (e.g. ``CLINE_DIR`` / ``CODEX_HOME`` that
    point an agent at its init-written config) is still applied so the
    launched agent finds what ``init`` set up.
    """
    config = load_config(args.project)
    pdir = Path(config["project_dir"])

    print(f"  Project:  {config['project']}")
    print(f"  Agent:    {config['agent']}")
    print(f"  Dir:      {pdir}")
    print()

    env = agent_env(config)
    return launch_agent(
        config,
        env,
        pdir,
        script_path=args.script,
        max_turns=args.max_turns,
    )


def _cmd_list(args):
    """List all registered projects with their status."""
    projects = list_projects()
    if not projects:
        print(
            "  No projects registered. Run 'dsagt init <name> --agent <platform>' to create one."
        )
        return

    for name, path in projects.items():
        pdir = Path(path)
        cfg_file = pdir / ".dsagt" / "config.yaml"

        # Best-effort: if the config is readable, show the agent.  If the
        # project dir is gone or the config is broken, just show the path.
        agent = ""
        if cfg_file.exists():
            try:
                config = load_config(name)
                agent = config.get("agent", "")
            except (FileNotFoundError, ValueError):
                pass

        print(f"  {name:<20} {agent:<14} {path}")


def _cmd_mv(args):
    """Move a project to a new location."""
    location = Path(args.location).resolve()
    new_path = move_project(args.project, location)
    print(f"  Moved {args.project} → {new_path}")


def _cmd_rm(args):
    """Unregister a project and (by default) delete its directory.

    With ``--all``: bulk-remove every registered project.
    """
    if args.all and args.project:
        raise SystemExit("dsagt rm: pass either a project name or --all, not both.")
    if not args.all and not args.project:
        raise SystemExit("dsagt rm: project name required (or pass --all).")

    if args.all:
        return _cmd_rm_all(args)

    pdir = project_dir(args.project)

    if args.keep_files:
        action = f"Unregister '{args.project}' (keep files at {pdir})"
    else:
        action = f"Delete project '{args.project}' at {pdir} and unregister it"

    if not args.yes:
        resp = input(f"{action}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("  Cancelled.")
            return 0

    remove_project(args.project, keep_files=args.keep_files)
    if args.keep_files:
        print(f"  Unregistered {args.project} (files kept at {pdir})")
    else:
        print(f"  Removed {args.project}")


def _cmd_rm_all(args) -> int:
    """``dsagt rm --all`` — bulk-remove every registered project."""
    projects = list_projects()
    if not projects:
        print("  No projects registered.")
        return 0

    verb = "Unregister" if args.keep_files else "Delete"
    print(f"  {verb} {len(projects)} project{'s' if len(projects) != 1 else ''}:")
    for name, path in sorted(projects.items()):
        print(f"    {name:<32}  {path}")
    print()

    if not args.yes:
        resp = (
            input(f"  Confirm {verb.lower()} all {len(projects)} projects? [y/N] ")
            .strip()
            .lower()
        )
        if resp not in ("y", "yes"):
            print("  Cancelled.")
            return 0

    failures: list[tuple[str, str]] = []
    for name in sorted(projects):
        try:
            remove_project(name, keep_files=args.keep_files)
            verb_past = "Unregistered" if args.keep_files else "Removed"
            print(f"  {verb_past} {name}")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"  FAILED {name}: {e}", file=sys.stderr)

    if failures:
        print(file=sys.stderr)
        print(f"  {len(failures)} of {len(projects)} removals failed.", file=sys.stderr)
        return 1
    return 0


def _cmd_info(args):
    """Triage summary of a project's MLflow traces."""
    from dsagt.commands.info import run

    return run(args.project, as_json=args.json)


def _cmd_smoke_test(args):
    """Run the end-to-end smoke test (non-interactive, with assertions).

    Thin wrapper around ``tests/smoke_test/run.sh`` so the script stays the
    source of truth — bash is the right shape for orchestrating processes
    and assertion checks.  CLI exposure is just for ergonomics.

    With ``--all``, run the harness in parallel for every agent in
    ``VALID_AGENTS``.  Each agent has its own project name (``smoke-test-X``)
    so they don't collide on MLflow ports, kb_index, or registry entries.
    Output is per-agent log files; the summary prints in finish order.
    """
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    script = pkg_dir / "tests" / "smoke_test" / "run.sh"
    if not script.exists():
        print(f"Error: smoke test script not found at {script}", file=sys.stderr)
        return 1

    if args.all:
        return _run_smoke_all(script)

    agent = args.agent or "goose"
    return subprocess.run(["bash", str(script), agent]).returncode


def _run_smoke_all(script: Path) -> int:
    """Run the smoke harness for every VALID_AGENTS entry in parallel.

    Streams each agent's stdout/stderr to a per-agent log file so the
    terminal stays readable.  Prints the verdict for each agent as it
    finishes — fastest-first, not input order — so the operator can see
    progress instead of waiting for the slowest agent before any output.
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    agents = list(VALID_AGENTS)
    log_dir = Path(tempfile.mkdtemp(prefix="dsagt-smoke-"))

    print(f"[smoke-all] launching {len(agents)} parallel runs", flush=True)
    print(f"[smoke-all] log dir: {log_dir}", flush=True)
    for a in agents:
        print(f"[smoke-all]   {a:7}  → {log_dir / f'smoke-{a}.log'}", flush=True)

    def _run_one(agent: str) -> tuple[str, int, float]:
        start = time.monotonic()
        with (log_dir / f"smoke-{agent}.log").open("w") as fh:
            rc = subprocess.run(
                ["bash", str(script), agent],
                stdout=fh,
                stderr=subprocess.STDOUT,
            ).returncode
        return agent, rc, time.monotonic() - start

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        futs = {ex.submit(_run_one, a): a for a in agents}
        for fut in as_completed(futs):
            agent, rc, elapsed = fut.result()
            verdict = "PASS" if rc == 0 else "FAIL"
            print(
                f"[smoke-all] {verdict}  {agent:7}  {elapsed:5.0f}s  "
                f"(rc={rc}, log={log_dir / f'smoke-{agent}.log'})",
                flush=True,
            )
            results[agent] = rc

    n_pass = sum(1 for rc in results.values() if rc == 0)
    print(
        f"[smoke-all] {n_pass}/{len(agents)} passed",
        flush=True,
    )
    return 0 if n_pass == len(agents) else 1


# User-facing exception types that should produce a clean one-line error
# message rather than a traceback.  Everything else crashes loudly.
_USER_ERRORS = (FileNotFoundError, FileExistsError, ValueError, RuntimeError)


def main(argv=None):
    from dsagt import __version__

    parser = argparse.ArgumentParser(
        prog="dsagt", description="DSAgt project and session management."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"dsagt {__version__}")

    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser(
        "init",
        help="Create or reconfigure a project (interactive; re-runnable)",
    )
    p_init.add_argument(
        "project",
        nargs="?",
        help="Project name (also the folder name).  Prompted if omitted in a TTY.",
    )
    p_init.add_argument(
        "--agent",
        choices=VALID_AGENTS,
        default=None,
        help="Agent platform.  Prompted interactively; required non-interactively.",
    )
    p_init.add_argument(
        "--location",
        default=None,
        help="Parent directory for the project (default: ~/dsagt-projects/).  "
        "Non-interactive path; interactive prompts for the full project dir.",
    )
    _kb_sel = p_init.add_mutually_exclusive_group()
    _kb_sel.add_argument(
        "--include",
        nargs="+",
        metavar="ASSET",
        help="KB assets to provision into the project (or 'all' for "
        "everything).  Default: bundled tools + the genesis skill catalog.",
    )
    _kb_sel.add_argument(
        "--exclude",
        nargs="+",
        metavar="ASSET",
        help="Provision the default KB set minus these assets ('all' to "
        "create the project with no bundled KB content).",
    )
    p_init.add_argument(
        "--episodic",
        action="store_true",
        help="Enable episodic memory (captures session turns into searchable "
        "memory).  Off by default.",
    )

    p_start = sub.add_parser("start", help="Start a project session")
    p_start.add_argument("project", help="Project name")
    p_start.add_argument(
        "--agent",
        choices=VALID_AGENTS,
        default=None,
        help="Agent platform.  Required on first start if init didn't set one; "
        "thereafter, a per-run override (doesn't update the YAML default).",
    )
    p_start.add_argument(
        "--script",
        default=None,
        help="Path to a goose-run instructions file. When set, the agent runs "
        "non-interactively (GOOSE_MODE=auto) against this script — used by "
        "the smoke test to share the full dsagt start lifecycle (config "
        "generation, memory extraction) with manual runs.",
    )
    p_start.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Cap on agent turn count when --script is set (default: 30).",
    )

    p_info = sub.add_parser(
        "info",
        help="Summarize a project's MLflow traces (tokens, errors, by session/source)",
    )
    p_info.add_argument("project", help="Project name")
    p_info.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured report as JSON instead of formatted text",
    )

    p_smoke = sub.add_parser(
        "smoke-test",
        help="Run the end-to-end smoke test (sources DSAGT/.env, drives the agent non-interactively, asserts artifacts)",
    )
    smoke_group = p_smoke.add_mutually_exclusive_group()
    smoke_group.add_argument(
        "--agent",
        choices=list(VALID_AGENTS),
        default=None,
        help="Which agent to drive (default: goose).",
    )
    smoke_group.add_argument(
        "--all",
        action="store_true",
        help="Run the smoke harness in parallel for every agent.  Per-agent "
        "logs go to a temp dir; verdicts print in finish order.",
    )

    sub.add_parser("list", help="List all registered projects and their status")

    p_mv = sub.add_parser("mv", help="Move a project to a new location")
    p_mv.add_argument("project", help="Project name")
    p_mv.add_argument("location", help="New parent directory")

    p_rm = sub.add_parser("rm", help="Unregister a project and delete its directory")
    p_rm.add_argument("project", nargs="?", help="Project name (omit when using --all)")
    p_rm.add_argument(
        "--all",
        action="store_true",
        help="Remove every registered project.  Auto-stops any running "
        "services before removing.  Still requires confirmation "
        "unless -y is also set.",
    )
    p_rm.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    p_rm.add_argument(
        "--keep-files",
        action="store_true",
        help="Unregister only; leave the project directory on disk",
    )

    args = parser.parse_args(argv)

    # The CLI speaks to the user via ``print()``; library logs are diagnostic.
    # Default the console to WARNING so INFO chatter (embedder load, route
    # registration, catalog indexing) doesn't bury the init/start output.
    # ``--verbose`` opts into the full DEBUG stream.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [dsagt] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        return 1

    cmds = {
        "init": _cmd_init,
        "start": _cmd_start,
        "info": _cmd_info,
        "smoke-test": _cmd_smoke_test,
        "list": _cmd_list,
        "mv": _cmd_mv,
        "rm": _cmd_rm,
    }
    try:
        return cmds[args.command](args)
    except _USER_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

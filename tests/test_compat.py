"""The platform layer: what dsagt asks of the OS, on whichever OS runs this.

Each test states behavior both platforms must give.  The few that describe
one platform's rule (a ``.cmd`` shim, a backslash path) set ``IS_WINDOWS`` on
the module, so the rule is exercised on every CI runner and the real system
call is exercised on its own.
"""

import errno
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dsagt import _compat
from dsagt._compat import (
    detached_process_kwargs,
    file_lock,
    is_dir_link,
    link_dir,
    remove_dir_link,
    remove_tree,
    replace_file,
    resolve_executable,
    split_command,
)

# -- file_lock ---------------------------------------------------------------


def test_file_lock_excludes_a_second_holder(tmp_path):
    """Two threads take the lock in turn, never together."""
    lock = tmp_path / "x.lock"
    inside = 0
    overlaps = []

    def work():
        nonlocal inside
        for _ in range(20):
            with file_lock(lock):
                inside += 1
                overlaps.append(inside)
                time.sleep(0.001)
                inside -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(overlaps) == {1}


def test_file_lock_blocks_another_process(tmp_path):
    """A second process waits for the lock, which is the case the lock is
    for: two ``dsagt init`` runs, or a burst of ``dsagt-run`` trace writers."""
    lock = tmp_path / "x.lock"
    out = tmp_path / "order.txt"
    child = (
        "import sys, time\n"
        "from dsagt._compat import file_lock\n"
        "with file_lock(sys.argv[1]):\n"
        "    open(sys.argv[2], 'a', encoding='utf-8').write('child\\n')\n"
    )
    with file_lock(lock):
        proc = subprocess.Popen([sys.executable, "-c", child, str(lock), str(out)])
        time.sleep(1.0)
        assert proc.poll() is None, "the child took a lock this process holds"
        out.open("a", encoding="utf-8").write("parent\n")
    assert proc.wait(timeout=30) == 0
    assert out.read_text(encoding="utf-8").split() == ["parent", "child"]


def test_file_lock_is_released_when_the_block_raises(tmp_path):
    lock = tmp_path / "x.lock"
    with pytest.raises(RuntimeError):
        with file_lock(lock):
            raise RuntimeError("boom")
    with file_lock(lock):
        pass


def test_file_lock_keeps_the_lock_file_contents(tmp_path):
    """Opened for append: Windows refuses to truncate a file that another
    process holds a lock in, so the lock never truncates."""
    lock = tmp_path / "x.lock"
    lock.write_text("kept", encoding="utf-8")
    with file_lock(lock):
        pass
    assert lock.read_text(encoding="utf-8") == "kept"


# -- replace_file / remove_tree ----------------------------------------------


def test_file_lock_gives_up_on_a_failure_that_will_not_clear(tmp_path, monkeypatch):
    """A contended region is waited on again; any other failure is raised.

    ``msvcrt.locking`` waits ten seconds and then reports a holder that is
    still working as ``EDEADLOCK``, which is worth asking about again.  A
    closed descriptor or a filesystem that refuses the region reports
    something else, and no amount of waiting fixes it; waiting on one anyway
    would hang whoever took the lock.
    """

    class FakeMsvcrt:
        LK_LOCK = LK_UNLCK = 0

        def __init__(self):
            self.calls = 0

        def locking(self, fd, mode, nbytes):
            self.calls += 1
            if self.calls <= 2:
                raise OSError(_compat._EDEADLOCK, "already locked")
            raise OSError(errno.EACCES, "permission denied")

    fake = FakeMsvcrt()
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_compat, "msvcrt", fake, raising=False)

    with pytest.raises(OSError) as caught:
        with file_lock(tmp_path / "x.lock"):
            pass

    assert caught.value.errno == errno.EACCES
    # Two contended waits, then the failure that was raised rather than spun on.
    assert fake.calls == 3


def test_replace_file_overwrites(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")
    replace_file(src, dst)
    assert dst.read_text(encoding="utf-8") == "new"
    assert not src.exists()


def test_replace_file_retries_while_the_target_is_busy(tmp_path, monkeypatch):
    """Windows refuses the replace while a reader has the target open; the
    reader closes within milliseconds and the retry goes through."""
    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("new", encoding="utf-8")
    real, calls = os.replace, []

    def busy_twice(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "in use")
        real(a, b)

    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_compat.os, "replace", busy_twice)
    replace_file(src, dst)
    assert len(calls) == 3
    assert dst.read_text(encoding="utf-8") == "new"


def test_remove_tree_removes_read_only_files(tmp_path):
    """A git clone's object files are read-only, which stops a plain rmtree
    on Windows."""
    tree = tmp_path / "clone" / ".git" / "objects"
    tree.mkdir(parents=True)
    obj = tree / "pack"
    obj.write_text("x", encoding="utf-8")
    obj.chmod(stat.S_IREAD)
    remove_tree(tmp_path / "clone")
    assert not (tmp_path / "clone").exists()


def test_the_rmtree_handler_waits_out_a_file_held_open(tmp_path, monkeypatch):
    """A sharing violation is not a read-only attribute: clearing the
    attribute and retrying at once still fails, and the tree would be left
    half deleted."""
    target = tmp_path / "held"
    target.write_text("x", encoding="utf-8")
    calls = []

    def busy_twice(path):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError(13, "in use")
        os.unlink(path)

    _compat._clear_readonly_and_retry(busy_twice, target, None)
    assert len(calls) == 3
    assert not target.exists()


def test_the_rmtree_handler_gives_up_on_a_refusal_that_will_not_clear(
    tmp_path, monkeypatch
):
    target = tmp_path / "locked"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(_compat, "_SHARING_RETRY_SECONDS", 0.0)

    def always_busy(_path):
        raise PermissionError(13, "in use")

    with pytest.raises(PermissionError):
        _compat._clear_readonly_and_retry(always_busy, target, None)


def test_remove_tree_ignore_errors_on_a_missing_path(tmp_path):
    remove_tree(tmp_path / "absent", ignore_errors=True)
    with pytest.raises(OSError):
        remove_tree(tmp_path / "absent")


# -- directory links ---------------------------------------------------------


def test_link_dir_is_live_and_removal_keeps_the_target(tmp_path):
    """Whatever kind of link the platform grants (a symlink, or a junction
    on Windows without the symlink privilege), it shows the source's current
    files and removing it leaves the source alone."""
    src = tmp_path / "skills" / "greet"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("one", encoding="utf-8")
    dest = tmp_path / ".claude" / "skills" / "greet"
    dest.parent.mkdir(parents=True)

    if not link_dir(src, dest):
        pytest.skip("this volume grants neither a symlink nor a junction")
    assert is_dir_link(dest)
    (src / "SKILL.md").write_text("two", encoding="utf-8")
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "two"

    remove_dir_link(dest)
    assert not dest.exists()
    assert (src / "SKILL.md").exists()


def test_remove_tree_on_a_link_keeps_the_target(tmp_path):
    src = tmp_path / "shared"
    src.mkdir()
    (src / "data").write_text("x", encoding="utf-8")
    dest = tmp_path / "link"
    if not link_dir(src, dest):
        pytest.skip("this volume grants neither a symlink nor a junction")
    remove_tree(dest)
    assert not dest.exists()
    assert (src / "data").exists()


def test_mirror_copies_when_no_link_can_be_made(tmp_path, monkeypatch):
    """With neither kind of link available the skill is copied, so the agent
    still discovers it."""
    from dsagt.agents import base

    src = tmp_path / "skills" / "greet"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Say hello.\n---\nbody\n", encoding="utf-8"
    )
    target = tmp_path / ".claude" / "skills"
    monkeypatch.setattr(base, "link_dir", lambda *_: False)

    base._mirror_skills_to(target, [src])
    assert (target / "greet" / "SKILL.md").is_file()
    assert not is_dir_link(target / "greet")
    # A second run replaces the copy instead of failing on it.
    base._mirror_skills_to(target, [src])
    assert (target / "greet" / "SKILL.md").is_file()


# -- processes ---------------------------------------------------------------


def test_resolve_executable_is_a_no_op_on_posix(monkeypatch):
    monkeypatch.setattr(_compat, "IS_WINDOWS", False)
    assert resolve_executable("claude") == "claude"


def test_resolve_executable_finds_a_cmd_shim(monkeypatch):
    """An npm-installed CLI is ``claude.cmd``; CreateProcess appends only
    ``.exe``, so the bare name is replaced by what PATH search finds."""
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    shim = r"C:\Users\x\AppData\Roaming\npm\claude.cmd"
    monkeypatch.setattr(
        _compat.shutil,
        "which",
        lambda name, path=None: shim if name == "claude" else None,
    )
    assert resolve_executable("claude") == shim
    assert resolve_executable("absent") == "absent"


def test_resolve_executable_maps_python3_to_python(monkeypatch):
    """A Windows venv has no ``python3``, and the one on PATH is the stub
    that opens the Store."""
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    found = {
        "python3": r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python3.exe",
        "python": r"C:\venv\Scripts\python.exe",
    }
    monkeypatch.setattr(
        _compat.shutil, "which", lambda name, path=None: found.get(name)
    )
    assert resolve_executable("python3") == found["python"]
    found.pop("python3")
    assert resolve_executable("python3") == found["python"]


def test_resolve_executable_leaves_a_path_alone(monkeypatch):
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_compat.shutil, "which", lambda name, path=None: "wrong")
    assert resolve_executable("scripts/run.exe") == "scripts/run.exe"


def test_split_command_keeps_windows_backslashes(monkeypatch):
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    assert split_command(r'python C:\data\x.py "two words"') == [
        "python",
        r"C:\data\x.py",
        "two words",
    ]
    # A quoted path keeps one backslash per separator, whichever quote it
    # was written with; doubling them all gave the single-quoted form two,
    # and a declared file role stopped matching the argument.
    for quoted in (
        r"python 'C:\data folder\convert.py'",
        'python "C:\\data folder\\convert.py"',
    ):
        assert split_command(quoted) == ["python", r"C:\data folder\convert.py"]
    # A backslash before a quote is that quote's escape, not a separator;
    # doubling it ended the word early and raised ValueError.
    assert split_command('python -c "print(\\"don\'t\\")"') == [
        "python",
        "-c",
        'print("don\'t")',
    ]
    assert split_command(r"python C:\a\x.py \\server\share\f.csv") == [
        "python",
        r"C:\a\x.py",
        r"\\server\share\f.csv",
    ]
    monkeypatch.setattr(_compat, "IS_WINDOWS", False)
    assert split_command("python skills/x/scripts/y.py 'a b'") == [
        "python",
        "skills/x/scripts/y.py",
        "a b",
    ]


def test_popen_command_quotes_for_a_batch_shim(monkeypatch):
    """cmd.exe parses the line a second time, so an & in a project path
    would start a second command."""
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    argv = [r"C:\npm\codex.cmd", "-C", r"C:\work\R&D", "exec"]
    assert _compat.popen_command(argv) == (
        r'"C:\npm\codex.cmd" "-C" "C:\work\R&D" "exec"'
    )
    # A quote inside an argument is doubled, the one escape cmd honours
    # inside a quoted word.
    assert _compat.popen_command([r"a.bat", 'say "hi"']) == r'"a.bat" "say ""hi"""'
    # Anything Windows runs itself is left as a list.
    assert _compat.popen_command([r"C:\py\python.exe", "-c", "1&2"]) == [
        r"C:\py\python.exe",
        "-c",
        "1&2",
    ]


def test_popen_command_is_a_no_op_on_posix(monkeypatch):
    monkeypatch.setattr(_compat, "IS_WINDOWS", False)
    argv = ["./run.cmd", "a&b"]
    assert _compat.popen_command(argv) == argv


def test_detached_child_outlives_and_runs(tmp_path):
    """The kwargs start a real process on this platform."""
    out = tmp_path / "ran.txt"
    code = "import sys; open(sys.argv[1], 'w', encoding='utf-8').write('ok')"
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(out)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **detached_process_kwargs(),
    )
    assert proc.wait(timeout=30) == 0
    assert out.read_text(encoding="utf-8") == "ok"


def test_importing_the_entry_points_needs_no_posix_module():
    """``fcntl`` does not exist on Windows; no dsagt module may import it,
    or the commands fail before they start."""
    src = Path(_compat.__file__).parent
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p.name != "_compat.py"
        and any(
            line.strip()
            in ("import fcntl", "import pwd", "import grp", "import termios")
            for line in p.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert offenders == []


# -- paths in provenance -----------------------------------------------------


def test_project_relative_paths_fold_windows_spellings(monkeypatch):
    """The agent types the project path with either slash and any case; the
    record still maps to one project-relative spelling."""
    from dsagt import provenance

    monkeypatch.setattr(provenance, "IS_WINDOWS", True)
    root = r"C:\Users\x\dsagt-projects\demo"
    rel = provenance._relative_to_project
    assert rel(r"C:\Users\x\dsagt-projects\demo\data\a.csv", root) == "data/a.csv"
    assert rel("c:/users/x/dsagt-projects/demo/data/a.csv", root) == "data/a.csv"
    assert rel(root.lower(), root) == "."
    assert rel(r"C:\Users\x\dsagt-projects\demo-other\a.csv", root).startswith("C:")
    # File paths come out with forward slashes, the spelling the bash script
    # and the Snakemake workflow are written in.
    assert (
        provenance._project_file(r"C:\Users\x\dsagt-projects\demo\data\a.csv", root)
        == "data/a.csv"
    )


def test_sqlite_store_uri_uses_forward_slashes():
    from pathlib import PureWindowsPath

    from dsagt.observability import sqlite_store_uri

    assert (
        sqlite_store_uri(PureWindowsPath(r"C:\Users\x\proj\mlflow.db"))
        == "sqlite:///C:/Users/x/proj/mlflow.db"
    )


# -- added after the first run on Windows -------------------------------------


def test_resolve_executable_prefers_this_interpreter_over_the_store_stub(monkeypatch):
    """dsagt started by its full path, environment not activated: the only
    ``python`` on PATH is the Store stub, which exits 9009."""
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    stub = r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe"
    monkeypatch.setattr(_compat.shutil, "which", lambda name, path=None: stub)
    assert resolve_executable("python") == sys.executable
    assert resolve_executable("python3") == sys.executable


def test_resolve_executable_never_picks_wsl_bash(monkeypatch):
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(
        _compat, "find_bash", lambda: r"C:\Program Files\Git\bin\bash.exe"
    )
    monkeypatch.setattr(
        _compat.shutil, "which", lambda name, path=None: r"C:\Windows\System32\bash.exe"
    )
    assert resolve_executable("bash") == r"C:\Program Files\Git\bin\bash.exe"


def test_resolve_command_puts_the_interpreter_before_a_script(tmp_path, monkeypatch):
    """Windows starts executables only; Git Bash runs ``./tool.py x`` by its
    ``#!`` line, so dsagt-run gives the same line the same meaning."""
    script = tmp_path / "tool.py"
    script.write_text("print(1)\n", encoding="utf-8")
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(
        _compat.shutil, "which", lambda name, path=None: rf"C:\venv\Scripts\{name}.exe"
    )
    assert _compat.resolve_command([str(script), "x"]) == [
        r"C:\venv\Scripts\python.exe",
        str(script),
        "x",
    ]
    assert (
        _compat.resolve_command(["fastp", "-i", "a"])[0] == r"C:\venv\Scripts\fastp.exe"
    )
    monkeypatch.setattr(_compat, "IS_WINDOWS", False)
    assert _compat.resolve_command([str(script), "x"]) == [str(script), "x"]


def test_native_path_reads_the_git_bash_spelling(monkeypatch):
    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    assert _compat.native_path("/c/Users/x/data") == "C:/Users/x/data"
    assert _compat.native_path("/d") == "D:/"
    assert _compat.native_path("/tmp/x") == "/tmp/x"
    assert _compat.native_path(r"C:\Users\x") == r"C:\Users\x"
    assert _compat.native_path("data/x.csv") == "data/x.csv"
    monkeypatch.setattr(_compat, "IS_WINDOWS", False)
    assert _compat.native_path("/c/Users/x/data") == "/c/Users/x/data"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_read_text_any_encoding(tmp_path, encoding):
    """PowerShell's ``>`` writes UTF-16 with a byte-order mark; the
    credentials file a person makes that way still loads."""
    f = tmp_path / "env"
    f.write_text("KEY=värde\n", encoding=encoding, newline="\n")
    assert _compat.read_text_any_encoding(f) == "KEY=värde\n"


def test_user_env_file_written_by_powershell_loads(tmp_path, monkeypatch):
    from dsagt.session import load_user_env

    f = tmp_path / "env"
    f.write_text("DSAGT_TEST_FROM_PS=1\n", encoding="utf-16")
    monkeypatch.delenv("DSAGT_TEST_FROM_PS", raising=False)
    try:
        assert load_user_env(f) == ["DSAGT_TEST_FROM_PS"]
    finally:
        os.environ.pop("DSAGT_TEST_FROM_PS", None)


def test_rename_path_retries_while_a_file_inside_is_busy(tmp_path, monkeypatch):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    real, calls = os.rename, []

    def busy_once(a, b):
        calls.append(1)
        if len(calls) < 2:
            raise PermissionError(5, "Access is denied")
        real(a, b)

    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_compat.os, "rename", busy_once)
    _compat.rename_path(src, dst)
    assert dst.is_dir() and len(calls) == 2


def test_rewriting_a_usage_line_keeps_backslashes(tmp_path):
    """The stored command is the replacement text of a regex substitution; a
    backslash in it must not be read as a group reference or an escape."""
    from dsagt.skills import rewrite_cli_invocations

    skill = tmp_path / "s"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: s\ndescription: d\n---\nRun `python scripts/x.py in.csv` now.\n",
        encoding="utf-8",
    )
    wrapped = r"dsagt-run --code s-x -- python C:\data\scripts\x.py"
    rewrite_cli_invocations(skill, [("python scripts/x.py", wrapped)])
    assert wrapped in (skill / "SKILL.md").read_text(encoding="utf-8")


def test_stdin_from_the_null_device_is_not_interactive(tmp_path):
    """A scripted run is asked nothing, so ``dsagt init`` works from flags.

    POSIX settles this with ``isatty()``, which is false for ``/dev/null``.
    Windows counts ``NUL`` a character device just as it counts a console
    one and answers true, so init tried to prompt and died on the EOF;
    only a handle that has a console mode is a console.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from dsagt._compat import stdin_is_interactive\n"
        "print(stdin_is_interactive())\n",
        encoding="utf-8",
    )
    with open(os.devnull) as devnull:
        done = subprocess.run(
            [sys.executable, str(probe)],
            stdin=devnull,
            capture_output=True,
            text=True,
        )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_a_tty_that_is_not_a_console_is_not_interactive(monkeypatch):
    """The Windows rule the test above can only reach on a Windows runner.

    ``NUL`` answers ``isatty()`` with true, so the console mode is what
    decides; a handle that has one is a person's console, and a handle that
    has none is a redirect.
    """

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return 0

    monkeypatch.setattr(_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_compat.sys, "stdin", FakeStdin())

    monkeypatch.setattr(_compat, "_has_console_mode", lambda fd: False)
    assert _compat.stdin_is_interactive() is False

    monkeypatch.setattr(_compat, "_has_console_mode", lambda fd: True)
    assert _compat.stdin_is_interactive() is True

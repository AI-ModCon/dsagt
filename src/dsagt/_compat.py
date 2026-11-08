"""What differs between POSIX and Windows, in one place.

Everything else in dsagt calls these and stays free of ``os.name`` checks:
an exclusive file lock, a file replace and a tree removal that tolerate
Windows' sharing rules, a directory link that needs no privilege, the
lookup of an executable the way a shell would find it, and the flags that
detach a child process.

Standard library only, and nothing heavy at module scope: ``dsagt-run``
imports this on every recorded command.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    import msvcrt
else:
    import fcntl


# ---------------------------------------------------------------------------
# File lock
# ---------------------------------------------------------------------------


#: ``msvcrt.locking``'s "someone else holds this region" errno.  Windows and
#: Linux spell it ``EDEADLOCK``, macOS only ``EDEADLK``; the comparison below
#: runs on Windows, where both names are the same number.
_EDEADLOCK = getattr(errno, "EDEADLOCK", errno.EDEADLK)

#: How long :func:`file_lock` waits on Windows for another process to let go.
#: Generous next to the sharing retry below: a holder is doing a unit of real
#: work (registering a project, indexing a pass of records), not closing a
#: file it has just written.  The bound is there so a lock that will never be
#: granted fails instead of hanging dsagt.
_LOCK_WAIT_SECONDS = 120.0


@contextmanager
def file_lock(path: str | os.PathLike):
    """Hold an exclusive lock on the file at *path* for the ``with`` block.

    Blocks until the lock is free.  The lock is advisory and is released when
    the block ends or the process dies.  ``flock`` on POSIX; on Windows a
    one-byte ``msvcrt.locking`` region, which waits ten seconds and then
    reports the region as contended rather than blocking, so it is asked
    again until the holder lets go or :data:`_LOCK_WAIT_SECONDS` is up.

    The file is opened for append, never truncated: Windows refuses to
    truncate a file another process holds a lock in.
    """
    with open(path, "a+b") as handle:
        if IS_WINDOWS:
            handle.seek(0)
            deadline = time.monotonic() + _LOCK_WAIT_SECONDS
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as e:
                    # LK_LOCK tries once a second for ten seconds, then
                    # raises EDEADLOCK; a holder that is still working is
                    # not an error.  Any other errno (a closed descriptor, a
                    # region the filesystem refuses to lock) would never
                    # clear, and waiting on it would hang the caller.
                    if e.errno != _EDEADLOCK or time.monotonic() >= deadline:
                        raise
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Replace and remove under Windows' sharing rules
# ---------------------------------------------------------------------------

#: How long a replace or a removal keeps retrying on Windows while another
#: process (a reader, an indexer, antivirus) holds the file open.
_SHARING_RETRY_SECONDS = 5.0


def replace_file(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """``os.replace``, retried on Windows while *dst* is open elsewhere.

    POSIX replaces a file other processes have open; Windows refuses with
    ``PermissionError`` until they close it, which a reader does within
    milliseconds.
    """
    if not IS_WINDOWS:
        os.replace(src, dst)
        return
    deadline = time.monotonic() + _SHARING_RETRY_SECONDS
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def rename_path(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """``os.rename`` for a file or directory, retried on Windows.

    Antivirus and the search indexer open freshly written files for a
    moment, and Windows refuses to rename a directory while any file in it
    is open; the handle is gone within a second or two.
    """
    if not IS_WINDOWS:
        os.rename(src, dst)
        return
    deadline = time.monotonic() + _SHARING_RETRY_SECONDS
    while True:
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def read_text_any_encoding(path: str | os.PathLike) -> str:
    """The text of a file a person wrote by hand, whatever their editor or
    shell saved it as.

    UTF-8 everywhere else in dsagt; but Windows PowerShell's ``>`` writes
    UTF-16 with a byte-order mark and older Notepad writes UTF-8 with one,
    and a credentials file made either way must still load.  A mark decides;
    without one the file is UTF-8.
    """
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def native_path(path: str) -> str:
    """*path* as this operating system spells it.

    Git Bash shows a Windows drive as ``/c/Users/x`` and converts that
    spelling only for command-line arguments, so an agent working in it
    passes the bash spelling to an MCP tool; on Windows that is turned back
    into ``C:/Users/x``.  Unchanged on POSIX, where ``/c/...`` is a real
    path.
    """
    if (
        IS_WINDOWS
        and len(path) >= 2
        and path[0] == "/"
        and path[1].isalpha()
        and (len(path) == 2 or path[2] == "/")
    ):
        return f"{path[1].upper()}:{path[2:] or '/'}"
    return path


def _clear_readonly_and_retry(func, path, _exc):
    """``shutil.rmtree`` error handler: make *path* writable and try again.

    Git marks its object files read-only, and Windows refuses to delete a
    read-only file (POSIX only needs the directory to be writable).
    """
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)


def remove_tree(path: str | os.PathLike, *, ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` that also removes read-only files on Windows.

    A directory link (symlink or junction) is removed as a link; its target
    is left alone.
    """
    path = Path(path)
    if is_dir_link(path):
        try:
            remove_dir_link(path)
        except OSError:
            if not ignore_errors:
                raise
        return
    if not IS_WINDOWS:
        shutil.rmtree(path, ignore_errors=ignore_errors)
        return
    try:
        shutil.rmtree(path, onexc=_clear_readonly_and_retry)
    except OSError:
        if not ignore_errors:
            raise


# ---------------------------------------------------------------------------
# Directory links
# ---------------------------------------------------------------------------


def is_dir_link(path: str | os.PathLike) -> bool:
    """Whether *path* is a symlink or, on Windows, a directory junction."""
    return os.path.islink(path) or os.path.isjunction(path)


def remove_dir_link(path: str | os.PathLike) -> None:
    """Remove a link made by :func:`link_dir`, leaving its target in place."""
    if os.path.isjunction(path):
        os.rmdir(path)
    else:
        os.unlink(path)


def link_dir(src: Path, dest: Path) -> bool:
    """Make *dest* a live link to the directory *src*; ``False`` if none can
    be made.

    A relative symlink where the platform allows one.  Windows grants
    symlinks only to administrators and to accounts with Developer Mode on,
    so there the fallback is a directory junction, which any user may create
    and which reads the same to every program.  A junction needs both ends
    on local volumes; when that fails too the caller copies.
    """
    try:
        dest.symlink_to(
            os.path.relpath(src.resolve(), dest.parent.resolve()),
            target_is_directory=True,
        )
        return True
    except (OSError, ValueError, NotImplementedError):
        # ValueError: relpath across two Windows drives.
        if not IS_WINDOWS:
            raise
    try:
        import _winapi

        _winapi.CreateJunction(str(src.resolve()), str(dest))
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


def resolve_executable(name: str, path: str | None = None) -> str:
    """The program a shell would run for *name*, as a path ``Popen`` accepts.

    POSIX ``Popen`` searches PATH itself, so *name* is returned as is.
    Windows' ``CreateProcess`` differs from a shell in three ways this
    undoes: it only appends ``.exe`` (an npm-installed CLI is a ``.cmd``
    shim), it searches the running interpreter's own directory before PATH
    (inside a venv that is the *base* Python, so ``python`` would lose the
    venv's packages), and it knows no ``python3`` (a venv has none, and the
    one Windows puts on PATH is a stub that opens the Store).

    *name* is returned unchanged when nothing on *path* matches, so the
    caller's ``FileNotFoundError`` names what the user typed.
    """
    if not IS_WINDOWS:
        return name
    if os.path.dirname(name):
        return name
    if name.lower() in ("bash", "sh", "bash.exe", "sh.exe"):
        # PATH's first bash is usually WSL's launcher; see find_bash.
        return find_bash() or name
    found = shutil.which(name, path=path)
    if name.lower() in _PYTHON_NAMES and (found is None or _is_store_stub(found)):
        # No Python on PATH but the Store's stub (dsagt was started by its
        # full path, without activating its environment), or a ``python3``
        # that Windows does not have: the interpreter running dsagt is the
        # one whose environment the user installed into.
        other = shutil.which("python", path=path)
        found = other if other and not _is_store_stub(other) else sys.executable
    return found or name


_PYTHON_NAMES = ("python", "python.exe", "python3", "python3.exe")


def _is_store_stub(program: str) -> bool:
    """Whether *program* is one of the zero-byte "app execution aliases"
    Windows puts on PATH, which print how to install Python from the Store
    and exit 9009."""
    return "\\windowsapps\\" in program.lower()


def find_bash() -> str | None:
    """A bash that runs in this operating system, or ``None``.

    ``bash`` from PATH on POSIX.  On Windows the first ``bash.exe`` on PATH
    is usually ``System32\\bash.exe``, the WSL launcher, which would run a
    script inside the Linux guest against a different filesystem and a
    different dsagt; Git for Windows' bash is looked up beside ``git``
    instead, then any other bash on PATH outside the Windows directory.
    """
    if not IS_WINDOWS:
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        # <Git>\cmd\git.exe or <Git>\bin\git.exe -> <Git>\bin\bash.exe
        candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    windir = os.environ.get("SYSTEMROOT", r"C:\Windows").lower()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / "bash.exe"
        if candidate.is_file() and not str(candidate).lower().startswith(windir):
            return str(candidate)
    return None


def split_command(command: str) -> list[str]:
    """Split a command line into arguments, as the stored ``executable`` of a
    code is read.

    POSIX shell rules.  On Windows a backslash is kept as a literal
    character instead of an escape, so ``python C:\\data\\x.py`` keeps its
    path; quotes group words on both.
    """
    import shlex

    if IS_WINDOWS:
        command = command.replace("\\", "\\\\")
    return shlex.split(command)


#: The interpreter that runs a script named as a command's first word, by
#: suffix.  A POSIX kernel reads the script's ``#!`` line; Windows runs only
#: executables, and answers anything else with "not a valid Win32
#: application".
_SCRIPT_INTERPRETERS = {".py": "python", ".sh": "bash", ".bash": "bash"}


def resolve_command(command: list[str], path: str | None = None) -> list[str]:
    """*command* as an argv ``Popen`` can start on this platform.

    On Windows the program is looked up as :func:`resolve_executable`
    describes, and a command that starts with a script file
    (``./run.sh x``, ``skills/s/scripts/tool.py x``) gets the script's
    interpreter in front, which is what Git Bash does with the same line.
    Unchanged on POSIX.
    """
    if not IS_WINDOWS or not command:
        return list(command)
    first = command[0]
    interpreter = _SCRIPT_INTERPRETERS.get(os.path.splitext(first)[1].lower())
    if interpreter and os.path.isfile(first):
        return [resolve_executable(interpreter, path), *command]
    return [resolve_executable(first, path), *command[1:]]


def detached_process_kwargs() -> dict:
    """``Popen`` keyword arguments for a child that outlives its parent's
    session: its own session on POSIX, its own process group and no console
    window on Windows."""
    if IS_WINDOWS:
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        }
    return {"start_new_session": True}


def popen_detached(argv: list[str], **kwargs) -> subprocess.Popen:
    """Start *argv* detached from this process's session.

    On Windows a harness may run its commands inside a job object that ends
    every process in it when the command returns; the child is asked to
    leave the job, and started inside it when the job does not allow that.
    """
    detached = detached_process_kwargs()
    if IS_WINDOWS:
        breakaway = dict(detached)
        breakaway["creationflags"] |= subprocess.CREATE_BREAKAWAY_FROM_JOB
        try:
            return subprocess.Popen(argv, **kwargs, **breakaway)
        except OSError:
            pass
    return subprocess.Popen(argv, **kwargs, **detached)


def _has_console_mode(fileno: int) -> bool:
    """Whether the Windows handle behind *fileno* is a console.

    Asking the handle for its console mode is the only way to tell: ``NUL``
    and a console are both character devices, which is all ``isatty()``
    reports.
    """
    import ctypes

    try:
        handle = msvcrt.get_osfhandle(fileno)
        mode = ctypes.c_ulong()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except (OSError, ValueError):
        return False


def stdin_is_interactive() -> bool:
    """Whether a person can answer a prompt on stdin.

    ``isatty()`` on POSIX.  On Windows that is also true for ``NUL``, the
    ``/dev/null`` a script redirects stdin from, since both are character
    devices; only a handle with a console mode is a console.
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
        if not IS_WINDOWS:
            return True
        fileno = sys.stdin.fileno()
    except (OSError, ValueError):
        return False
    return _has_console_mode(fileno)


def use_utf8_stdio() -> None:
    """Make this process's stdout and stderr accept any character.

    A Windows pipe or redirected stream defaults to the ANSI code page, and
    printing a character outside it raises; the text dsagt echoes (a code's
    output, a skill's description) is not limited to that page.  A stream
    that is not a console is switched to UTF-8, which is what the agent
    harness reading the pipe expects; a console already takes any character.
    Unencodable characters are replaced either way.  POSIX streams are left
    as they are.
    """
    if not IS_WINDOWS:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

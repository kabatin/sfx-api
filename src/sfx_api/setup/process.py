"""Running the installers: child output goes to stderr (stdout is kept for ``--json``), and a
download runs under a watchdog. A transfer left bound to an old network or trickling at a few
KB/s would otherwise stall for hours: when the download directory stops growing the command
is interrupted (Ctrl-C, partial files kept) and started again, resuming where it left off.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

STALL_S = 600.0
POLL_S = 30.0
MAX_STALLS = 30
MAX_FAILURES = 3
STOP_GRACE_S = 30.0
# what a watched command returns when its directory stopped growing
STALLED = -1000

Emit = Callable[[str], None]


class SetupError(RuntimeError):
    """A setup step cannot continue; the message tells the user what to do."""


class Run(Protocol):
    def __call__(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        watch: Path | None = None,
    ) -> int: ...


def dir_size(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except FileNotFoundError:
                continue  # a partial file renamed while we were walking
    return total


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    watch: Path | None = None,
    stall_s: float = STALL_S,
    poll_s: float = POLL_S,
) -> int:
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr.fileno(),
            start_new_session=True,  # Ctrl-C on a stall reaches the whole installer
        )
    except OSError as exc:
        raise SetupError(f"cannot run {cmd[0]}: {exc}") from exc
    if watch is None:
        return proc.wait()
    last, idle = dir_size(watch), 0.0
    while True:
        try:
            return proc.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
        size = dir_size(watch)
        idle = idle + poll_s if size == last else 0.0
        last = size
        if idle >= stall_s:
            _stop(proc)
            return STALLED


def _stop(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()


def run_resuming(
    run: Run,
    name: str,
    cmd: Sequence[str],
    *,
    watch: Path,
    emit: Emit,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    max_stalls: int = MAX_STALLS,
    max_failures: int = MAX_FAILURES,
) -> None:
    """Run a resumable download until it succeeds; stalls and failures start it again."""
    stalls = failures = 0
    while True:
        code = run(cmd, cwd=cwd, env=env, watch=watch)
        if code == 0:
            return
        if code == STALLED:
            stalls += 1
            if stalls > max_stalls:
                raise SetupError(f"{name}: download did not finish after {max_stalls} restarts")
            emit(f"{name}: stalled; resuming ({stalls}/{max_stalls})")
            continue
        failures += 1
        if failures >= max_failures:
            raise SetupError(f"{name} failed (exit {code}, {failures} tries); see the output above")
        emit(f"{name}: exit {code}; trying again ({failures}/{max_failures - 1})")


def git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"

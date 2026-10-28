"""Installer processes: output kept off stdout, stalled downloads restarted."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sfx_api.setup.process import STALLED, SetupError, dir_size, run_command, run_resuming


def test_child_output_goes_to_stderr(capfd: pytest.CaptureFixture[str]) -> None:
    code = run_command([sys.executable, "-c", "print('hello'); raise SystemExit(3)"])
    out, err = capfd.readouterr()
    assert code == 3 and out == "" and "hello" in err


def test_env_and_cwd_reach_the_child(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    script = "import os; print(os.environ['SFX_X'], os.getcwd())"
    assert run_command([sys.executable, "-c", script], cwd=tmp_path, env={"SFX_X": "v"}) == 0
    assert f"v {tmp_path.resolve()}" in capfd.readouterr().err


def test_a_missing_program_is_a_setup_error() -> None:
    with pytest.raises(SetupError, match="cannot run"):
        run_command(["/nonexistent/tool"])


def test_watchdog_stops_a_stalled_command(tmp_path: Path) -> None:
    code = run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        watch=tmp_path,
        stall_s=0.2,
        poll_s=0.1,
    )
    assert code == STALLED


def test_watchdog_lets_a_growing_download_finish(tmp_path: Path) -> None:
    script = (
        "import time, pathlib\n"
        f"p = pathlib.Path({str(tmp_path / 'f')!r})\n"
        "for i in range(6):\n"
        "    p.write_bytes(b'x' * (i + 1)); time.sleep(0.1)\n"
    )
    code = run_command([sys.executable, "-c", script], watch=tmp_path, stall_s=0.35, poll_s=0.1)
    assert code == 0 and dir_size(tmp_path) == 6


def test_run_resuming_restarts_stalls_and_retries_failures(tmp_path: Path) -> None:
    codes = [STALLED, STALLED, 1, 0]
    calls: list[tuple[str, ...]] = []

    def run(cmd, *, cwd=None, env=None, watch=None) -> int:
        calls.append(tuple(cmd))
        return codes.pop(0)

    lines: list[str] = []
    run_resuming(run, "dl", ["hf"], watch=tmp_path, emit=lines.append)
    assert len(calls) == 4 and any("resuming (2/" in line for line in lines)


def test_run_resuming_gives_up(tmp_path: Path) -> None:
    def stalls(cmd, *, cwd=None, env=None, watch=None) -> int:
        return STALLED

    with pytest.raises(SetupError, match="after 2 restarts"):
        run_resuming(stalls, "dl", ["hf"], watch=tmp_path, emit=lambda _: None, max_stalls=2)

    def fails(cmd, *, cwd=None, env=None, watch=None) -> int:
        return 2

    with pytest.raises(SetupError, match="exit 2, 3 tries"):
        run_resuming(fails, "dl", ["hf"], watch=tmp_path, emit=lambda _: None)

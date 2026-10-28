"""``--service``: run ``sfx-api serve`` at login with launchd, then wait for it to answer."""

from __future__ import annotations

import os
import plistlib
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sfx_api.setup.files import make_private_dirs, write_atomic
from sfx_api.setup.process import Emit, Run, SetupError

LABEL = "com.github.kabatin.sfx-api"
HEALTH_TIMEOUT_S = 60.0
HEALTH_POLL_S = 2.0
BOOTOUT_WAIT_S = 10.0
BOOTOUT_POLL_S = 0.5
# EIO and EALREADY: the previous job is still being removed
BOOTSTRAP_BUSY = (5, 37)
BOOTSTRAP_TRIES = 5
_SUFFIX = re.compile(r"[A-Za-z0-9-]{1,32}")
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# GET url with an optional bearer token -> HTTP status, or None when nothing answered
HttpGet = Callable[[str, str | None], int | None]


def label_for(suffix: str | None) -> str:
    if suffix is None:
        return LABEL
    if not _SUFFIX.fullmatch(suffix):
        raise ValueError("--label-suffix takes 1-32 letters, digits or dashes")
    return f"{LABEL}.{suffix}"


def plist_path(home: Path, label: str) -> Path:
    return home / "Library" / "LaunchAgents" / f"{label}.plist"


def log_path(home: Path, label: str) -> Path:
    name = label.removeprefix("com.github.kabatin.")
    return home / "Library" / "Logs" / f"{name}.log"


def render_plist(label: str, program: str, log: Path, config_path: Path | None) -> bytes:
    env = {"PATH": PATH}
    if config_path is not None:
        env["SFX_API_CONFIG"] = str(config_path)
    return plistlib.dumps(
        {
            "Label": label,
            "ProgramArguments": [program, "serve"],
            "EnvironmentVariables": env,
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
    )


def http_get(url: str, token: str | None, timeout: float = 2.0) -> int | None:
    # straight to 127.0.0.1: no proxy, so the token goes nowhere else
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url)  # noqa: S310 - always http://127.0.0.1
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return None


@dataclass(frozen=True)
class ServiceDeps:
    run: Run
    http_get: HttpGet
    emit: Emit
    home: Path
    uid: int = field(default_factory=os.getuid)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    timeout_s: float = HEALTH_TIMEOUT_S


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/health"


def is_healthy(deps: ServiceDeps, port: int, token: str | None) -> bool:
    return deps.http_get(health_url(port), token) == 200


def _wait_until_gone(deps: ServiceDeps, target: str) -> None:
    """bootout returns before launchd has removed the job; bootstrapping it again too soon
    fails with "5: Input/output error"."""
    deadline = deps.clock() + BOOTOUT_WAIT_S
    while deps.run(["launchctl", "print", target], quiet=True) == 0:
        if deps.clock() >= deadline:
            deps.emit(f"launchd: {target} still listed after {BOOTOUT_WAIT_S:.0f} s")
            return
        deps.sleep(BOOTOUT_POLL_S)


def _bootstrap(deps: ServiceDeps, plist: Path, log: Path) -> None:
    cmd = ["launchctl", "bootstrap", f"gui/{deps.uid}", str(plist)]
    for attempt in range(1, BOOTSTRAP_TRIES + 1):
        code = deps.run(cmd)
        if code == 0:
            return
        if code not in BOOTSTRAP_BUSY or attempt == BOOTSTRAP_TRIES:
            raise SetupError(f"launchctl bootstrap failed (exit {code}); see {log}")
        deps.emit(f"launchd: bootstrap busy (exit {code}); trying again")
        deps.sleep(BOOTOUT_POLL_S)


def install_service(
    deps: ServiceDeps,
    *,
    label: str,
    program: str,
    config_path: Path | None,
    port: int,
    token: str | None,
) -> bool:
    """Write the plist, (re)load it and wait up to a minute for ``/v1/health``."""
    plist = plist_path(deps.home, label)
    log = log_path(deps.home, label)
    make_private_dirs(plist.parent)
    make_private_dirs(log.parent)
    write_atomic(plist, render_plist(label, program, log, config_path), mode=0o644)
    deps.emit(f"launchd: loading {plist}")
    target = f"gui/{deps.uid}/{label}"
    deps.run(["launchctl", "bootout", target])  # not loaded is fine
    _wait_until_gone(deps, target)
    _bootstrap(deps, plist, log)
    deadline = deps.clock() + deps.timeout_s
    while True:
        if is_healthy(deps, port, token):
            deps.emit(f"launchd: {label} answers on {health_url(port)}")
            return True
        if deps.clock() >= deadline:
            deps.emit(f"launchd: {label} did not answer within {deps.timeout_s:.0f} s; see {log}")
            return False
        deps.sleep(HEALTH_POLL_S)

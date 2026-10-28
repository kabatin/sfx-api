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

from sfx_api.setup.process import Emit, Run, SetupError

LABEL = "com.github.kabatin.sfx-api"
HEALTH_TIMEOUT_S = 60.0
HEALTH_POLL_S = 2.0
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
    plist.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(render_plist(label, program, log, config_path))
    deps.emit(f"launchd: loading {plist}")
    deps.run(["launchctl", "bootout", f"gui/{deps.uid}/{label}"])  # not loaded is fine
    code = deps.run(["launchctl", "bootstrap", f"gui/{deps.uid}", str(plist)])
    if code != 0:
        raise SetupError(f"launchctl bootstrap failed (exit {code}); see {log}")
    deadline = deps.clock() + deps.timeout_s
    while True:
        if is_healthy(deps, port, token):
            deps.emit(f"launchd: {label} answers on {health_url(port)}")
            return True
        if deps.clock() >= deadline:
            deps.emit(f"launchd: {label} did not answer within {deps.timeout_s:.0f} s; see {log}")
            return False
        deps.sleep(HEALTH_POLL_S)

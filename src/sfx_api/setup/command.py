"""``sfx-api setup``: the uniform install command (one per AI server, same flags and JSON).

``--check`` changes nothing. Otherwise: download what is missing (after one confirmation),
add the installed engines to the config, and with ``--service`` run the server at login.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sfx_api import __version__
from sfx_api.settings import DEFAULT_PORT, ENV_PREFIX, is_loopback, resolve_config_path
from sfx_api.setup import service
from sfx_api.setup.components import (
    BY_ID,
    DEFAULT_PREFIX,
    IDS,
    SPECS,
    MossPaths,
    Paths,
    Sa3Paths,
    Spec,
    config_table,
    missing_downloads,
    paths_for,
    unavailable_reason,
)
from sfx_api.setup.config_write import (
    ConfigRequest,
    Peer,
    check_peer,
    env_token,
    parse_peer,
    plan_config,
    read_config,
    write_config,
)
from sfx_api.setup.installers import HeadCommit, Tools, install_moss, install_sa3
from sfx_api.setup.machine import HasTool, Machine, default_has_tool, probe_machine
from sfx_api.setup.pins import SA3_BUNDLES
from sfx_api.setup.process import Run, SetupError, git_head, run_command
from sfx_api.setup.report import ComponentResult, Fix, Report, ServiceState, gb, human_lines

SERVER = "sfx-api"
# free disk wanted beyond the downloads: unpacking, the OS, a margin
DISK_MARGIN_BYTES = 2_000_000_000
DISK_FACTOR = 1.1

EXIT_OK, EXIT_FAILED, EXIT_BAD_ARGS, EXIT_CONFIRM = 0, 1, 2, 3


def _stderr(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _ask(prompt: str) -> str:
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return sys.stdin.readline()


@dataclass(frozen=True)
class SetupDeps:
    run: Run = run_command
    has_tool: HasTool = default_has_tool
    head_commit: HeadCommit = git_head
    http_get: service.HttpGet = service.http_get
    which: Callable[[str], str | None] = shutil.which
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    home: Path = field(default_factory=Path.home)
    is_tty: Callable[[], bool] = lambda: sys.stdin.isatty()
    ask: Callable[[str], str] = _ask
    emit: Callable[[str], None] = _stderr
    probe: Callable[[Path, HasTool], Machine] = probe_machine
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic


def add_parser(sub: Any) -> None:
    p = sub.add_parser("setup", help="install engines and weights, write the config")
    p.add_argument("--models", help=f"comma list of {', '.join(IDS)} (default: all)")
    p.add_argument("--check", action="store_true", help="change nothing, report state")
    p.add_argument("--yes", action="store_true", help="download without asking")
    p.add_argument("--json", action="store_true", help="one JSON object on stdout")
    p.add_argument("--lan", action="store_true", help="listen on the LAN (sets a token)")
    p.add_argument("--service", action="store_true", help="run the server at login (launchd)")
    p.add_argument("--port", type=int)
    p.add_argument("--prefix", help=f"where engines go (default {DEFAULT_PREFIX})")
    p.add_argument("--label-suffix", help="launchd label suffix, for a second install")
    p.add_argument(
        "--coordinate-with",
        action="append",
        default=[],
        metavar="NAME=HEALTH_URL",
        help="another GPU server to wait for (repeatable)",
    )


@dataclass(frozen=True)
class Context:
    args: argparse.Namespace
    deps: SetupDeps
    config_path: Path
    config: dict[str, Any]
    prefix: Path
    machine: Machine
    label: str


@dataclass(frozen=True)
class Planned:
    spec: Spec
    paths: Paths
    result: ComponentResult


def cmd_setup(config: str | None, args: argparse.Namespace, deps: SetupDeps | None = None) -> int:
    deps = deps or SetupDeps()
    code, report = run_setup(config, args, deps)
    if args.json:
        for line in human_lines(report):
            deps.emit(line)
        sys.stdout.write(json.dumps(report.to_json()) + "\n")
    else:
        sys.stdout.write("\n".join(human_lines(report)) + "\n")
    sys.stdout.flush()
    return code


def cmd_usage_error(message: str, deps: SetupDeps | None = None) -> int:
    """``setup --json`` with arguments argparse refused: still one JSON object, exit 2."""
    deps = deps or SetupDeps()
    config_path = resolve_config_path(deps.env, None)
    machine = deps.probe(DEFAULT_PREFIX.expanduser(), deps.has_tool)
    args = argparse.Namespace(check=False)
    report = _bare_report(config_path, machine, deps, args, message)
    sys.stdout.write(json.dumps(report.to_json()) + "\n")
    sys.stdout.flush()
    return EXIT_BAD_ARGS


def _selected(models: str | None) -> list[Spec]:
    if not models:
        return list(SPECS)
    ids = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in ids if m not in BY_ID]
    if unknown or not ids:
        raise ValueError(
            f"unknown component(s) {', '.join(unknown) or '-'}; valid: {', '.join(IDS)}"
        )
    return [spec for spec in SPECS if spec.id in ids]


def run_setup(config: str | None, args: argparse.Namespace, deps: SetupDeps) -> tuple[int, Report]:
    config_path = resolve_config_path(deps.env, Path(config) if config else None)
    prefix = Path(args.prefix or DEFAULT_PREFIX).expanduser().absolute()
    machine = deps.probe(prefix, deps.has_tool)
    try:
        label = service.label_for(args.label_suffix)
        specs = _selected(args.models)
        peers = [parse_peer(text) for text in args.coordinate_with]
        for peer in peers:
            check_peer(peer, deps.env)
        if args.lan:
            env_token(deps.env)
        if args.port is not None and not 1 <= args.port <= 65535:
            raise ValueError("--port must be 1-65535")
        if args.label_suffix is not None and args.port is None:
            # a second install on the first one's port would fight it for the socket
            raise ValueError("--label-suffix needs --port (a port of its own)")
    except ValueError as exc:
        return EXIT_BAD_ARGS, _bare_report(config_path, machine, deps, args, str(exc))
    try:
        existing = read_config(config_path)
    except SetupError as exc:
        return EXIT_FAILED, _bare_report(config_path, machine, deps, args, str(exc))
    ctx = Context(args, deps, config_path, existing, prefix, machine, label)
    planned = [_plan(ctx, spec) for spec in specs]
    if args.check:
        return EXIT_OK, _report(ctx, existing, planned, ok=True)
    return _install(ctx, planned, peers)


def _plan(ctx: Context, spec: Spec) -> Planned:
    backends = ctx.config.get("backends", {})
    paths = paths_for(spec, backends if isinstance(backends, dict) else {}, ctx.prefix)
    m = ctx.machine
    if not m.apple_silicon:
        why = "needs an Apple Silicon Mac (MLX / MPS)"
        return Planned(spec, paths, ComponentResult(spec.id, "unsupported", why))
    if m.ram_gb and m.ram_gb < spec.min_ram_gb:
        why = f"needs {spec.min_ram_gb} GB of memory; this Mac has {m.ram_gb} GB"
        return Planned(spec, paths, ComponentResult(spec.id, "unsupported", why))
    reason = unavailable_reason(spec, paths)
    if reason is None:
        where = paths.code_dir if isinstance(paths, MossPaths) else paths.mlx_dir
        return Planned(spec, paths, ComponentResult(spec.id, "installed", f"{where}"))
    command = f"sfx-api setup --models {spec.id} --yes"
    fix = Fix(f"run `{command}`", command)
    return Planned(spec, paths, ComponentResult(spec.id, "missing", reason, fix=fix))


def _with_shared_bytes_once(planned: list[Planned]) -> list[Planned]:
    seen: set[str] = set()
    out = []
    for item in planned:
        if item.result.status != "missing":
            out.append(item)
            continue
        parts = missing_downloads(item.spec, item.paths)
        size = sum(v for k, v in parts.items() if k not in seen)
        seen.update(parts)
        out.append(replace(item, result=replace(item.result, download_bytes=size)))
    return out


Blocker = Callable[[Planned], ComponentResult]


def _install(ctx: Context, planned: list[Planned], peers: Sequence[Peer]) -> tuple[int, Report]:
    planned = _with_shared_bytes_once(planned)
    todo = [p for p in planned if p.result.status == "missing"]
    blocker = _tools_blocker(ctx) if todo else None
    blocker = blocker or (_disk_blocker(ctx, todo) if todo else None)
    if blocker is not None:
        stopped = [replace(p, result=blocker(p)) if p in todo else p for p in planned]
        return _exit_code(stopped), _report(ctx, ctx.config, stopped, ok=False)
    if todo and not ctx.args.yes and not _confirmed(ctx, todo):
        consent = Fix("rerun with --yes to download", _rerun_command(ctx), needs_human=True)
        asked = [
            replace(p, result=replace(p.result, fix=consent)) if p in todo else p for p in planned
        ]
        return EXIT_CONFIRM, _report(ctx, ctx.config, asked, ok=False)
    done = [_already(p) for p in planned if p not in todo] + _run_installs(ctx, todo)
    done.sort(key=lambda p: IDS.index(p.spec.id))
    return _configure(ctx, done, peers)


def _already(item: Planned) -> Planned:
    if item.result.status != "installed":
        return item
    return replace(item, result=replace(item.result, status="already"))


def _tools_blocker(ctx: Context) -> Blocker | None:
    tools = ctx.machine.tools
    if not tools.get("git", False):
        fix = Fix(
            "install the Xcode command line tools (a dialog on the Mac's screen)",
            "xcode-select --install",
            needs_human=True,
        )
        return lambda p: replace(p.result, status="needs_human", message="git not found", fix=fix)
    if not tools.get("uv", False):
        fix = Fix("install uv", "brew install uv")
        return lambda p: replace(p.result, status="failed", message="uv not found", fix=fix)
    return None


def _disk_blocker(ctx: Context, todo: list[Planned]) -> Blocker | None:
    total = sum(p.result.download_bytes for p in todo)
    needed = int(total * DISK_FACTOR) + DISK_MARGIN_BYTES
    free = ctx.machine.free_bytes
    if free >= needed:
        return None
    message = (
        f"not enough disk: {gb(total)} to download needs ~{gb(needed)} free, "
        f"{gb(free)} free at {ctx.prefix}"
    )
    fix = Fix("free disk space, or pass --prefix on a bigger volume", needs_human=True)
    return lambda p: replace(p.result, status="failed", message=message, fix=fix)


def _confirmed(ctx: Context, todo: list[Planned]) -> bool:
    if not ctx.deps.is_tty():
        return False
    emit = ctx.deps.emit
    total = sum(p.result.download_bytes for p in todo)
    emit(f"setup will download about {gb(total)}:")
    for p in todo:
        emit(f"  {p.spec.label} ({p.spec.id}): {gb(p.result.download_bytes)}")
        emit(f"    licence: {p.spec.licence.name} — {p.spec.licence.note}")
    return ctx.deps.ask("continue? [y/N] ").strip().lower() in ("y", "yes")


def _rerun_command(ctx: Context) -> str:
    args = ctx.args
    parts = ["sfx-api"]
    if ctx.config_path != resolve_config_path({}, None):
        parts += ["--config", str(ctx.config_path)]
    parts.append("setup")
    if args.models:
        parts += ["--models", args.models]
    if args.prefix:
        parts += ["--prefix", str(ctx.prefix)]
    for flag in ("lan", "service", "json"):
        if getattr(args, flag):
            parts.append(f"--{flag}")
    if args.port is not None:
        parts += ["--port", str(args.port)]
    if args.label_suffix:
        parts += ["--label-suffix", args.label_suffix]
    for peer in args.coordinate_with:
        parts += ["--coordinate-with", peer]
    return " ".join([*parts, "--yes"])


def _run_installs(ctx: Context, todo: list[Planned]) -> list[Planned]:
    tools = Tools(run=ctx.deps.run, emit=ctx.deps.emit, head_commit=ctx.deps.head_commit)
    out: list[Planned] = []
    groups: dict[Sa3Paths, list[Planned]] = {}
    for p in todo:
        if isinstance(p.paths, Sa3Paths):
            groups.setdefault(p.paths, []).append(p)
    for sa3, group in groups.items():
        bundles = [SA3_BUNDLES[p.spec.id] for p in group]
        out.extend(_attempt(group, lambda s=sa3, b=bundles: install_sa3(s, b, tools)))
    for p in todo:
        if isinstance(p.paths, MossPaths):
            moss = p.paths
            out.extend(_attempt([p], lambda m=moss: install_moss(m, tools)))
    return out


def _attempt(group: list[Planned], step: Callable[[], None]) -> list[Planned]:
    try:
        step()
    except (SetupError, OSError) as exc:
        return [
            replace(p, result=replace(p.result, status="failed", message=str(exc))) for p in group
        ]
    out = []
    for p in group:
        reason = unavailable_reason(p.spec, p.paths)
        if reason is None:
            out.append(
                replace(
                    p, result=replace(p.result, status="installed", message="installed", fix=None)
                )
            )
        else:
            message = f"still unavailable after setup: {reason}"
            out.append(replace(p, result=replace(p.result, status="failed", message=message)))
    return out


def _exit_code(planned: Sequence[Planned]) -> int:
    statuses = {p.result.status for p in planned}
    if "failed" in statuses:
        return EXIT_FAILED
    return EXIT_CONFIRM if "needs_human" in statuses else EXIT_OK


def _requested_unsupported(ctx: Context, planned: Sequence[Planned]) -> list[Planned]:
    """Asked for by name but impossible here: a failure (with all, it is just skipped)."""
    if not ctx.args.models:
        return []
    return [p for p in planned if p.result.status == "unsupported"]


def _configure(ctx: Context, done: list[Planned], peers: Sequence[Peer]) -> tuple[int, Report]:
    ready = [p for p in done if p.result.status in ("installed", "already")]
    request = ConfigRequest(
        backends={p.spec.backend: config_table(p.spec, p.paths) for p in ready},
        lan=ctx.args.lan,
        port=ctx.args.port,
        peers=peers,
    )
    notes: list[str] = []
    data = ctx.config
    try:
        planned_data = plan_config(ctx.config, request, ctx.deps.env)
        if planned_data != ctx.config:
            write_config(ctx.config_path, planned_data)
            notes.append(f"config updated: {ctx.config_path}")
        data = planned_data
    except SetupError as exc:
        report = _report(ctx, data, done, ok=False, error=str(exc))
        return EXIT_FAILED, report
    code = _exit_code(done)
    if _requested_unsupported(ctx, done):
        code = EXIT_FAILED
    service_state: ServiceState | None = None
    if ctx.args.service:
        service_state, error = _install_service(ctx, data)
        if error is not None:
            report = _report(ctx, data, done, ok=False, error=error, service_state=service_state)
            return EXIT_FAILED, report
    report = _report(
        ctx, data, done, ok=code == EXIT_OK, service_state=service_state, notes=tuple(notes)
    )
    return code, report


def _install_service(ctx: Context, data: Mapping[str, Any]) -> tuple[ServiceState, str | None]:
    program = ctx.deps.which(SERVER)
    if program is None:
        state = ServiceState(installed=False, label=ctx.label, healthy=False)
        return (
            state,
            "sfx-api is not on PATH: uv tool install git+https://github.com/kabatin/sfx-api",
        )
    deps = _service_deps(ctx)
    default = resolve_config_path({}, None)
    try:
        healthy = service.install_service(
            deps,
            label=ctx.label,
            program=str(Path(program).absolute()),
            config_path=None if ctx.config_path == default else ctx.config_path,
            port=_port(ctx, data),
            token=_token(ctx, data),
        )
    except SetupError as exc:
        return ServiceState(installed=True, label=ctx.label, healthy=False), str(exc)
    state = ServiceState(installed=True, label=ctx.label, healthy=healthy)
    if not healthy:
        log = service.log_path(ctx.deps.home, ctx.label)
        return state, f"the service did not answer within 60 s; see {log}"
    return state, None


def _service_deps(ctx: Context) -> service.ServiceDeps:
    d = ctx.deps
    return service.ServiceDeps(
        run=d.run, http_get=d.http_get, emit=d.emit, home=d.home, sleep=d.sleep, clock=d.clock
    )


def _port(ctx: Context, data: Mapping[str, Any]) -> int:
    raw = ctx.deps.env.get(ENV_PREFIX + "PORT") or data.get("port") or DEFAULT_PORT
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PORT


def _token(ctx: Context, data: Mapping[str, Any]) -> str | None:
    token = ctx.deps.env.get(ENV_PREFIX + "TOKEN") or data.get("token")
    return token if isinstance(token, str) and token else None


def _lan(ctx: Context, data: Mapping[str, Any]) -> bool:
    host = ctx.deps.env.get(ENV_PREFIX + "HOST") or data.get("host") or "127.0.0.1"
    return not is_loopback(str(host))


def _current_service(ctx: Context, data: Mapping[str, Any]) -> ServiceState:
    installed = service.plist_path(ctx.deps.home, ctx.label).is_file()
    healthy = installed and service.is_healthy(
        _service_deps(ctx), _port(ctx, data), _token(ctx, data)
    )
    return ServiceState(installed=installed, label=ctx.label, healthy=healthy)


def _report(
    ctx: Context,
    data: Mapping[str, Any],
    planned: Sequence[Planned],
    ok: bool,
    *,
    error: str | None = None,
    service_state: ServiceState | None = None,
    notes: tuple[str, ...] = (),
) -> Report:
    results = tuple(p.result for p in _with_shared_bytes_once(list(planned)))
    return Report(
        server=SERVER,
        version=__version__,
        ok=ok,
        action="check" if ctx.args.check else "install",
        url=f"http://127.0.0.1:{_port(ctx, data)}",
        lan=_lan(ctx, data),
        token_set=_token(ctx, data) is not None,
        config_path=str(ctx.config_path),
        service=service_state or _current_service(ctx, data),
        machine=ctx.machine,
        components=results,
        error=error,
        notes=notes,
    )


def _bare_report(
    config_path: Path, machine: Machine, deps: SetupDeps, args: argparse.Namespace, error: str
) -> Report:
    return Report(
        server=SERVER,
        version=__version__,
        ok=False,
        action="check" if args.check else "install",
        url=f"http://127.0.0.1:{DEFAULT_PORT}",
        lan=False,
        token_set=False,
        config_path=str(config_path),
        service=ServiceState(installed=False, label=service.LABEL, healthy=False),
        machine=machine,
        error=error,
    )

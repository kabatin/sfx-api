"""What ``setup`` found and did, as one JSON object (``--json``) or readable lines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sfx_api.setup.machine import Machine


@dataclass(frozen=True)
class Fix:
    hint: str
    command: str | None = None
    needs_human: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"hint": self.hint, "command": self.command, "needsHuman": self.needs_human}


@dataclass(frozen=True)
class ComponentResult:
    id: str
    status: str  # installed | already | missing | unsupported | failed | needs_human
    message: str
    download_bytes: int = 0
    fix: Fix | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "downloadBytes": self.download_bytes,
        }
        if self.fix is not None:
            out["fix"] = self.fix.to_json()
        return out


@dataclass(frozen=True)
class ServiceState:
    installed: bool
    label: str
    healthy: bool

    def to_json(self) -> dict[str, Any]:
        return {"installed": self.installed, "label": self.label, "healthy": self.healthy}


@dataclass(frozen=True)
class Report:
    server: str
    version: str
    ok: bool
    action: str
    url: str
    lan: bool
    token_set: bool
    config_path: str
    service: ServiceState
    machine: Machine
    components: tuple[ComponentResult, ...] = ()
    error: str | None = None
    notes: tuple[str, ...] = field(default=())

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "server": self.server,
            "version": self.version,
            "ok": self.ok,
            "action": self.action,
            "url": self.url,
            "lan": self.lan,
            "tokenSet": self.token_set,
            "configPath": self.config_path,
            "service": self.service.to_json(),
            "machine": self.machine.to_json(),
            "components": [c.to_json() for c in self.components],
        }
        if self.error is not None:
            out["error"] = self.error
        return out


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def human_lines(report: Report) -> list[str]:
    m = report.machine
    tools = ", ".join(f"{name} {'yes' if ok else 'no'}" for name, ok in m.tools.items())
    lines = [
        f"{report.server} {report.version} setup ({report.action}): "
        + ("ok" if report.ok else "not ok"),
        f"  machine: {'arm64' if m.arm64 else 'not arm64'}, macOS {m.macos or '-'}, "
        f"{m.ram_gb} GB RAM, {gb(m.free_bytes)} free; {tools}",
        f"  config:  {report.config_path} (lan: {'yes' if report.lan else 'no'}, "
        f"token: {'set' if report.token_set else 'none'})",
        f"  url:     {report.url}",
        f"  service: {report.service.label} "
        + (
            ("healthy" if report.service.healthy else "not answering")
            if report.service.installed
            else "not installed"
        ),
    ]
    if report.error:
        lines.append(f"  error:   {report.error}")
    for c in report.components:
        size = f", {gb(c.download_bytes)} to download" if c.download_bytes else ""
        lines.append(f"  {c.id:<11} {c.status}{size}: {c.message}")
        if c.fix is not None:
            lines.append(f"              fix: {c.fix.hint}")
            if c.fix.command:
                lines.append(f"                   {c.fix.command}")
    lines.extend(f"  note: {note}" for note in report.notes)
    return lines

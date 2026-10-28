"""Facts about this Mac that decide what ``setup`` can install."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS = ("git", "uv")

HasTool = Callable[[str], bool]


def default_has_tool(name: str) -> bool:
    return shutil.which(name) is not None


@dataclass(frozen=True)
class Machine:
    arm64: bool
    macos: str | None
    ram_gb: int
    free_bytes: int
    tools: dict[str, bool] = field(default_factory=dict)

    @property
    def apple_silicon(self) -> bool:
        return self.arm64 and self.macos is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "arm64": self.arm64,
            "macos": self.macos,
            "ramGb": self.ram_gb,
            "freeBytes": self.free_bytes,
            "tools": dict(self.tools),
        }


def existing_parent(path: Path) -> Path:
    """The nearest directory that exists (where the disk space will be taken from)."""
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(existing_parent(path)).free


def _ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def _arm64() -> bool:
    """The hardware, not this Python: an x86 Python under Rosetta still sees an Apple chip."""
    if platform.machine() == "arm64":
        return True
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, check=False
        )
    except OSError:
        return False
    return result.stdout.strip() == "1"


def probe_machine(prefix: Path, has_tool: HasTool = default_has_tool) -> Machine:
    mac = platform.mac_ver()[0] if platform.system() == "Darwin" else ""
    return Machine(
        arm64=_arm64(),
        macos=mac or None,
        ram_gb=round(_ram_bytes() / 2**30),
        free_bytes=free_bytes(prefix),
        tools={name: has_tool(name) for name in TOOLS},
    )

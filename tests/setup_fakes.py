"""Fakes for ``sfx-api setup`` tests: no network, no downloads, no launchctl."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sfx_api.backends.moss import MODEL_FILES as MOSS_FILES
from sfx_api.setup import pins
from sfx_api.setup.command import SetupDeps
from sfx_api.setup.machine import Machine

APPLE = Machine(
    arm64=True, macos="26.0", ram_gb=32, free_bytes=200 * 10**9, tools={"git": True, "uv": True}
)


def make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


@dataclass
class FakeInstallers:
    """Plays git, install.sh, uv, uvx hf and launchctl by creating what they would create."""

    calls: list[tuple[str, ...]] = field(default_factory=list)
    envs: list[Mapping[str, str]] = field(default_factory=list)
    fail: set[str] = field(default_factory=set)  # a first word, a command prefix or an argument
    clones: dict[str, str] = field(default_factory=dict)  # dest -> repo
    # exit codes per command prefix, in order; the last one repeats
    codes: dict[str, list[int]] = field(default_factory=dict)
    quiet: list[bool] = field(default_factory=list)

    def __call__(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        watch: Path | None = None,
        quiet: bool = False,
    ) -> int:
        cmd = tuple(cmd)
        self.calls.append(cmd)
        self.envs.append(dict(env or {}))
        self.quiet.append(quiet)
        line = " ".join(cmd)
        for prefix, codes in self.codes.items():
            if line.startswith(prefix):
                return codes.pop(0) if len(codes) > 1 else codes[0]
        if any(cmd[0] == f or line.startswith(f + " ") or f in cmd for f in self.fail):
            return 1
        if cmd[:2] == ("git", "clone"):
            Path(cmd[-1]).mkdir(parents=True)
            self.clones[cmd[-1]] = cmd[-2]
        elif cmd[:2] == ("git", "-C"):
            self._checkout(Path(cmd[2]))
        elif cmd[0] == "./install.sh":
            assert cwd is not None
            make_exe(cwd / ".venv" / "bin" / "python")
            bundles = cmd[cmd.index("--download") + 1].split(",")
            for name in [n for b in bundles for n in pins.SA3_FILES[b]] + list(
                pins.SA3_SHARED_FILES
            ):
                touch(cwd / "models" / "mlx" / name)
        elif cmd[:2] == ("uv", "venv"):
            make_exe(Path(cmd[-1]) / "bin" / "python")
        elif cmd[0] == "uvx" and "--local-dir" in cmd:
            local = Path(cmd[cmd.index("--local-dir") + 1])
            for name in MOSS_FILES:
                touch(local / name)
        return 0

    def _checkout(self, dest: Path) -> None:
        repo = self.clones[str(dest)]
        if repo == pins.SA3_REPO:
            touch(dest / "optimized" / "mlx" / "scripts" / "sa3_mlx.py")
        else:
            touch(dest / "moss_soundeffect_v2" / "pipeline_moss_soundeffect.py")

    def commands(self, first: str) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[0] == first]


@dataclass
class FakeHttp:
    statuses: list[int | None] = field(default_factory=lambda: [200])
    seen: list[tuple[str, str | None]] = field(default_factory=list)

    def __call__(self, url: str, token: str | None) -> int | None:
        self.seen.append((url, token))
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


def deps(
    tmp_path: Path,
    *,
    run: FakeInstallers | None = None,
    machine: Machine = APPLE,
    env: Mapping[str, str] | None = None,
    tty: bool = False,
    answer: str = "n",
    http: FakeHttp | None = None,
    which: str | None = "/Users/x/.local/bin/sfx-api",
) -> tuple[SetupDeps, list[str]]:
    lines: list[str] = []
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    base_env = {"SFX_API_CONFIG": str(tmp_path / "config" / "config.toml")}
    clock = iter(range(0, 10_000, 5))
    return (
        SetupDeps(
            run=run or FakeInstallers(),
            has_tool=lambda _: True,
            head_commit=lambda path: (
                pins.SA3_COMMIT if "stable-audio-3" in str(path) else pins.MOSS_COMMIT
            ),
            http_get=http or FakeHttp(),
            which=lambda _: which,
            env={**base_env, **(env or {})},
            home=home,
            is_tty=lambda: tty,
            ask=lambda _: answer,
            emit=lines.append,
            probe=lambda _prefix, _has_tool: machine,
            sleep=lambda _: None,
            clock=lambda: float(next(clock)),
        ),
        lines,
    )


def tree(root: Path) -> list[str]:
    return sorted(str(Path(d, f).relative_to(root)) for d, _, fs in os.walk(root) for f in fs)

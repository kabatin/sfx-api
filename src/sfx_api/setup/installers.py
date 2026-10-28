"""The install steps: Stability AI's own MLX installer for Stable Audio 3, and the README's
manual steps for MOSS-SoundEffect. Every step is safe to run again: an existing checkout is
kept, ``install.sh`` skips present weights, and both downloads resume."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sfx_api.setup import pins
from sfx_api.setup.components import MossPaths, Sa3Paths
from sfx_api.setup.process import Emit, Run, SetupError, git_head, run_resuming

HeadCommit = Callable[[Path], str]


@dataclass(frozen=True)
class Tools:
    run: Run
    emit: Emit
    head_commit: HeadCommit = git_head


def ensure_checkout(dest: Path, repo: str, commit: str, tools: Tools) -> None:
    if dest.exists():
        tools.emit(f"using existing checkout {dest} (not changing its version)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # an interrupted clone of ours: start it again
    tools.emit(f"cloning {repo} at {commit[:12]} into {dest}")
    # blobs only for the pinned commit
    _must(tools, ["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(partial)])
    _must(tools, ["git", "-C", str(partial), "checkout", "--quiet", "--detach", commit])
    actual = tools.head_commit(partial)
    if actual != commit:
        raise SetupError(f"{repo} checked out {actual}, expected {commit}")
    partial.rename(dest)


def _must(tools: Tools, cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    code = tools.run(cmd, cwd=cwd)
    if code != 0:
        raise SetupError(f"`{' '.join(cmd[:3])} …` failed (exit {code}); see the output above")


def install_sa3(paths: Sa3Paths, bundles: Sequence[str], tools: Tools) -> None:
    checkout = paths.checkout
    if checkout is None and not paths.mlx_dir.is_dir():
        raise SetupError(
            f"mlx_dir {paths.mlx_dir} is not <checkout>/optimized/mlx, so setup cannot clone "
            "it: fix mlx_dir in the config or clone stable-audio-3 there by hand"
        )
    if checkout is not None:
        ensure_checkout(checkout, pins.SA3_REPO, pins.SA3_COMMIT, tools)
    paths.hf_home.mkdir(parents=True, exist_ok=True)
    tools.emit(f"Stable Audio 3: install.sh --download {','.join(bundles)}")
    cmd = ["./install.sh", "-y", "--python", pins.SA3_PYTHON, "--download", ",".join(bundles)]
    run_resuming(
        tools.run,
        "Stable Audio 3",
        cmd,
        cwd=paths.mlx_dir,
        env={"HF_HOME": str(paths.hf_home)},
        watch=paths.hf_home,
        emit=tools.emit,
    )


def install_moss(paths: MossPaths, tools: Tools) -> None:
    ensure_checkout(paths.code_dir, pins.MOSS_REPO, pins.MOSS_COMMIT, tools)
    if not paths.python.exists():
        tools.emit(f"MOSS: creating {paths.venv} (Python {pins.MOSS_PYTHON})")
        _must(tools, ["uv", "venv", "--python", pins.MOSS_PYTHON, str(paths.venv)])
    tools.emit("MOSS: installing PyTorch and the pipeline's pinned packages")
    _must(tools, ["uv", "pip", "install", "--python", str(paths.python), *pins.MOSS_PACKAGES])
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    tools.emit(f"MOSS: downloading {pins.MOSS_MODEL_REPO} into {paths.model_dir}")
    cmd = [
        "uvx",
        "--from",
        pins.HF_CLI,
        "hf",
        "download",
        pins.MOSS_MODEL_REPO,
        "--revision",
        pins.MOSS_MODEL_REVISION,
        "--local-dir",
        str(paths.model_dir),
    ]
    run_resuming(tools.run, "MOSS-SoundEffect-v2.0", cmd, watch=paths.model_dir, emit=tools.emit)

"""The three things ``setup`` can install, where they go, whether they are there already and
how many bytes are missing. "Installed" means exactly what the backends call available."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sfx_api.backends.moss import MODEL_FILES as MOSS_MODEL_FILES
from sfx_api.backends.moss import MossBackend, MossOptions
from sfx_api.backends.sa3 import Sa3Backend, Sa3MediumBackend, Sa3Options
from sfx_api.setup import pins
from sfx_api.setup.process import dir_size

DEFAULT_PREFIX = Path("~/.local/opt/sfx-api")


@dataclass(frozen=True)
class Spec:
    id: str
    backend: str  # its ``[backends.<name>]`` table
    label: str
    workflows: tuple[str, ...]
    licence: pins.Licence
    # below this the model swaps (peak footprints in the README's measurements)
    min_ram_gb: int


SPECS: tuple[Spec, ...] = (
    Spec(
        "sa3-medium",
        "sa3-medium",
        "Stable Audio 3 medium",
        ("stable-audio-3-medium-sfx", "stable-audio-3-medium-music"),
        pins.SA3_LICENCE,
        16,
    ),
    Spec(
        "sa3-small",
        "sa3",
        "Stable Audio 3 sm-sfx",
        ("stable-audio-3-sfx",),
        pins.SA3_LICENCE,
        8,
    ),
    Spec("moss", "moss", "MOSS-SoundEffect v2.0", ("moss-sfx-v2",), pins.MOSS_LICENCE, 32),
)
IDS = tuple(spec.id for spec in SPECS)
BY_ID = {spec.id: spec for spec in SPECS}


def _path(options: Mapping[str, Any], key: str, default: Path) -> Path:
    value = options.get(key)
    return Path(value).expanduser() if isinstance(value, str) and value else default


@dataclass(frozen=True)
class Sa3Paths:
    mlx_dir: Path
    hf_home: Path
    python: Path

    @property
    def checkout(self) -> Path | None:
        """The clone ``mlx_dir`` lives in, when it has the upstream layout."""
        mlx = self.mlx_dir
        if mlx.name == "mlx" and mlx.parent.name == "optimized":
            return mlx.parent.parent
        return None

    @property
    def models(self) -> Path:
        return self.mlx_dir / "models" / "mlx"


@dataclass(frozen=True)
class MossPaths:
    code_dir: Path
    model_dir: Path
    python: Path

    @property
    def venv(self) -> Path:
        return self.python.parent.parent


def sa3_paths(options: Mapping[str, Any], prefix: Path) -> Sa3Paths:
    mlx = _path(options, "mlx_dir", prefix / "stable-audio-3" / "optimized" / "mlx")
    return Sa3Paths(
        mlx_dir=mlx,
        hf_home=_path(options, "hf_home", prefix / "hf"),
        python=_path(options, "python", mlx / ".venv" / "bin" / "python"),
    )


def moss_paths(options: Mapping[str, Any], prefix: Path) -> MossPaths:
    code = _path(options, "code_dir", prefix / "moss" / "src")
    return MossPaths(
        code_dir=code,
        model_dir=_path(options, "model_dir", prefix / "moss" / "MOSS-SoundEffect-v2.0"),
        python=_path(options, "python", code.parent / ".venv" / "bin" / "python"),
    )


Paths = Sa3Paths | MossPaths


def paths_for(spec: Spec, backends: Mapping[str, Any], prefix: Path) -> Paths:
    options = backends.get(spec.backend)
    table = options if isinstance(options, Mapping) else {}
    return moss_paths(table, prefix) if spec.id == "moss" else sa3_paths(table, prefix)


def unavailable_reason(spec: Spec, paths: Paths) -> str | None:
    """The backend's own answer, for the paths setup would use."""
    if isinstance(paths, MossPaths):
        moss = MossOptions(code_dir=paths.code_dir, model_dir=paths.model_dir, python=paths.python)
        return MossBackend(moss, state_dir=Path()).unavailable_reason()
    options = Sa3Options(mlx_dir=paths.mlx_dir, hf_home=paths.hf_home, python=paths.python)
    backend = Sa3MediumBackend if spec.id == "sa3-medium" else Sa3Backend
    return backend(options, state_dir=Path()).unavailable_reason()


def missing_downloads(spec: Spec, paths: Paths) -> dict[str, int]:
    """Bytes still to download, keyed so that what two components share is counted once."""
    if isinstance(paths, MossPaths):
        out: dict[str, int] = {}
        if not paths.python.exists():
            out[f"env:{paths.venv}"] = pins.MOSS_ENV_BYTES
        if any(not (paths.model_dir / name).exists() for name in MOSS_MODEL_FILES):
            have = dir_size(paths.model_dir)
            out[f"model:{paths.model_dir}"] = max(0, pins.MOSS_MODEL_BYTES - have)
        return out
    bundle = pins.SA3_BUNDLES[spec.id]
    files = {**pins.SA3_FILES[bundle], **pins.SA3_SHARED_FILES}
    out = {
        f"file:{paths.models / name}": size
        for name, size in files.items()
        if not (paths.models / name).exists()
    }
    if not paths.python.exists():
        out[f"env:{paths.python}"] = pins.SA3_ENV_BYTES
    return out


def config_table(spec: Spec, paths: Paths) -> dict[str, str]:
    """The ``[backends.<name>]`` keys the server needs for this component."""
    if isinstance(paths, MossPaths):
        return {"code_dir": str(paths.code_dir), "model_dir": str(paths.model_dir)}
    return {"mlx_dir": str(paths.mlx_dir), "hf_home": str(paths.hf_home)}

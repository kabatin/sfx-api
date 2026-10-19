"""Stable Audio 3 backends: the official MLX build (``optimized/mlx`` of
Stability-AI/stable-audio-3), run by :mod:`sfx_api.workers.sa3_worker`. One backend per model:
``sa3`` runs sm-sfx, ``sa3-medium`` runs medium; both use the same checkout and worker.

Facts this module relies on (measured on the target Mac, M5 32 GB, README "Measurements"):
- ``optimized/mlx/install.sh -y --download sm-sfx`` makes ``.venv`` and links the weights
  into ``models/mlx/`` (sm-sfx DiT, SAME-S decoder, T5Gemma: ~1.7 GB used); medium adds its
  DiT and the SAME-L decoder (~4.6 GB);
- 8 steps, guidance off (the models are distilled for it); 44.1 kHz stereo, trimmed to the
  requested length; the same seed gives the same clip;
- sm-sfx: models load in < 1 s; a clip takes 0.4-0.6 s once loaded, ~2.6 s for the first one.
  It breaks at fractional lengths, so the worker generates whole seconds and cuts;
- medium: models load in ~1 s (files in the OS cache); a clip takes 1.1 s (1-2 s) to 1.7 s
  (10 s) once loaded; peak footprint ~11 GB. Fractional lengths are fine (72 clips checked).
  Music: 10.3 s for 120 s, 15.8 s for 190 s, 32.8 s for 380 s; up to ~14 GB for the longest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from sfx_api.backends.engine import EngineConfig
from sfx_api.backends.resident import WORKERS_DIR, ResidentBackend, parse_options
from sfx_api.checks import Check, Level
from sfx_api.settings import Settings


@dataclass(frozen=True)
class Sa3Model:
    """One Stable Audio 3 model as the worker loads it, with its measured timings."""

    label: str
    dit: str
    decoder: str
    # what the worker loads (``models/mlx/`` of the checkout)
    model_files: tuple[str, ...]
    # generate the next whole second and cut (the model breaks at fractional lengths)
    whole_seconds: bool
    load_seconds: float
    clip_overhead_seconds: float
    seconds_per_audio_second: float


SM_SFX = Sa3Model(
    label="sm-sfx",
    dit="sm-sfx",
    decoder="same-s",
    model_files=("dit_sm-sfx_f16.npz", "same_s_decoder_f32.npz", "t5gemma_f16.npz"),
    whole_seconds=True,
    load_seconds=5.0,
    clip_overhead_seconds=1.0,
    seconds_per_audio_second=0.05,
)
MEDIUM = Sa3Model(
    label="medium",
    dit="medium",
    decoder="same-l",
    model_files=("dit_medium_f16.npz", "same_l_decoder_f32.npz", "t5gemma_f16.npz"),
    whole_seconds=False,
    # ~1 s with the files in the OS cache; reading 4.6 GB from disk was not measured
    load_seconds=15.0,
    clip_overhead_seconds=1.0,
    seconds_per_audio_second=0.075,
)
MODEL_FILES = SM_SFX.model_files


def _or(option: float | None, default: float) -> float:
    """An estimate option, or the model's measured default when it is unset (0 is a value)."""
    return default if option is None else option


class Sa3Options(BaseModel):
    """``[backends.sa3]`` (sm-sfx) or ``[backends.sa3-medium]`` in the config file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # the stable-audio-3 checkout's ``optimized/mlx`` directory (install.sh run in it)
    mlx_dir: Path | None = None
    python: Path | None = None  # default: <mlx_dir>/.venv/bin/python
    # HF_HOME the weights were downloaded into (models/mlx/ links point there)
    hf_home: Path | None = None
    # guidance used when a job has a negative prompt; 1.0 (the default) ignores negative
    # prompts and keeps the distilled model's own sound (3.0 is the checkout's example)
    negative_prompt_cfg: float = Field(default=1.0, ge=1.0, le=10.0)
    resident_idle_seconds: float = Field(default=300.0, ge=0.0, le=3600.0)
    start_timeout_s: float = Field(default=120.0, ge=10.0, le=3600.0)
    stop_grace_s: float = Field(default=10.0, ge=1.0, le=120.0)
    # estimates; unset means the model's measured defaults
    load_seconds: float | None = Field(default=None, gt=0.0, le=3600.0)
    clip_overhead_seconds: float | None = Field(default=None, ge=0.0, le=3600.0)
    seconds_per_audio_second: float | None = Field(default=None, gt=0.0, le=60.0)


class Sa3Backend(ResidentBackend):
    name: ClassVar[str] = "sa3"
    model: ClassVar[Sa3Model] = SM_SFX

    def __init__(
        self, options: Sa3Options | None = None, *, state_dir: Path, poll_seconds: float = 5.0
    ) -> None:
        self.options = options or Sa3Options()
        self.state_dir = state_dir
        self.poll_seconds = poll_seconds
        super().__init__(load_seconds=_or(self.options.load_seconds, self.model.load_seconds))

    @classmethod
    def from_settings(cls, settings: Settings) -> Sa3Backend:
        options = parse_options(Sa3Options, settings, cls.name)
        return cls(
            options,
            state_dir=settings.data_dir / "engine" / cls.name,
            poll_seconds=settings.coordination_poll_seconds,
        )

    @property
    def label(self) -> str:
        return f"Stable Audio 3 ({self.model.label})"

    # -- paths -----------------------------------------------------------------------
    @property
    def mlx_dir(self) -> Path | None:
        root = self.options.mlx_dir
        return root.expanduser() if root is not None else None

    @property
    def python_path(self) -> Path:
        if self.options.python is not None:
            return self.options.python.expanduser()
        return (self.mlx_dir or Path()) / ".venv" / "bin" / "python"

    def engine_config(self) -> EngineConfig:
        o, m = self.options, self.model
        env = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        if o.hf_home is not None:
            env["HF_HOME"] = str(o.hf_home.expanduser())
        args = (
            "--mlx-dir",
            str(self.mlx_dir or Path()),
            "--negative-cfg",
            str(o.negative_prompt_cfg),
            "--dit",
            m.dit,
            "--decoder",
            m.decoder,
        )
        return EngineConfig(
            name=self.name,
            python=self.python_path,
            script=WORKERS_DIR / "sa3_worker.py",
            args=(*args, "--whole-seconds") if m.whole_seconds else args,
            cwd=self.mlx_dir or Path(),
            state_dir=self.state_dir,
            idle_seconds=o.resident_idle_seconds,
            poll_seconds=self.poll_seconds,
            start_timeout_s=o.start_timeout_s,
            stop_grace_s=o.stop_grace_s,
            env=env,
        )

    @property
    def uses_negative_prompt(self) -> bool:
        return self.options.negative_prompt_cfg > 1.0

    def typical_seconds(self, duration_s: float) -> float:
        o, m = self.options, self.model
        overhead = _or(o.clip_overhead_seconds, m.clip_overhead_seconds)
        return overhead + _or(o.seconds_per_audio_second, m.seconds_per_audio_second) * duration_s

    # -- health ----------------------------------------------------------------------
    def unavailable_reason(self) -> str | None:
        root = self.mlx_dir
        if root is None:
            return f"not configured (set mlx_dir in [backends.{self.name}])"
        if not (root / "scripts" / "sa3_mlx.py").is_file():
            return "the Stable Audio 3 MLX checkout is missing"
        if not os.access(self.python_path, os.X_OK):
            return "its Python environment is missing (run install.sh)"
        models = root / "models" / "mlx"
        missing = [name for name in self.model.model_files if not (models / name).exists()]
        if missing:
            return f"model files missing: {', '.join(missing)}"
        return None

    def checks(self) -> list[Check]:
        reason = self.unavailable_reason()
        if reason is not None:
            return [Check(self.label, Level.FAIL, reason)]
        idle = self.options.resident_idle_seconds
        return [
            Check(self.label, Level.OK, f"{self.mlx_dir} (MLX)"),
            Check(
                f"{self.label} resident",
                Level.OK,
                f"models stay loaded {idle:.0f} s after a job" if idle else "off (load per job)",
            ),
        ]


class Sa3MediumBackend(Sa3Backend):
    name: ClassVar[str] = "sa3-medium"
    model: ClassVar[Sa3Model] = MEDIUM

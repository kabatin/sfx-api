"""Stable Audio 3 backend: the official MLX build (``optimized/mlx`` of
Stability-AI/stable-audio-3) with the sm-sfx model, run by
:mod:`sfx_api.workers.sa3_worker`.

Facts this module relies on (measured on the target Mac, M5 32 GB, README "Measurements"):
- ``optimized/mlx/install.sh -y --download sm-sfx`` makes ``.venv`` and links the weights
  into ``models/mlx/`` (sm-sfx DiT, SAME-S decoder, T5Gemma: ~1.7 GB used);
- 8 steps, guidance off (the model is distilled for it); 44.1 kHz stereo, trimmed to the
  requested length; the same seed gives the same clip;
- models load in < 1 s; a clip takes 0.4-0.6 s once loaded, ~2.6 s for the first one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from sfx_api.backends.engine import EngineConfig
from sfx_api.backends.resident import WORKERS_DIR, ResidentBackend, parse_options
from sfx_api.checks import Check, Level
from sfx_api.settings import Settings

ENGINE = "sa3"
# what the worker loads (``models/mlx/`` of the checkout)
MODEL_FILES = ("dit_sm-sfx_f16.npz", "same_s_decoder_f32.npz", "t5gemma_f16.npz")
LOAD_SECONDS = 5.0
CLIP_OVERHEAD_SECONDS = 1.0
SECONDS_PER_AUDIO_SECOND = 0.05


class Sa3Options(BaseModel):
    """``[backends.sa3]`` in the config file."""

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
    load_seconds: float = Field(default=LOAD_SECONDS, gt=0.0, le=3600.0)
    clip_overhead_seconds: float = Field(default=CLIP_OVERHEAD_SECONDS, ge=0.0, le=3600.0)
    seconds_per_audio_second: float = Field(default=SECONDS_PER_AUDIO_SECOND, gt=0.0, le=60.0)


class Sa3Backend(ResidentBackend):
    name: ClassVar[str] = ENGINE

    def __init__(
        self, options: Sa3Options | None = None, *, state_dir: Path, poll_seconds: float = 5.0
    ) -> None:
        self.options = options or Sa3Options()
        self.state_dir = state_dir
        self.poll_seconds = poll_seconds
        super().__init__(load_seconds=self.options.load_seconds)

    @classmethod
    def from_settings(cls, settings: Settings) -> Sa3Backend:
        options = parse_options(Sa3Options, settings, ENGINE)
        return cls(
            options,
            state_dir=settings.data_dir / "engine" / ENGINE,
            poll_seconds=settings.coordination_poll_seconds,
        )

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
        o = self.options
        env = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        if o.hf_home is not None:
            env["HF_HOME"] = str(o.hf_home.expanduser())
        return EngineConfig(
            name=ENGINE,
            python=self.python_path,
            script=WORKERS_DIR / "sa3_worker.py",
            args=(
                "--mlx-dir",
                str(self.mlx_dir or Path()),
                "--negative-cfg",
                str(o.negative_prompt_cfg),
            ),
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
        o = self.options
        return o.clip_overhead_seconds + o.seconds_per_audio_second * duration_s

    # -- health ----------------------------------------------------------------------
    def unavailable_reason(self) -> str | None:
        root = self.mlx_dir
        if root is None:
            return "not configured (set mlx_dir in [backends.sa3])"
        if not (root / "scripts" / "sa3_mlx.py").is_file():
            return "the Stable Audio 3 MLX checkout is missing"
        if not os.access(self.python_path, os.X_OK):
            return "its Python environment is missing (run install.sh)"
        missing = [name for name in MODEL_FILES if not (root / "models" / "mlx" / name).exists()]
        if missing:
            return f"model files missing: {', '.join(missing)}"
        return None

    def checks(self) -> list[Check]:
        reason = self.unavailable_reason()
        if reason is not None:
            return [Check("Stable Audio 3 (sm-sfx)", Level.FAIL, reason)]
        idle = self.options.resident_idle_seconds
        return [
            Check("Stable Audio 3 (sm-sfx)", Level.OK, f"{self.mlx_dir} (MLX)"),
            Check(
                "Stable Audio 3 resident",
                Level.OK,
                f"models stay loaded {idle:.0f} s after a job" if idle else "off (load per job)",
            ),
        ]

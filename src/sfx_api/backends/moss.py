"""MOSS-SoundEffect v2.0 backend: OpenMOSS's pipeline (``moss_soundeffect_v2`` in the
MOSS-TTS repository) with PyTorch on MPS, run by :mod:`sfx_api.workers.moss_worker`.

Facts this module relies on (README "Measurements" has the numbers):
- the model repository (OpenMOSS-Team/MOSS-SoundEffect-v2.0, 11.2 GB) is downloaded to a
  local directory: a 1.4 B DiT (fp32), Qwen3-1.7B text encoder, DAC VAE;
- output is 48 kHz mono; the pipeline always denoises a fixed 30 s latent and crops it, so a
  short clip costs as much as a long one (``steps * seconds_per_step``). A shorter latent
  (upstream's ``max_inference_seconds``) breaks the model: with 10 s, every clip held only
  ~1.5 s of sound followed by silence, at a different pitch;
- 50 steps (the upstream demo's default; upstream's API default is 100) measured the same as 100
  (spectrum and envelope) at half the time;
- guidance is on (cfg > 1), so negative prompts work. It must stay on: the guided step returns
  float32 latents, which the float32 VAE decode needs off CUDA.
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

ENGINE = "moss"
PACKAGE = "moss_soundeffect_v2"
MODEL_FILES = (
    "model_index.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/vae_128d_48k.pth",
    "text_encoder",
    "tokenizer",
)
# 15-16 s with the files in the OS cache, 28 s without
LOAD_SECONDS = 30.0
# one guided DiT step (two passes) on a 30 s latent, M5 32 GB, GPU otherwise idle:
# 86-88 s for 50 steps, 168 s for 100 (the rest is text encoding and the VAE decode)
SECONDS_PER_STEP = 1.62
CLIP_OVERHEAD_SECONDS = 6.0


class MossOptions(BaseModel):
    """``[backends.moss]`` in the config file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # the OpenMOSS/MOSS-TTS checkout (contains moss_soundeffect_v2/)
    code_dir: Path | None = None
    # the downloaded OpenMOSS-Team/MOSS-SoundEffect-v2.0 repository
    model_dir: Path | None = None
    python: Path | None = None  # default: <code_dir>/../.venv/bin/python
    device: str = Field(default="mps", pattern=r"^(mps|cpu)$")
    steps: int = Field(default=50, ge=1, le=200)
    cfg: float = Field(default=4.0, gt=1.0, le=10.0)
    resident_idle_seconds: float = Field(default=300.0, ge=0.0, le=3600.0)
    start_timeout_s: float = Field(default=300.0, ge=10.0, le=3600.0)
    stop_grace_s: float = Field(default=15.0, ge=1.0, le=120.0)
    load_seconds: float = Field(default=LOAD_SECONDS, gt=0.0, le=3600.0)
    clip_overhead_seconds: float = Field(default=CLIP_OVERHEAD_SECONDS, ge=0.0, le=3600.0)
    seconds_per_step: float = Field(default=SECONDS_PER_STEP, gt=0.0, le=600.0)


class MossBackend(ResidentBackend):
    name: ClassVar[str] = ENGINE

    def __init__(
        self, options: MossOptions | None = None, *, state_dir: Path, poll_seconds: float = 5.0
    ) -> None:
        self.options = options or MossOptions()
        self.state_dir = state_dir
        self.poll_seconds = poll_seconds
        super().__init__(load_seconds=self.options.load_seconds)

    @classmethod
    def from_settings(cls, settings: Settings) -> MossBackend:
        return cls(
            parse_options(MossOptions, settings, ENGINE),
            state_dir=settings.data_dir / "engine" / ENGINE,
            poll_seconds=settings.coordination_poll_seconds,
        )

    # -- paths -----------------------------------------------------------------------
    @property
    def code_dir(self) -> Path | None:
        path = self.options.code_dir
        return path.expanduser() if path is not None else None

    @property
    def model_dir(self) -> Path | None:
        path = self.options.model_dir
        return path.expanduser() if path is not None else None

    @property
    def python_path(self) -> Path:
        if self.options.python is not None:
            return self.options.python.expanduser()
        return (self.code_dir or Path()).parent / ".venv" / "bin" / "python"

    def engine_config(self) -> EngineConfig:
        o = self.options
        return EngineConfig(
            name=ENGINE,
            python=self.python_path,
            script=WORKERS_DIR / "moss_worker.py",
            args=(
                "--code-dir",
                str(self.code_dir or Path()),
                "--model-dir",
                str(self.model_dir or Path()),
                "--device",
                o.device,
                "--steps",
                str(o.steps),
                "--cfg",
                str(o.cfg),
            ),
            cwd=self.code_dir or Path(),
            state_dir=self.state_dir,
            idle_seconds=o.resident_idle_seconds,
            poll_seconds=self.poll_seconds,
            start_timeout_s=o.start_timeout_s,
            stop_grace_s=o.stop_grace_s,
            env={
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            },
        )

    @property
    def uses_negative_prompt(self) -> bool:
        return True  # guidance is always on

    def typical_seconds(self, duration_s: float) -> float:
        o = self.options
        return o.clip_overhead_seconds + o.seconds_per_step * o.steps

    # -- health ----------------------------------------------------------------------
    def unavailable_reason(self) -> str | None:
        code, model = self.code_dir, self.model_dir
        if code is None or model is None:
            return "not configured (set code_dir and model_dir in [backends.moss])"
        if not (code / PACKAGE / "pipeline_moss_soundeffect.py").is_file():
            return "the MOSS-TTS checkout is missing"
        if not os.access(self.python_path, os.X_OK):
            return "its Python environment is missing"
        missing = [name for name in MODEL_FILES if not (model / name).exists()]
        if missing:
            return f"model files missing: {', '.join(missing)}"
        return None

    def checks(self) -> list[Check]:
        reason = self.unavailable_reason()
        if reason is not None:
            return [Check("MOSS-SoundEffect v2.0", Level.FAIL, reason)]
        o = self.options
        return [
            Check(
                "MOSS-SoundEffect v2.0",
                Level.OK,
                f"{self.model_dir} ({o.device}, {o.steps} steps, cfg {o.cfg:g})",
            ),
            Check(
                "MOSS resident",
                Level.OK,
                f"models stay loaded {o.resident_idle_seconds:.0f} s after a job"
                if o.resident_idle_seconds
                else "off (load per job)",
            ),
        ]

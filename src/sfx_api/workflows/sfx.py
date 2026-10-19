"""The sound-effect workflows: an English description (and optionally what to avoid) to 1-4
short clips, delivered as 16-bit PCM WAV.

All workflows take the same params; they differ in engine, sample rate, channels and the tag
put in front of the prompt (the medium workflow sends "TrackType: SFX, " first). Clip i
uses ``seed + 2 * i``: Stable Audio 3 draws its noise from ``seed`` and its sampler from
``seed + 1``, so consecutive seeds would share randomness. Sending ``seed_used`` back with
``count: 1`` reproduces that clip. Every clip is read back (PCM WAV, the workflow's rate and
channels, non-empty) before the job succeeds; the result reports each clip's real length.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sfx_api.audio import AudioError, inspect_wav, wav_problem
from sfx_api.backends.base import GenerationRequest, GenerationResult
from sfx_api.workflows.base import Workflow, WorkflowFailedError, WorkflowOutput

OUTPUT_MEDIA_TYPE = "audio/wav"
MAX_PROMPT_CHARS = 500
MIN_DURATION_S = 0.5
MAX_DURATION_S = 10.0
DEFAULT_DURATION_S = 2.0
MAX_COUNT = 4
MAX_SEED = 2**32 - 1
SEED_STEP = 2

# C0 control characters and DEL; tab and newline are ordinary text.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _text(value: str, field: str) -> str:
    value = value.replace("\r\n", "\n")
    if _CONTROL.search(value):
        raise ValueError(f"{field} contains control characters")
    return value.strip()


class SfxParams(BaseModel):
    """POST body for every sound-effect workflow."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_CHARS,
        description="English description, e.g. 'tyre screech on wet asphalt, close, short'",
    )
    negative_prompt: str = Field(
        default="",
        max_length=MAX_PROMPT_CHARS,
        description="what the sound should not contain, e.g. 'music, speech' (see capabilities)",
    )
    duration_sec: float = Field(default=DEFAULT_DURATION_S, ge=MIN_DURATION_S, le=MAX_DURATION_S)
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED, description="clip i uses seed + 2i; random if omitted"
    )
    count: int = Field(default=1, ge=1, le=MAX_COUNT, description="clips, one after another")

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        value = _text(value, "prompt")
        if not value:
            raise ValueError("prompt is blank")
        return value

    @field_validator("negative_prompt")
    @classmethod
    def _negative(cls, value: str) -> str:
        return _text(value, "negative_prompt")

    @model_validator(mode="after")
    def _seed_range(self) -> SfxParams:
        if self.seed is not None and self.seed + SEED_STEP * (self.count - 1) > MAX_SEED:
            raise ValueError(f"seed + 2 * (count - 1) must not exceed {MAX_SEED}")
        return self


def clip_seeds(seed: int, count: int) -> tuple[int, ...]:
    return tuple(seed + SEED_STEP * index for index in range(count))


class SfxWorkflow(Workflow):
    """One engine's sound-effect workflow; the subclasses fix the engine and its output."""

    engine: ClassVar[str]
    sample_rate: ClassVar[int]
    channels: ClassVar[int]
    # how the engine treats negative_prompt (shown in capabilities)
    negative_prompt_note: ClassVar[str]
    # put in front of the client's prompt unless it already starts with a TrackType tag
    prompt_prefix: ClassVar[str] = ""
    output_media_type = OUTPUT_MEDIA_TYPE
    params_model = SfxParams

    def __init__(self, *, typical_seconds_2s: float, uses_negative_prompt: bool) -> None:
        self.typical_seconds_2s = typical_seconds_2s
        self.uses_negative_prompt = uses_negative_prompt

    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        p = cast(SfxParams, params)
        stored = p.model_dump()
        if p.seed is None:
            stored["seed"] = secrets.randbelow(MAX_SEED - SEED_STEP * (MAX_COUNT - 1) + 1)
        return stored

    def engine_prompt(self, prompt: str) -> str:
        """The prompt the engine is given."""
        if not self.prompt_prefix or prompt.lower().startswith("tracktype:"):
            return prompt
        return self.prompt_prefix + prompt

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        return GenerationRequest(
            job_id=job_id,
            engine=self.engine,
            prompt=self.engine_prompt(params["prompt"]),
            negative_prompt=params.get("negative_prompt", ""),
            duration_s=float(params["duration_sec"]),
            seeds=clip_seeds(params["seed"], params["count"]),
        )

    def finalize(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path, raw: GenerationResult
    ) -> WorkflowOutput:
        count = params["count"]
        if len(raw.outputs) != count:
            raise WorkflowFailedError(
                "invalid_output",
                f"the engine delivered {len(raw.outputs)} of {count} clips",
                retryable=True,
            )
        seeds = clip_seeds(params["seed"], count)
        clips: list[dict[str, Any]] = []
        finals: list[Path] = []
        for index, produced in enumerate(raw.outputs):
            try:
                info = inspect_wav(produced)
            except AudioError as exc:
                raise WorkflowFailedError("invalid_output", str(exc), retryable=True) from exc
            problem = wav_problem(info, sample_rate=self.sample_rate, channels=self.channels)
            if problem is not None:
                raise WorkflowFailedError("invalid_output", problem, retryable=True)
            clips.append(
                {
                    "index": index,
                    "media_type": self.output_media_type,
                    "duration_sec": round(info.seconds, 3),
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "seed_used": seeds[index],
                }
            )
            finals.append(job_dir / f"output-{index}.wav")
        for produced, final in zip(raw.outputs, finals, strict=True):
            produced.replace(final)
        negative = params.get("negative_prompt", "")
        return WorkflowOutput(
            files=tuple(finals),
            result={
                "outputs": clips,
                "seed_used": params["seed"],
                "details": {
                    "generation": {
                        "requested_duration_sec": params["duration_sec"],
                        "prompt_used": self.engine_prompt(params["prompt"]),
                        "negative_prompt_applied": bool(negative) and self.uses_negative_prompt,
                        "engine": raw.details,
                    }
                },
            },
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "output": {
                "media_type": self.output_media_type,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_format": "pcm_s16le",
            },
            "prompt": {
                "max_chars": MAX_PROMPT_CHARS,
                "language": "en",
                **({"prefix": self.prompt_prefix} if self.prompt_prefix else {}),
            },
            "negative_prompt": {
                "max_chars": MAX_PROMPT_CHARS,
                "applied": self.uses_negative_prompt,
                "note": self.negative_prompt_note,
            },
            "duration_sec": {
                "min": MIN_DURATION_S,
                "max": MAX_DURATION_S,
                "default": DEFAULT_DURATION_S,
            },
            "count": {"min": 1, "max": MAX_COUNT},
            "seed": {"min": 0, "max": MAX_SEED, "per_clip": "seed + 2i"},
            "typical_seconds": {"2s_clip": round(self.typical_seconds_2s, 1)},
        }

    def smoke_params(self) -> dict[str, Any]:
        return {
            "prompt": "single wooden knock, short",
            "negative_prompt": "",
            "duration_sec": 1.0,
            "seed": 1,
            "count": 1,
        }


class StableAudio3Workflow(SfxWorkflow):
    id = "stable-audio-3-sfx"
    description = "Stable Audio 3 (sm-sfx, MLX): sound effects as 16-bit 44.1 kHz stereo WAV"
    engine = "sa3"
    sample_rate = 44100
    channels = 2
    negative_prompt_note = (
        "accepted; used only when the server enables guidance for it "
        "(the distilled model runs without guidance by default)"
    )


class StableAudio3MediumWorkflow(SfxWorkflow):
    id = "stable-audio-3-medium-sfx"
    description = (
        "Stable Audio 3 (medium, MLX) with the TrackType: SFX tag: sound effects as 16-bit "
        "44.1 kHz stereo WAV"
    )
    engine = "sa3-medium"
    sample_rate = 44100
    channels = 2
    negative_prompt_note = StableAudio3Workflow.negative_prompt_note
    # Stable Audio 3's own prompt assistant tags sound effects this way; in a blind test on a
    # trailer's 9 sounds, medium with the tag was preferred for 8 (none fitted the 9th)
    prompt_prefix = "TrackType: SFX, "


class MossWorkflow(SfxWorkflow):
    id = "moss-sfx-v2"
    description = "MOSS-SoundEffect v2.0 (PyTorch, MPS): sound effects as 16-bit 48 kHz mono WAV"
    engine = "moss"
    sample_rate = 48000
    channels = 1
    negative_prompt_note = "used (classifier-free guidance is always on)"

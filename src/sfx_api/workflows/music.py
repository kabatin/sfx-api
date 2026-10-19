"""The music workflow: an English description of an instrumental track to 1-4 tracks, delivered as
16-bit 44.1 kHz stereo WAV.

It runs Stable Audio 3 medium, the engine of the medium sound-effect workflow, so no other model is
loaded and the two share one queue. The prompt gets the prompting guide's
"TrackType: Music, VocalType: Instrumental, " in front. The text encoder (T5Gemma) reads the first
256 tokens of the tagged prompt; a 741-character trailer brief came to 174. The model makes no
vocals. Seeds, read-back and results are as for the sound-effect workflows.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sfx_api.workflows.sfx import (
    MAX_PROMPT_CHARS,
    MIN_DURATION_S,
    SfxParams,
    SfxWorkflow,
    StableAudio3Workflow,
)

MAX_MUSIC_PROMPT_CHARS = 1000
MAX_MUSIC_DURATION_S = 190.0
DEFAULT_MUSIC_DURATION_S = 30.0


class MusicParams(SfxParams):
    """POST body for the music workflow: the sound-effect body with longer prompts and tracks."""

    prompt: str = Field(
        min_length=1,
        max_length=MAX_MUSIC_PROMPT_CHARS,
        description="English description of the track, e.g. 'lo-fi piano, calm, 80 BPM'",
    )
    negative_prompt: str = Field(
        default="",
        max_length=MAX_PROMPT_CHARS,
        description="what the track should not contain (ignored unless guidance is enabled)",
    )
    duration_sec: float = Field(
        default=DEFAULT_MUSIC_DURATION_S, ge=MIN_DURATION_S, le=MAX_MUSIC_DURATION_S
    )


class StableAudio3MusicWorkflow(SfxWorkflow):
    id = "stable-audio-3-medium-music"
    description = (
        "Stable Audio 3 (medium, MLX) with the TrackType: Music tag: instrumental music as 16-bit "
        "44.1 kHz stereo WAV"
    )
    engine = "sa3-medium"
    sample_rate = 44100
    channels = 2
    negative_prompt_note = StableAudio3Workflow.negative_prompt_note
    # the prompting guide's tag for instrumental music; in a blind test on four instrumental
    # briefs it was preferred to ACE-Step 1.5 for three
    prompt_prefix = "TrackType: Music, VocalType: Instrumental, "
    params_model = MusicParams
    max_prompt_chars = MAX_MUSIC_PROMPT_CHARS
    max_duration_s = MAX_MUSIC_DURATION_S
    default_duration_s = DEFAULT_MUSIC_DURATION_S

    def smoke_params(self) -> dict[str, Any]:
        return {
            "prompt": "calm solo piano, one short phrase",
            "negative_prompt": "",
            "duration_sec": 2.0,
            "seed": 1,
            "count": 1,
        }

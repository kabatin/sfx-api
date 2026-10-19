from pathlib import Path

import pytest
from pydantic import ValidationError

from sfx_api.backends.base import GenerationPlan, GenerationResult
from sfx_api.workflows.music import MusicParams, StableAudio3MusicWorkflow
from sfx_api.workflows.sfx import MAX_SEED, SfxParams
from tests.conftest import make_wav

PREFIX = "TrackType: Music, VocalType: Instrumental, "


def music() -> StableAudio3MusicWorkflow:
    return StableAudio3MusicWorkflow(typical_seconds_2s=1.2, uses_negative_prompt=False)


def store(params: dict, job_dir: Path) -> dict:
    return music().store_inputs(MusicParams.model_validate(params), job_dir)


def test_music_tags_the_prompt_once_on_the_medium_engine(tmp_path: Path) -> None:
    stored = store({"prompt": "lo-fi piano, calm", "seed": 7}, tmp_path)
    assert stored["duration_sec"] == 30.0 and stored["count"] == 1
    request = music().build_request("job_1", stored, tmp_path)
    assert request.engine == "sa3-medium"
    assert request.prompt == PREFIX + "lo-fi piano, calm"
    assert request.duration_s == 30.0 and request.seeds == (7,)
    tagged = "TrackType: Music, VocalType: Instrumental, ambient drone"
    own = store({"prompt": tagged, "seed": 7}, tmp_path)
    assert music().build_request("j", own, tmp_path).prompt == tagged
    clip = tmp_path / "raw-0.wav"
    make_wav(clip, seconds=1.0, rate=44100, channels=2)
    raw = GenerationResult(plan=GenerationPlan(1, 1.0), duration_s=1.0, outputs=(clip,))
    out = music().finalize("j", stored, tmp_path, raw)
    assert out.result["details"]["generation"]["prompt_used"] == PREFIX + "lo-fi piano, calm"


def test_music_takes_long_tracks_and_long_prompts() -> None:
    MusicParams.model_validate({"prompt": "x" * 1000, "duration_sec": 190, "count": 4})
    MusicParams.model_validate({"prompt": "x", "duration_sec": 10.5})
    MusicParams.model_validate({"prompt": "x", "duration_sec": 0.5})


@pytest.mark.parametrize(
    "params",
    [
        {"prompt": "x" * 1001},
        {"prompt": "   "},
        {"prompt": "a\x07b"},
        {"prompt": "x", "seed": MAX_SEED, "count": 2},
        {"prompt": "x", "negative_prompt": "n" * 501},
        {"prompt": "x", "duration_sec": 190.01},
        {"prompt": "x", "duration_sec": 0.4},
        {"prompt": "x", "count": 5},
        {"prompt": "x", "seed": MAX_SEED + 1},
        {"prompt": "x", "lyrics": "la la"},
    ],
)
def test_invalid_music_params(params: dict) -> None:
    with pytest.raises(ValidationError):
        MusicParams.model_validate(params)


def test_sound_effects_keep_their_limits() -> None:
    for params in ({"prompt": "x" * 501}, {"prompt": "x", "duration_sec": 10.01}):
        with pytest.raises(ValidationError):
            SfxParams.model_validate(params)


def test_music_schema_describes_music() -> None:
    schema = MusicParams.model_json_schema()["properties"]
    assert "track" in schema["negative_prompt"]["description"]
    assert schema["duration_sec"]["maximum"] == 190.0


def test_music_capabilities() -> None:
    caps = music().capabilities()
    assert caps["id"] == "stable-audio-3-medium-music"
    assert caps["output"] == {"media_type": "audio/wav", "sample_rate": 44100, "channels": 2,
                              "sample_format": "pcm_s16le"}  # fmt: skip
    assert caps["prompt"] == {"max_chars": 1000, "language": "en", "prefix": PREFIX}
    assert caps["duration_sec"] == {"min": 0.5, "max": 190.0, "default": 30.0}
    assert caps["negative_prompt"]["applied"] is False
    assert caps["count"] == {"min": 1, "max": 4}


def test_music_smoke_is_short_and_valid() -> None:
    params = music().params_model.model_validate(music().smoke_params())
    assert params.duration_sec <= 3.0

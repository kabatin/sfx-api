import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from sfx_api.backends.base import GenerationPlan, GenerationResult
from sfx_api.workflows.base import WorkflowFailedError
from sfx_api.workflows.sfx import (
    MAX_SEED,
    MossWorkflow,
    SfxParams,
    SfxWorkflow,
    StableAudio3Workflow,
    clip_seeds,
)
from tests.conftest import make_wav


def sa3(**kwargs: object) -> StableAudio3Workflow:
    return StableAudio3Workflow(
        typical_seconds_2s=float(kwargs.get("typical", 1.1)),
        uses_negative_prompt=bool(kwargs.get("negative", False)),
    )


def moss() -> MossWorkflow:
    return MossWorkflow(typical_seconds_2s=60.0, uses_negative_prompt=True)


def store(params: dict, job_dir: Path, workflow: SfxWorkflow | None = None) -> dict:
    flow = workflow or sa3()
    return flow.store_inputs(SfxParams.model_validate(params), job_dir)


def test_defaults_and_stored_params(tmp_path: Path) -> None:
    stored = store({"prompt": "  tyre screech, close  ", "seed": 7}, tmp_path)
    assert stored == {
        "prompt": "tyre screech, close",
        "negative_prompt": "",
        "duration_sec": 2.0,
        "seed": 7,
        "count": 1,
    }
    request = sa3().build_request("job_1", stored, tmp_path)
    assert (request.engine, request.prompt, request.negative_prompt) == (
        "sa3",
        "tyre screech, close",
        "",
    )
    assert request.duration_s == 2.0 and request.seeds == (7,) and request.count == 1


def test_clips_are_two_seeds_apart(tmp_path: Path) -> None:
    stored = store(
        {"prompt": "x", "seed": 1234, "count": 3, "negative_prompt": " music "}, tmp_path
    )
    request = moss().build_request("job_1", stored, tmp_path)
    assert request.engine == "moss" and request.seeds == (1234, 1236, 1238)
    assert request.negative_prompt == "music"
    assert clip_seeds(0, 4) == (0, 2, 4, 6)


def test_random_seed_leaves_room_for_every_clip(tmp_path: Path) -> None:
    for _ in range(50):
        stored = store({"prompt": "x", "count": 4}, tmp_path)
        assert stored["seed"] >= 0 and stored["seed"] + 6 <= MAX_SEED


def test_seed_range_includes_the_last_clip() -> None:
    SfxParams.model_validate({"prompt": "x", "seed": MAX_SEED - 6, "count": 4})
    with pytest.raises(ValidationError, match="seed \\+ 2"):
        SfxParams.model_validate({"prompt": "x", "seed": MAX_SEED - 5, "count": 4})
    SfxParams.model_validate({"prompt": "x", "seed": MAX_SEED, "count": 1})


@pytest.mark.parametrize(
    "params",
    [
        {"prompt": ""},
        {"prompt": "   "},
        {"prompt": "x" * 501},
        {"prompt": "a\x07b"},
        {"prompt": "x", "negative_prompt": "a\x00"},
        {"prompt": "x", "negative_prompt": "n" * 501},
        {"prompt": "x", "duration_sec": 0.4},
        {"prompt": "x", "duration_sec": 10.01},
        {"prompt": "x", "count": 0},
        {"prompt": "x", "count": 5},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "seed": MAX_SEED + 1},
        {"prompt": "x", "steps": 50},
        {"prompt": "x", "cfg": 3},
    ],
)
def test_invalid_params(params: dict) -> None:
    with pytest.raises(ValidationError):
        SfxParams.model_validate(params)


def test_limits_are_accepted() -> None:
    SfxParams.model_validate({"prompt": "x" * 500, "negative_prompt": "y" * 500})
    SfxParams.model_validate({"prompt": "x", "duration_sec": 0.5, "count": 4})
    SfxParams.model_validate({"prompt": "line one\n\tline two\r\n", "duration_sec": 10})


def raw(
    tmp_path: Path, count: int, *, rate: int, channels: int, seconds: float = 0.5
) -> GenerationResult:
    outputs = []
    for i in range(count):
        path = tmp_path / f"raw-{i}.wav"
        make_wav(path, seconds, rate=rate, channels=channels)
        outputs.append(path)
    return GenerationResult(
        plan=GenerationPlan(count, 1.0),
        duration_s=1.0,
        outputs=tuple(outputs),
        details={"load_seconds": 0.6, "clips": [{"index": 0, "seconds": 0.4}]},
    )


def test_finalize_reports_each_clip(tmp_path: Path) -> None:
    stored = store({"prompt": "x", "seed": 10, "count": 2, "duration_sec": 0.5}, tmp_path)
    out = sa3().finalize("job_1", stored, tmp_path, raw(tmp_path, 2, rate=44100, channels=2))
    assert [p.name for p in out.files] == ["output-0.wav", "output-1.wav"]
    assert all(p.is_file() for p in out.files) and not (tmp_path / "raw-0.wav").exists()
    assert out.result["outputs"] == [
        {"index": 0, "media_type": "audio/wav", "duration_sec": 0.5, "sample_rate": 44100,
         "channels": 2, "seed_used": 10},
        {"index": 1, "media_type": "audio/wav", "duration_sec": 0.5, "sample_rate": 44100,
         "channels": 2, "seed_used": 12},
    ]  # fmt: skip
    assert out.result["seed_used"] == 10
    generation = out.result["details"]["generation"]
    assert generation["requested_duration_sec"] == 0.5
    assert generation["negative_prompt_applied"] is False
    assert generation["engine"]["load_seconds"] == 0.6


def test_negative_prompt_applied_only_where_the_engine_uses_it(tmp_path: Path) -> None:
    stored = store({"prompt": "x", "seed": 1, "negative_prompt": "music"}, tmp_path)
    moss_out = moss().finalize("j", stored, tmp_path, raw(tmp_path, 1, rate=48000, channels=1))
    assert moss_out.result["details"]["generation"]["negative_prompt_applied"] is True
    assert moss_out.result["outputs"][0]["channels"] == 1
    off = sa3().finalize("j", stored, tmp_path, raw(tmp_path, 1, rate=44100, channels=2))
    assert off.result["details"]["generation"]["negative_prompt_applied"] is False
    on = sa3(negative=True).finalize(
        "j", stored, tmp_path, raw(tmp_path, 1, rate=44100, channels=2)
    )
    assert on.result["details"]["generation"]["negative_prompt_applied"] is True
    none = store({"prompt": "x", "seed": 1}, tmp_path)
    plain = moss().finalize("j", none, tmp_path, raw(tmp_path, 1, rate=48000, channels=1))
    assert plain.result["details"]["generation"]["negative_prompt_applied"] is False


@pytest.mark.parametrize(
    ("workflow", "rate", "channels", "message"),
    [
        (sa3(), 48000, 2, "sample rate is 48000 Hz, expected 44100 Hz"),
        (sa3(), 44100, 1, "1 channel(s), expected 2"),
        (moss(), 48000, 2, "2 channel(s), expected 1"),
        (moss(), 44100, 1, "sample rate is 44100 Hz, expected 48000 Hz"),
    ],
)
def test_finalize_refuses_the_wrong_format(
    tmp_path: Path, workflow: SfxWorkflow, rate: int, channels: int, message: str
) -> None:
    stored = store({"prompt": "x", "seed": 1}, tmp_path)
    with pytest.raises(WorkflowFailedError, match=re.escape(message)):
        workflow.finalize("j", stored, tmp_path, raw(tmp_path, 1, rate=rate, channels=channels))


def test_finalize_refuses_missing_empty_or_broken_clips(tmp_path: Path) -> None:
    stored = store({"prompt": "x", "seed": 1, "count": 2}, tmp_path)
    with pytest.raises(WorkflowFailedError, match="1 of 2 clips"):
        sa3().finalize("j", stored, tmp_path, raw(tmp_path, 1, rate=44100, channels=2))
    one = store({"prompt": "x", "seed": 1}, tmp_path)
    with pytest.raises(WorkflowFailedError, match="empty"):
        sa3().finalize("j", one, tmp_path, raw(tmp_path, 1, rate=44100, channels=2, seconds=0))
    broken = tmp_path / "raw-0.wav"
    broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    result = GenerationResult(plan=GenerationPlan(1, 1.0), duration_s=1.0, outputs=(broken,))
    with pytest.raises(WorkflowFailedError, match="not a readable PCM WAV"):
        sa3().finalize("j", one, tmp_path, result)


def test_capabilities() -> None:
    caps = sa3(typical=1.1).capabilities()
    assert caps == {
        "id": "stable-audio-3-sfx",
        "output": {"media_type": "audio/wav", "sample_rate": 44100, "channels": 2,
                   "sample_format": "pcm_s16le"},
        "prompt": {"max_chars": 500, "language": "en"},
        "negative_prompt": {"max_chars": 500, "applied": False,
                            "note": StableAudio3Workflow.negative_prompt_note},
        "duration_sec": {"min": 0.5, "max": 10.0, "default": 2.0},
        "count": {"min": 1, "max": 4},
        "seed": {"min": 0, "max": MAX_SEED, "per_clip": "seed + 2i"},
        "typical_seconds": {"2s_clip": 1.1},
    }  # fmt: skip
    m = moss().capabilities()
    assert m["id"] == "moss-sfx-v2" and m["output"]["sample_rate"] == 48000
    assert m["output"]["channels"] == 1 and m["negative_prompt"]["applied"] is True


def test_smoke_params_are_valid() -> None:
    for workflow in (sa3(), moss()):
        SfxParams.model_validate(workflow.smoke_params())

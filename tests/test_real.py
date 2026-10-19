"""Real engine runs on the Mac (opt-in: SFX_API_REAL_TESTS=1). They read the engines' locations
from the config file sfx-api uses (``$SFX_API_CONFIG`` or ~/.config/sfx-api/config.toml) and skip
an engine that is not available there. Stop the server first: they use the GPU."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from sfx_api.audio import inspect_wav, wav_problem
from sfx_api.backends import build_backend
from sfx_api.settings import load_settings
from sfx_api.workflows import build_registry, workflow_engine
from sfx_api.workflows.base import Workflow, WorkflowOutput

pytestmark = pytest.mark.real


def run(tmp_path: Path, workflow_id: str, params: dict[str, Any]) -> WorkflowOutput:
    settings = load_settings().with_overrides(data_dir=tmp_path / "data")
    backend = build_backend(settings)
    registry = build_registry(backend)
    workflow: Workflow | None = registry.get(workflow_id)
    assert workflow is not None
    engine = workflow_engine(workflow)
    assert engine is not None
    status = backend.engine_status(engine)
    if not status.available:
        pytest.skip(f"{workflow_id}: {status.reason}")
    stored = workflow.store_inputs(workflow.params_model.model_validate(params), tmp_path)
    request = workflow.build_request("job_real", stored, tmp_path)
    try:
        raw = backend.generate(request, tmp_path, timeout_s=1800, cancel=threading.Event())
        assert raw.ok, raw.describe_failure()
        return workflow.finalize("job_real", stored, tmp_path, raw)
    finally:
        backend.shutdown()


def assert_in_range(output: WorkflowOutput) -> None:
    """A clip that left [-1, 1] many times over is broadband noise once it is turned down. Loud
    pops legitimately peak near 1.8 (-6 dB); the broken clips needed -20 to -32 dB."""
    clips = output.result["details"]["generation"]["engine"]["clips"]
    assert all(c["gain_db"] > -12 for c in clips), clips


@pytest.mark.parametrize(
    "workflow_id", ["stable-audio-3-sfx", "stable-audio-3-medium-sfx", "moss-sfx-v2"]
)
def test_a_real_clip(tmp_path: Path, workflow_id: str) -> None:
    output = run(
        tmp_path,
        workflow_id,
        {"prompt": "single wooden knock, short", "duration_sec": 1.5, "seed": 5, "count": 2},
    )
    for index, path in enumerate(output.files):
        info = inspect_wav(path)
        rate = output.result["outputs"][index]["sample_rate"]
        channels = output.result["outputs"][index]["channels"]
        assert wav_problem(info, sample_rate=rate, channels=channels) is None
        assert info.seconds == pytest.approx(1.5, abs=0.05)
    assert [o["seed_used"] for o in output.result["outputs"]] == [5, 7]
    assert_in_range(output)


def test_sa3_fractional_length(tmp_path: Path) -> None:
    """Conditioned on 0.7 s itself, this prompt and seed peaked at 37 (gain -32 dB)."""
    prompt = "Short mechanical gear shift clunk of a metal shifter, close and dry."
    output = run(
        tmp_path, "stable-audio-3-sfx", {"prompt": prompt, "duration_sec": 0.7, "seed": 21}
    )
    assert inspect_wav(output.files[0]).seconds == pytest.approx(0.7, abs=0.01)
    assert_in_range(output)


def test_sa3_medium_sends_the_sfx_tag(tmp_path: Path) -> None:
    output = run(
        tmp_path,
        "stable-audio-3-medium-sfx",
        {"prompt": "Car speeding past at high velocity, doppler effect.", "duration_sec": 1.4,
         "seed": 29},
    )  # fmt: skip
    generation = output.result["details"]["generation"]
    assert generation["prompt_used"].startswith("TrackType: SFX, Car speeding past")
    assert inspect_wav(output.files[0]).seconds == pytest.approx(1.4, abs=0.01)
    assert_in_range(output)


def test_sa3_medium_music_runs_past_the_sound_effect_limit(tmp_path: Path) -> None:
    output = run(
        tmp_path,
        "stable-audio-3-medium-music",
        {"prompt": "lofi hip hop, mellow piano, 80 BPM", "duration_sec": 12.5, "seed": 203},
    )
    generation = output.result["details"]["generation"]
    assert generation["prompt_used"].startswith("TrackType: Music, VocalType: Instrumental, lofi")
    assert inspect_wav(output.files[0]).seconds == pytest.approx(12.5, abs=0.01)
    assert_in_range(output)

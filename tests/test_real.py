"""Real engine runs on the Mac (opt-in: SFX_API_REAL_TESTS=1). They read the engines' locations
from the config file sfx-api uses (``$SFX_API_CONFIG`` or ~/.config/sfx-api/config.toml) and skip
an engine that is not available there. Stop the server first: they use the GPU."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from sfx_api.audio import inspect_wav, wav_problem
from sfx_api.backends import build_backend
from sfx_api.settings import load_settings
from sfx_api.workflows import build_registry, workflow_engine
from sfx_api.workflows.base import Workflow

pytestmark = pytest.mark.real


@pytest.mark.parametrize("workflow_id", ["stable-audio-3-sfx", "moss-sfx-v2"])
def test_a_real_clip(tmp_path: Path, workflow_id: str) -> None:
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
    params = workflow.params_model.model_validate(
        {"prompt": "single wooden knock, short", "duration_sec": 1.5, "seed": 5, "count": 2}
    )
    stored = workflow.store_inputs(params, tmp_path)
    request = workflow.build_request("job_real", stored, tmp_path)
    try:
        raw = backend.generate(request, tmp_path, timeout_s=1800, cancel=threading.Event())
        assert raw.ok, raw.describe_failure()
        output = workflow.finalize("job_real", stored, tmp_path, raw)
    finally:
        backend.shutdown()
    for index, path in enumerate(output.files):
        info = inspect_wav(path)
        rate = output.result["outputs"][index]["sample_rate"]
        channels = output.result["outputs"][index]["channels"]
        assert wav_problem(info, sample_rate=rate, channels=channels) is None
        assert info.seconds == pytest.approx(1.5, abs=0.05)
    assert [o["seed_used"] for o in output.result["outputs"]] == [5, 7]

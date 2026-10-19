import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfx_api.api.app import create_app
from sfx_api.backends.base import EngineStatus
from sfx_api.jobs.queue import JobQueue
from sfx_api.jobs.store import JobStore
from sfx_api.workflows.base import Workflow, WorkflowRegistry
from sfx_api.workflows.music import StableAudio3MusicWorkflow
from sfx_api.workflows.sfx import MossWorkflow, StableAudio3Workflow
from tests.fakes import EchoWorkflow, FakeSoundBackend

TOKEN = "test-token-0123456789-abcdefghijklmnop"
SFX = "/v1/workflows/stable-audio-3-sfx/jobs"
MOSS = "/v1/workflows/moss-sfx-v2/jobs"
MUSIC = "/v1/workflows/stable-audio-3-medium-music/jobs"


def make_client(
    tmp_path: Path,
    backend: FakeSoundBackend,
    token: str | None = None,
    max_body: int = 1 << 20,
    waits_for: list[str] | None = None,
    extra_workflows: tuple[Workflow, ...] = (),
) -> TestClient:
    registry = WorkflowRegistry(
        [
            EchoWorkflow(),
            StableAudio3Workflow(typical_seconds_2s=1.1, uses_negative_prompt=False),
            MossWorkflow(typical_seconds_2s=60.0, uses_negative_prompt=True),
            *extra_workflows,
        ]
    )
    queue = JobQueue(
        JobStore(tmp_path), registry, backend, max_waiting=1, timeout_factor=2, retention_days=7
    )
    app = create_app(
        queue,
        registry,
        token=token,
        max_body_bytes=max_body,
        waits_for=waits_for,
        engine_status=backend.engine_status,
    )
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def backend() -> FakeSoundBackend:
    return FakeSoundBackend()


@pytest.fixture
def client(tmp_path: Path, backend: FakeSoundBackend) -> Iterator[TestClient]:
    with make_client(tmp_path, backend) as c:
        yield c
    backend.release.set()


def poll(client: TestClient, job_id: str, want: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job never reached {want}")


def test_health_matches_vpipe_api_shape(client: TestClient) -> None:
    expected = {
        "status": "ok",
        "version": "0.1.0",
        "running": 0,
        "waiting": 0,
        "max_waiting": 1,
        "waiting_for_peer": 0,
        "workflows": {
            "stable-audio-3-sfx": {"available": True, "loaded": False, "reason": None},
            "moss-sfx-v2": {"available": True, "loaded": False, "reason": None},
        },
    }
    assert client.get("/v1/health").json() == expected
    assert client.get("/healthz").json() == expected


def test_health_reports_unavailable_and_loaded_workflows(
    client: TestClient, backend: FakeSoundBackend
) -> None:
    backend.statuses = {
        "sa3": EngineStatus(available=True, loaded=True),
        "moss": EngineStatus(available=False, reason="model files missing: vae/vae_128d_48k.pth"),
    }
    flows = client.get("/v1/health").json()["workflows"]
    assert flows["stable-audio-3-sfx"] == {"available": True, "loaded": True, "reason": None}
    assert flows["moss-sfx-v2"]["available"] is False
    refused = client.post(MOSS, json={"prompt": "boom"})
    assert refused.status_code == 503
    assert refused.json()["error"] == {
        "code": "workflow_unavailable",
        "message": "moss-sfx-v2 cannot run on this server: model files missing: "
        "vae/vae_128d_48k.pth",
        "retryable": False,
        "details": None,
    }
    assert client.post(SFX, json={"prompt": "boom"}).status_code == 202


def test_capabilities_match_the_contract(tmp_path: Path, backend: FakeSoundBackend) -> None:
    with make_client(tmp_path, backend, waits_for=["vpipe-api"]) as c:
        body = c.get("/v1/capabilities").json()
    assert body["version"] == "0.1.0"
    assert body["queue"] == {"max_running": 1, "max_waiting": 1}
    assert body["coordination"] == {"waits_for": ["vpipe-api"]}
    flows = {w["id"]: w for w in body["workflows"]}
    assert flows["stable-audio-3-sfx"]["output"] == {
        "media_type": "audio/wav",
        "sample_rate": 44100,
        "channels": 2,
        "sample_format": "pcm_s16le",
    }
    assert flows["moss-sfx-v2"]["output"]["sample_rate"] == 48000
    assert flows["moss-sfx-v2"]["output"]["channels"] == 1
    assert flows["stable-audio-3-sfx"]["duration_sec"] == {"min": 0.5, "max": 10.0, "default": 2.0}
    text = str(body).lower()
    for secret in ("safetensors", "npz", "steps", "checkpoint", "mlx", "qwen", "/users"):
        assert secret not in text


def test_no_coordination_is_an_empty_list(client: TestClient) -> None:
    assert client.get("/v1/capabilities").json()["coordination"] == {"waits_for": []}


def test_workflows_and_openapi(client: TestClient) -> None:
    flows = {w["id"]: w for w in client.get("/v1/workflows").json()["workflows"]}
    assert set(flows) == {"echo", "stable-audio-3-sfx", "moss-sfx-v2"}
    for flow in ("stable-audio-3-sfx", "moss-sfx-v2"):
        props = flows[flow]["params_schema"]["properties"]
        assert set(props) == {"prompt", "negative_prompt", "duration_sec", "seed", "count"}
        assert flows[flow]["output_media_type"] == "audio/wav"
    spec = client.get("/openapi.json").json()
    route = spec["paths"][SFX]["post"]
    schema_ref = route["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref.endswith("SfxParams")
    assert {"422", "429"} <= set(route["responses"])
    assert client.get("/docs").status_code == 200


def test_a_sound_job_end_to_end(tmp_path: Path) -> None:
    backend = FakeSoundBackend(wav=True, seconds=2.5, rate=44100, channels=2)
    with make_client(tmp_path, backend) as c:
        body = {
            "prompt": "tyre screech on wet asphalt, close, short",
            "negative_prompt": "music, speech",
            "duration_sec": 2.5,
            "count": 2,
            "seed": 1234,
        }
        job = c.post(SFX, json=body, headers={"Idempotency-Key": "k-1"})
        assert job.status_code == 202
        again = c.post(SFX, json=body, headers={"Idempotency-Key": "k-1"})
        assert again.status_code == 200 and again.json()["id"] == job.json()["id"]
        done = poll(c, job.json()["id"], "succeeded")
        assert done["result"]["seed_used"] == 1234
        assert done["result"]["outputs"] == [
            {"index": 0, "media_type": "audio/wav", "duration_sec": 2.5, "sample_rate": 44100,
             "channels": 2, "seed_used": 1234},
            {"index": 1, "media_type": "audio/wav", "duration_sec": 2.5, "sample_rate": 44100,
             "channels": 2, "seed_used": 1236},
        ]  # fmt: skip
        call = backend.calls[0]
        assert call.engine == "sa3" and call.seeds == (1234, 1236)
        assert call.negative_prompt == "music, speech" and call.duration_s == 2.5
        wav = c.get(f"/v1/jobs/{job.json()['id']}/output?index=1")
        assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
        assert wav.content[:4] == b"RIFF"


def test_full_job_flow(client: TestClient) -> None:
    accepted = client.post("/v1/workflows/echo/jobs", json={"text": "hi"})
    assert accepted.status_code == 202
    job = accepted.json()
    assert set(job) == {"id", "workflow", "status", "created_at"}
    assert (job["workflow"], job["status"]) == ("echo", "queued")
    done = poll(client, job["id"], "succeeded")
    assert done["result"] == {"length": 2} and done["error"] is None
    assert done["progress"] == 1.0 and done["cancel_requested"] is False
    assert done["waiting_for"] is None and done["queue_position"] is None
    assert {
        "queue_seconds",
        "coordination_seconds",
        "backend_seconds",
        "postprocess_seconds",
        "total_seconds",
    } <= set(done["timings"])
    assert done["timings"]["coordination_seconds"] == 0.0
    out = client.get(f"/v1/jobs/{job['id']}/output")
    assert out.status_code == 200 and out.text == "HI"
    assert out.headers["content-type"].startswith("text/plain")
    assert client.get(f"/v1/jobs/{job['id']}/output?index=0").text == "HI"
    missing = client.get(f"/v1/jobs/{job['id']}/output?index=1")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get(f"/v1/jobs/{job['id']}/output?index=-1").status_code == 404
    assert client.get(f"/v1/jobs/{job['id']}/output?index=x").status_code == 422
    assert client.delete(f"/v1/jobs/{job['id']}").json()["error"]["code"] == "conflict"


def test_failed_job_over_http(client: TestClient, backend: FakeSoundBackend) -> None:
    backend.mode = "fail"
    job = client.post("/v1/workflows/echo/jobs", json={"text": "x"}).json()
    body = poll(client, job["id"], "failed")
    assert body["error"]["code"] == "generation_failed" and body["error"]["retryable"]
    assert body["result"] is None
    # 409 before the job succeeded, whatever the index
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409
    assert client.get(f"/v1/jobs/{job['id']}/output?index=7").status_code == 409


def test_errors_use_the_envelope(client: TestClient) -> None:
    missing = client.get("/v1/jobs/job_nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get("/v1/jobs/job_nope/output").status_code == 404
    assert client.delete("/v1/jobs/job_nope").status_code == 404
    assert client.post("/v1/workflows/nope/jobs", json={}).status_code == 404
    rejected = client.post("/v1/workflows/echo/jobs", json={"text": "x", "reject": True})
    assert rejected.status_code == 422 and "rejected" in rejected.json()["error"]["message"]
    assert rejected.json()["error"]["details"] == [{"loc": ["body", "reject"], "msg": "no"}]
    assert client.get("/v1/nothing-here").json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x", "steps": 8},
        {"prompt": "x", "cfg": 3.0},
        {"prompt": "x", "model": "sm-sfx"},
        {"prompt": "x", "output_path": "/tmp/x"},
        {"prompt": " "},
        {"prompt": ""},
        {"prompt": "x" * 501},
        {"prompt": "x\x00y"},
        {"negative_prompt": "music"},
        {"prompt": "x", "negative_prompt": "n" * 501},
        {"prompt": "x", "negative_prompt": "a\x07b"},
        {"prompt": "x", "duration_sec": 0.49},
        {"prompt": "x", "duration_sec": 10.1},
        {"prompt": "x", "duration_sec": "long"},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "seed": 2**32},
        {"prompt": "x", "seed": 2**32 - 4, "count": 3},
        {"prompt": "x", "count": 0},
        {"prompt": "x", "count": 5},
    ],
)
def test_request_validation(client: TestClient, body: dict) -> None:
    response = client.post(SFX, json=body)
    error = response.json()["error"]
    assert response.status_code == 422 and error["code"] == "invalid_params"
    assert all(set(d) == {"loc", "msg"} for d in error["details"] or [])


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x"},
        {"prompt": "x", "duration_sec": 0.5},
        {"prompt": "x", "duration_sec": 10},
        {"prompt": "x", "duration_sec": 2.37},
        {"prompt": "x", "negative_prompt": ""},
        {"prompt": "x", "seed": 2**32 - 7, "count": 4},
        {"prompt": "x" * 500, "negative_prompt": "y" * 500, "count": 4},
    ],
)
def test_bounds_are_accepted(client: TestClient, body: dict) -> None:
    assert client.post(SFX, json=body).status_code == 202


@pytest.mark.parametrize(
    ("route", "body", "status"),
    [
        (MUSIC, {"prompt": "lo-fi piano, calm", "duration_sec": 190}, 202),
        (MUSIC, {"prompt": "x" * 1000}, 202),
        (MUSIC, {"prompt": "x", "duration_sec": 190.01}, 422),
        (MUSIC, {"prompt": "x" * 1001}, 422),
        (SFX, {"prompt": "x", "duration_sec": 10.01}, 422),
        (SFX, {"prompt": "x" * 501}, 422),
    ],
)
def test_each_workflow_validates_with_its_own_limits(
    tmp_path: Path, backend: FakeSoundBackend, route: str, body: dict, status: int
) -> None:
    music = StableAudio3MusicWorkflow(typical_seconds_2s=1.2, uses_negative_prompt=False)
    with make_client(tmp_path, backend, extra_workflows=(music,)) as client:
        assert client.post(route, json=body).status_code == status
    backend.release.set()


def test_busy_returns_429_with_retry_after(client: TestClient, backend: FakeSoundBackend) -> None:
    backend.block = True
    first = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    assert backend.started.wait(2)
    second = client.post("/v1/workflows/echo/jobs", json={"text": "b"}).json()
    running = client.get(f"/v1/jobs/{first['id']}").json()
    queued = client.get(f"/v1/jobs/{second['id']}").json()
    assert (running["status"], running["queue_position"]) == ("running", None)
    assert (queued["status"], queued["queue_position"]) == ("queued", 1)
    assert client.get("/v1/health").json()["running"] == 1
    busy = client.post("/v1/workflows/echo/jobs", json={"text": "c"})
    assert busy.status_code == 429
    assert busy.json()["error"] == {
        "code": "busy",
        "message": "the GPU slot and the waiting queue are full",
        "retryable": True,
        "details": None,
    }
    assert int(busy.headers["retry-after"]) >= 30
    assert client.get(f"/v1/jobs/{first['id']}/output").status_code == 409
    assert client.delete(f"/v1/jobs/{second['id']}").json()["status"] == "canceled"
    stopped = client.delete(f"/v1/jobs/{first['id']}")
    assert stopped.status_code == 200 and stopped.json()["status"] == "canceled"


def test_cancel_of_a_slow_stop_is_202(
    client: TestClient, backend: FakeSoundBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sfx_api.jobs.queue.CANCEL_WAIT_S", 0.2)
    backend.mode = "ignore_cancel"
    backend.block = True
    job = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    assert backend.started.wait(2)
    pending = client.delete(f"/v1/jobs/{job['id']}")
    assert pending.status_code == 202
    assert pending.json()["status"] == "running" and pending.json()["cancel_requested"]
    backend.release.set()
    final = poll(client, job["id"], "canceled")
    assert final["result"] is None
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409


def test_bearer_auth_everywhere(tmp_path: Path, backend: FakeSoundBackend) -> None:
    with make_client(tmp_path, backend, token=TOKEN) as c:
        for path in ("/healthz", "/v1/health", "/v1/capabilities", "/openapi.json", "/docs"):
            denied = c.get(path)
            assert denied.status_code == 401 and denied.headers["www-authenticate"] == "Bearer"
            assert denied.json()["error"]["code"] == "unauthorized"
        assert c.post(SFX, json={}).status_code == 401
        assert c.get("/v1/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = c.get("/v1/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200


def test_body_limits(tmp_path: Path, backend: FakeSoundBackend) -> None:
    with make_client(tmp_path, backend, max_body=100) as c:
        big = c.post("/v1/workflows/echo/jobs", json={"text": "x" * 500})
        assert big.status_code == 413 and big.json()["error"]["code"] == "payload_too_large"
        chunked = c.post(
            "/v1/workflows/echo/jobs",
            content=iter([b'{"text":"x"}']),
            headers={"Content-Type": "application/json"},
        )
        assert chunked.status_code == 411
        assert chunked.json()["error"]["code"] == "length_required"

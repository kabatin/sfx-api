"""The engine backends (Stable Audio 3, MOSS), their shared job loop and the router.

Generation runs the stand-in worker (tests/fake_worker.py) through the shared loop; the real
engines' configuration, health and worker arguments are checked without running them."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest

from sfx_api.backends import build_backend
from sfx_api.backends.base import GenerationRequest
from sfx_api.backends.engine import EngineConfig
from sfx_api.backends.moss import MossBackend, MossOptions
from sfx_api.backends.resident import WORKERS_DIR, ResidentBackend
from sfx_api.backends.router import EngineRouter
from sfx_api.backends.sa3 import MODEL_FILES, Sa3Backend, Sa3Options
from sfx_api.checks import Check, Level
from sfx_api.settings import SettingsError, load_settings

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "ok")


class FakeEngineBackend(ResidentBackend):
    """The shared job loop with the stand-in worker as its engine."""

    name: ClassVar[str] = "fake"

    def __init__(self, tmp_path: Path, *, reason: str | None = None) -> None:
        self.tmp_path = tmp_path
        self.reason = reason
        super().__init__(load_seconds=7.0)

    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            name="fake",
            python=Path(sys.executable),
            script=FAKE_WORKER,
            args=(),
            cwd=self.tmp_path,
            state_dir=self.tmp_path / "state",
            idle_seconds=60.0,
            poll_seconds=0.05,
            start_timeout_s=30.0,
            stop_grace_s=2.0,
        )

    def unavailable_reason(self) -> str | None:
        return self.reason

    def checks(self) -> list[Check]:
        return [Check("fake", Level.OK, "ok")]

    def typical_seconds(self, duration_s: float) -> float:
        return 1.0 + duration_s


def request(**overrides: object) -> GenerationRequest:
    values: dict[str, object] = {
        "job_id": "job_1",
        "engine": "fake",
        "prompt": "whoosh",
        "negative_prompt": "",
        "duration_s": 0.5,
        "seeds": (41, 43),
    }
    return GenerationRequest(**(values | overrides))  # type: ignore[arg-type]


def test_every_clip_gets_its_seed(tmp_path: Path) -> None:
    backend = FakeEngineBackend(tmp_path)
    try:
        assert backend.plan(request()).estimate_seconds == 7.0 + 2 * 1.5  # load included
        progress: list[float] = []
        result = backend.generate(
            request(),
            tmp_path / "run",
            timeout_s=60,
            cancel=threading.Event(),
            on_progress=progress.append,
        )
        assert result.ok, result.failure
        assert [p.name for p in result.outputs] == ["raw-0.wav", "raw-1.wav"]
        seeds = [
            json.loads(p.with_name(p.name + ".params.json").read_text())["seed"]
            for p in result.outputs
        ]
        assert seeds == [41, 43]
        assert result.details["load_seconds"] is not None
        assert [c["gain_db"] for c in result.details["clips"]] == [-1.0, -1.0]
        assert progress[0] == pytest.approx(0.05) and progress[-1] == pytest.approx(0.995)
        assert backend.engine_status("fake").loaded  # resident after a good job
        assert backend.plan(request()).estimate_seconds == 2 * 1.5  # no load now
    finally:
        backend.shutdown()


def test_a_failed_clip_discards_the_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "gen_error")
    backend = FakeEngineBackend(tmp_path)
    try:
        result = backend.generate(
            request(), tmp_path / "run", timeout_s=60, cancel=threading.Event()
        )
        assert not result.ok and "out of memory" in (result.failure or "")
        assert not backend.engine.running()
        monkeypatch.setenv("FAKE_WORKER_MODE", "load_fail")
        failed = backend.generate(
            request(), tmp_path / "run2", timeout_s=60, cancel=threading.Event()
        )
        assert failed.failure_code == "engine_failed" and "checkpoint missing" in (
            failed.failure or ""
        )
    finally:
        backend.shutdown()


def test_cancel_before_the_load_finishes(tmp_path: Path) -> None:
    backend = FakeEngineBackend(tmp_path)
    cancel = threading.Event()
    cancel.set()
    result = backend.generate(request(), tmp_path / "run", timeout_s=60, cancel=cancel)
    assert result.canceled and not result.ok
    backend.shutdown()


def test_an_unavailable_engine_never_starts(tmp_path: Path) -> None:
    backend = FakeEngineBackend(tmp_path, reason="model files missing: x")
    result = backend.generate(request(), tmp_path / "run", timeout_s=60, cancel=threading.Event())
    assert (
        result.failure_code == "backend_unavailable" and result.failure == "model files missing: x"
    )
    status = backend.engine_status("fake")
    assert not status.available and status.reason == "model files missing: x"
    assert not backend.engine.running()


# -- Stable Audio 3 ------------------------------------------------------------------------------


def sa3_checkout(tmp_path: Path, models: tuple[str, ...] = MODEL_FILES) -> Path:
    root = tmp_path / "stable-audio-3" / "optimized" / "mlx"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "sa3_mlx.py").write_text("")
    (root / ".venv" / "bin").mkdir(parents=True)
    python = root / ".venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    (root / "models" / "mlx").mkdir(parents=True)
    for name in models:
        (root / "models" / "mlx" / name).write_bytes(b"x")
    return root


def test_sa3_worker_arguments_and_env(tmp_path: Path) -> None:
    root = sa3_checkout(tmp_path)
    backend = Sa3Backend(
        Sa3Options(mlx_dir=root, hf_home=tmp_path / "hf", negative_prompt_cfg=3.0),
        state_dir=tmp_path / "engine",
    )
    config = backend.engine.config
    assert config.script == WORKERS_DIR / "sa3_worker.py" and config.script.is_file()
    assert config.python == root / ".venv" / "bin" / "python"
    assert config.args == ("--mlx-dir", str(root), "--negative-cfg", "3.0")
    assert config.cwd == root and config.state_dir == tmp_path / "engine"
    assert config.env["HF_HUB_OFFLINE"] == "1" and config.env["HF_HOME"] == str(tmp_path / "hf")
    assert backend.uses_negative_prompt
    assert not Sa3Backend(Sa3Options(mlx_dir=root), state_dir=tmp_path).uses_negative_prompt


def test_sa3_health(tmp_path: Path) -> None:
    assert Sa3Backend(state_dir=tmp_path).unavailable_reason() == (
        "not configured (set mlx_dir in [backends.sa3])"
    )
    root = sa3_checkout(tmp_path)
    backend = Sa3Backend(Sa3Options(mlx_dir=root), state_dir=tmp_path)
    assert backend.unavailable_reason() is None
    assert [c.level for c in backend.health()] == [Level.OK, Level.OK]
    (root / "models" / "mlx" / "t5gemma_f16.npz").unlink()
    assert backend.unavailable_reason() == "model files missing: t5gemma_f16.npz"
    assert backend.health()[0].level is Level.FAIL
    (root / ".venv" / "bin" / "python").unlink()
    assert "Python environment" in (backend.unavailable_reason() or "")
    other = Sa3Backend(Sa3Options(mlx_dir=tmp_path / "nowhere"), state_dir=tmp_path)
    assert other.unavailable_reason() == "the Stable Audio 3 MLX checkout is missing"


def test_sa3_estimates(tmp_path: Path) -> None:
    backend = Sa3Backend(Sa3Options(mlx_dir=sa3_checkout(tmp_path)), state_dir=tmp_path)
    assert backend.typical_seconds(10.0) == pytest.approx(1.0 + 0.5)
    plan = backend.plan(request(engine="sa3", duration_s=2.0, seeds=(1, 3, 5)))
    assert plan.count == 3 and plan.estimate_seconds == pytest.approx(5.0 + 3 * 1.1)


# -- MOSS ------------------------------------------------------------------------------------------


def moss_install(tmp_path: Path) -> tuple[Path, Path]:
    code = tmp_path / "moss" / "src"
    (code / "moss_soundeffect_v2").mkdir(parents=True)
    (code / "moss_soundeffect_v2" / "pipeline_moss_soundeffect.py").write_text("")
    (tmp_path / "moss" / ".venv" / "bin").mkdir(parents=True)
    python = tmp_path / "moss" / ".venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    model = tmp_path / "MOSS-SoundEffect-v2.0"
    for name in ("transformer", "vae", "text_encoder", "tokenizer"):
        (model / name).mkdir(parents=True)
    (model / "model_index.json").write_text("{}")
    (model / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    (model / "vae" / "vae_128d_48k.pth").write_bytes(b"x")
    return code, model


def test_moss_worker_arguments(tmp_path: Path) -> None:
    code, model = moss_install(tmp_path)
    backend = MossBackend(
        MossOptions(code_dir=code, model_dir=model, steps=30),
        state_dir=tmp_path / "engine",
    )
    config = backend.engine.config
    assert config.script == WORKERS_DIR / "moss_worker.py" and config.script.is_file()
    assert config.python == tmp_path / "moss" / ".venv" / "bin" / "python"
    assert config.args == (
        "--code-dir", str(code), "--model-dir", str(model), "--device", "mps",
        "--steps", "30", "--cfg", "4.0",
    )  # fmt: skip
    assert config.env["HF_HUB_OFFLINE"] == "1"
    assert backend.uses_negative_prompt


def test_moss_estimates_follow_steps(tmp_path: Path) -> None:
    code, model = moss_install(tmp_path)

    def backend(**options: object) -> MossBackend:
        opts = MossOptions.model_validate({"code_dir": code, "model_dir": model} | options)
        return MossBackend(opts, state_dir=tmp_path)

    assert backend().typical_seconds(2.0) == pytest.approx(5.0 + 2.7 * 50)
    assert backend().typical_seconds(10.0) == backend().typical_seconds(0.5)  # fixed latent
    slow = backend(steps=100)
    assert slow.typical_seconds(2.0) == pytest.approx(5.0 + 270.0)
    plan = slow.plan(request(engine="moss", seeds=(1, 3)))
    assert plan.estimate_seconds == pytest.approx(40.0 + 2 * 275.0)


@pytest.mark.parametrize(
    "options", [{"cfg": 1.0}, {"steps": 0}, {"latent_seconds": 10}, {"device": "cuda"}]
)
def test_moss_refuses_settings_that_would_fail_jobs(options: dict) -> None:
    with pytest.raises(ValueError):
        MossOptions.model_validate(options)


def test_moss_health(tmp_path: Path) -> None:
    assert "not configured" in (MossBackend(state_dir=tmp_path).unavailable_reason() or "")
    code, model = moss_install(tmp_path)
    backend = MossBackend(MossOptions(code_dir=code, model_dir=model), state_dir=tmp_path)
    assert backend.unavailable_reason() is None and backend.health()[0].level is Level.OK
    (model / "vae" / "vae_128d_48k.pth").unlink()
    assert backend.unavailable_reason() == "model files missing: vae/vae_128d_48k.pth"
    gone = MossBackend(MossOptions(code_dir=tmp_path / "x", model_dir=model), state_dir=tmp_path)
    assert gone.unavailable_reason() == "the MOSS-TTS checkout is missing"


# -- router and settings --------------------------------------------------------------------------


def test_the_router_hands_each_request_to_its_engine(tmp_path: Path) -> None:
    fake = FakeEngineBackend(tmp_path)
    off = FakeEngineBackend(tmp_path / "b", reason="not installed")
    router = EngineRouter({"fake": fake, "off": off})
    try:
        assert router.engine_status("off").reason == "not installed"
        assert router.engine_status("nope").available is False
        assert router.typical_seconds_for("fake", 2.0) == 3.0
        assert router.plan(request()).estimate_seconds == 10.0
        result = router.generate(
            request(seeds=(1,)), tmp_path / "run", timeout_s=60, cancel=threading.Event()
        )
        assert result.ok
        with pytest.raises(ValueError, match="unknown engine"):
            router.plan(request(engine="nope"))
        assert len(router.health()) == 2
    finally:
        router.shutdown()
    assert not fake.engine.running()


def test_the_router_unloads_the_other_engine(tmp_path: Path) -> None:
    one, two = FakeEngineBackend(tmp_path / "1"), FakeEngineBackend(tmp_path / "2")
    router = EngineRouter({"one": one, "two": two})
    try:
        assert router.generate(
            request(engine="one", seeds=(1,)),
            tmp_path / "r1",
            timeout_s=60,
            cancel=threading.Event(),
        ).ok
        assert one.engine.running() and not two.engine.running()
        assert router.generate(
            request(engine="two", seeds=(1,)),
            tmp_path / "r2",
            timeout_s=60,
            cancel=threading.Event(),
        ).ok
        assert two.engine.running() and not one.engine.running()  # one was unloaded first
    finally:
        router.shutdown()


def test_loaded_is_only_reported_once_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "hang")
    backend = FakeEngineBackend(tmp_path)
    try:
        assert not backend.engine_status("fake").loaded
        assert backend.engine.acquire("job_1", threading.Event(), time.monotonic() + 30)
        assert backend.engine_status("fake").loaded
    finally:
        backend.shutdown()
    assert not backend.engine_status("fake").loaded


def test_build_backend_reads_both_tables(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        f'[backends.sa3]\nmlx_dir = "{tmp_path / "sa3"}"\nnegative_prompt_cfg = 3.0\n'
        f"[backends.moss]\nsteps = 50\n"
    )
    router = build_backend(load_settings(config_path=config, env={}))
    sa3, moss = router.engines["sa3"], router.engines["moss"]
    assert isinstance(sa3, Sa3Backend) and isinstance(moss, MossBackend)
    assert sa3.options.negative_prompt_cfg == 3.0 and moss.options.steps == 50
    assert sa3.engine.config.state_dir == tmp_path / "data" / "engine" / "sa3"
    assert not router.engine_status("moss").available

    config.write_text("[backends.moss]\nsteps = 0\n")
    with pytest.raises(SettingsError, match=r"\[backends.moss\].*steps"):
        build_backend(load_settings(config_path=config, env={}))


def test_the_router_fans_out_lifecycle_calls(tmp_path: Path) -> None:
    calls: list[str] = []

    class Recorder(FakeEngineBackend):
        def recover(self, run_dir: Path) -> str | None:
            calls.append("recover")
            return "stopped x"

        def startup(self) -> str | None:
            calls.append("startup")
            return None

        def shutdown(self) -> None:
            calls.append("shutdown")
            raise RuntimeError("stuck")

    router = EngineRouter({"a": Recorder(tmp_path / "a"), "b": Recorder(tmp_path / "b")})
    assert router.recover(tmp_path) == "stopped x; stopped x"
    assert router.startup() is None
    seen: list[bool] = []
    router.set_peer_check(lambda: seen.append(True) or False)
    with pytest.raises(RuntimeError, match="stuck"):
        router.shutdown()
    assert calls.count("shutdown") == 2  # the second engine is stopped even so
    assert router.typical_seconds(2.0) == 3.0
    assert router.uses_negative_prompt_for("a") is False
    with pytest.raises(ValueError, match="no engines"):
        EngineRouter({})

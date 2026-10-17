"""The resident engine and the worker protocol, driving a stand-in worker (tests/fake_worker.py,
which speaks the real protocol module) in a real child process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from sfx_api.backends.engine import EngineConfig, EngineError, ResidentEngine, worker_command
from sfx_api.backends.process import PID_FILE
from sfx_api.workers import sfx_worker_protocol as protocol

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "ok")


def config(tmp_path: Path, **overrides: object) -> EngineConfig:
    values: dict[str, object] = {
        "name": "fake",
        "python": Path(sys.executable),
        "script": FAKE_WORKER,
        "args": ("--model", "m"),
        "cwd": tmp_path,
        "state_dir": tmp_path / "state",
        "idle_seconds": 60.0,
        "poll_seconds": 0.05,
        "start_timeout_s": 30.0,
        "stop_grace_s": 2.0,
    }
    return EngineConfig(**(values | overrides))  # type: ignore[arg-type]


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[ResidentEngine]:
    eng = ResidentEngine(config(tmp_path))
    yield eng
    eng.shutdown()


def clip(tmp_path: Path, name: str = "raw-0.wav", **overrides: object) -> dict[str, object]:
    return {
        "prompt": "tyre screech",
        "negative_prompt": "music",
        "duration": 0.5,
        "seed": 7,
        "out": str(tmp_path / name),
    } | overrides


def far() -> float:
    return time.monotonic() + 30


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return (
        subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()[:1]
        != "Z"
    )


def wait(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.02)


def test_worker_command_is_fixed(tmp_path: Path) -> None:
    argv = worker_command(config(tmp_path), 9)
    assert argv == [sys.executable, "-P", str(FAKE_WORKER), "--proto-fd", "9", "--model", "m"]


def test_clips_reuse_one_loaded_worker(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_WORKER_RECORD", str(record))
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    assert engine.last_load_s is not None and engine.running()
    assert json.loads(record.read_text()) == ["--model", "m"]
    pid = engine.pid
    progress: list[float] = []
    first = engine.run_clip(
        "job_1", clip(tmp_path), cancel=cancel, deadline=far(), on_progress=progress.append
    )
    assert first.ok and first.reply["sample_rate"] == 44100 and first.reply["channels"] == 2
    assert first.reply["frames"] == 22050 and first.reply["cfg"] == 1.0
    assert first.reply["gain_db"] == -1.0  # the full-scale square was turned down
    assert progress == [0.1, 0.52, 0.99]
    asked = json.loads((tmp_path / "raw-0.wav.params.json").read_text())
    assert asked == {
        "prompt": "tyre screech",
        "negative_prompt": "music",
        "duration": 0.5,
        "seed": 7,
    }
    engine.release(job_id="job_1")
    assert engine.acquire("job_2", cancel, far())
    assert engine.pid == pid and engine.last_load_s is None  # no second load
    second = engine.run_clip(
        "job_2", clip(tmp_path, "b.wav", seed=9), cancel=cancel, deadline=far()
    )
    assert second.ok
    assert json.loads((tmp_path / "b.wav.params.json").read_text())["seed"] == 9
    engine.release(job_id="job_2")
    # nothing half-written is left next to the outputs
    assert sorted(p.name for p in tmp_path.glob("*.partial")) == []


def test_the_written_wav_is_16_bit_pcm_under_the_ceiling(
    engine: ResidentEngine, tmp_path: Path
) -> None:
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    assert engine.run_clip("job_1", clip(tmp_path), cancel=cancel, deadline=far()).ok
    with wave.open(str(tmp_path / "raw-0.wav"), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 44100)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert int(np.abs(pcm.astype(np.int32)).max()) <= round(32767 * 10 ** (-1 / 20))


def test_stdout_noise_does_not_break_the_protocol(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "noisy")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    assert engine.run_clip("job_1", clip(tmp_path), cancel=cancel, deadline=far()).ok
    assert "Sampling... 52%" in (tmp_path / "state" / "worker.log").read_text()


def test_a_failed_load_raises_with_the_reason(
    engine: ResidentEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "load_fail")
    with pytest.raises(EngineError, match="checkpoint missing"):
        engine.acquire("job_1", threading.Event(), far())
    assert not engine.running()


def test_an_engine_error_keeps_the_worker(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "gen_error")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    outcome = engine.run_clip("job_1", clip(tmp_path), cancel=cancel, deadline=far())
    assert not outcome.ok and "out of memory" in (outcome.failure or "")
    assert engine.running()


def test_cancel_and_timeout_stop_the_worker(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "hang")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    pid = engine.pid
    assert pid is not None
    threading.Timer(0.5, cancel.set).start()
    outcome = engine.run_clip("job_1", clip(tmp_path), cancel=cancel, deadline=far())
    assert outcome.canceled and not outcome.ok
    assert not engine.running() and not alive(pid)
    assert not (tmp_path / "state" / PID_FILE).exists()

    assert engine.acquire("job_2", threading.Event(), far())
    pid = engine.pid
    assert pid is not None
    late = engine.run_clip(
        "job_2", clip(tmp_path), cancel=threading.Event(), deadline=time.monotonic() + 0.5
    )
    assert late.timed_out and not alive(pid)


def test_a_worker_that_dies_mid_clip_is_a_failure(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "crash")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    outcome = engine.run_clip("job_1", clip(tmp_path), cancel=cancel, deadline=far())
    assert not outcome.ok and "exited" in (outcome.failure or "")
    # the next job starts a fresh worker
    monkeypatch.setenv("FAKE_WORKER_MODE", "ok")
    engine.release(discard=True)
    assert engine.acquire("job_2", cancel, far())
    assert engine.run_clip("job_2", clip(tmp_path), cancel=cancel, deadline=far()).ok


def test_idle_and_busy_peers_stop_the_worker(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path, idle_seconds=0.3))
    try:
        cancel = threading.Event()
        assert eng.acquire("job_1", cancel, far())
        time.sleep(0.6)
        assert eng.running(), "never stopped while in use"
        eng.release()
        wait(lambda: not eng.running())

        busy = threading.Event()
        eng2 = ResidentEngine(config(tmp_path, idle_seconds=60))
        eng2.set_peer_check(busy.is_set)
        try:
            assert eng2.acquire("job_2", cancel, far())
            eng2.release()
            time.sleep(0.2)
            assert eng2.running()
            busy.set()
            wait(lambda: not eng2.running())
        finally:
            eng2.shutdown()
    finally:
        eng.shutdown()


def test_resident_off_stops_after_each_job(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path, idle_seconds=0))
    try:
        assert eng.acquire("job_1", threading.Event(), far())
        eng.release()
        assert not eng.running()
    finally:
        eng.shutdown()


def test_cancel_while_loading_and_shutdown(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path))
    cancel = threading.Event()
    cancel.set()
    assert not eng.acquire("job_1", cancel, far())
    assert not eng.running()
    eng.shutdown()
    with pytest.raises(EngineError, match="stopping"):
        eng.acquire("job_2", threading.Event(), far())


def test_missing_python_is_an_engine_error(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path, python=tmp_path / "nope" / "python"))
    with pytest.raises(EngineError, match="not an executable Python"):
        eng.acquire("job_1", threading.Event(), far())


def test_a_leftover_worker_is_reaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "hang")
    state = tmp_path / "state"
    state.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-P", str(FAKE_WORKER), "--proto-fd", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        (state / PID_FILE).write_text(str(proc.pid))
        message = ResidentEngine(config(tmp_path)).reap_leftover()
        assert message == f"stopped leftover fake_worker.py (pid {proc.pid})"
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# -- the worker side of the protocol -----------------------------------------------------------


def test_fit_peak_only_turns_down() -> None:
    quiet = np.full((100, 2), 0.25, dtype=np.float32)
    same, gain = protocol.fit_peak(quiet)
    assert gain == 0.0 and np.array_equal(same, quiet)
    loud = np.zeros((1000, 2), dtype=np.float32)
    loud[500] = 1.6  # one spike: the whole clip goes down, the shape stays
    loud[100] = 0.8
    fitted, gain = protocol.fit_peak(loud)
    ceiling = 10 ** (-1 / 20)
    assert abs(float(np.abs(fitted).max()) - ceiling) < 1e-6
    assert gain == pytest.approx(20 * np.log10(ceiling / 1.6), abs=0.01)
    assert fitted[100, 0] / fitted[500, 0] == pytest.approx(0.5)


def test_fit_peak_cleans_nan() -> None:
    data = np.array([[np.nan, 0.5], [0.1, np.inf]], dtype=np.float32)
    fitted, _ = protocol.fit_peak(data)
    assert np.isfinite(fitted).all() and float(np.abs(fitted).max()) <= 1.0


def test_write_wav_mono(tmp_path: Path) -> None:
    path = tmp_path / "m.wav"
    protocol.write_wav(path, np.linspace(-1, 1, 480, dtype=np.float32)[:, None], 48000)
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 48000, 480)

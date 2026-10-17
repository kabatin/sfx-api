"""What every engine backend shares: a resident worker (:mod:`sfx_api.backends.engine`) in the
engine checkout's own Python, one clip per request, and the job loop around it.

An engine backend only says where its checkout and models are (``health``), how to start its
worker (``engine_config``) and how long a clip takes (``typical_seconds``).
"""

from __future__ import annotations

import logging
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from sfx_api.backends.base import (
    EngineStatus,
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
    SoundBackend,
)
from sfx_api.backends.engine import EngineConfig, EngineError, ResidentEngine
from sfx_api.checks import Check
from sfx_api.logs import event
from sfx_api.settings import Settings, SettingsError

WORKERS_DIR = Path(__file__).resolve().parent.parent / "workers"


class ResidentBackend(SoundBackend):
    supports_cancel = True  # the worker is killed; the GPU is released

    def __init__(self, *, load_seconds: float) -> None:
        self.load_seconds = load_seconds
        self.engine = ResidentEngine(self.engine_config())
        self.log = logging.getLogger(f"sfx_api.{self.engine.config.name}")

    @abstractmethod
    def engine_config(self) -> EngineConfig:
        """How to start this engine's worker."""

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Why the engine cannot take jobs (operator-facing, short), or None."""

    def engine_status(self, engine: str) -> EngineStatus:
        reason = self.unavailable_reason()
        return EngineStatus(available=reason is None, loaded=self.engine.running(), reason=reason)

    # -- planning --------------------------------------------------------------------
    def plan(self, request: GenerationRequest) -> GenerationPlan:
        load = 0.0 if self.engine.running() else self.load_seconds
        estimate = load + self.typical_seconds(request.duration_s) * request.count
        return GenerationPlan(request.count, estimate_seconds=round(estimate, 1))

    # -- generation ------------------------------------------------------------------
    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        plan = self.plan(request)
        started = time.monotonic()
        deadline = started + timeout_s
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return GenerationResult(plan=plan, duration_s=0.0, failure=f"cannot prepare: {exc}")
        reason = self.unavailable_reason()
        if reason is not None:
            return GenerationResult(
                plan=plan, duration_s=0.0, failure=reason, failure_code="backend_unavailable"
            )
        result: GenerationResult | None = None
        try:
            result = self._run(request, plan, run_dir, started, deadline, cancel, on_progress)
            return result
        finally:
            # a canceled or failed clip may leave the GPU busy: never reuse that worker
            ok = result is not None and result.ok
            self.engine.release(discard=not ok, job_id=request.job_id)

    def _run(
        self,
        request: GenerationRequest,
        plan: GenerationPlan,
        run_dir: Path,
        started: float,
        deadline: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None,
    ) -> GenerationResult:
        def elapsed() -> float:
            return time.monotonic() - started

        try:
            loaded = self.engine.acquire(request.job_id, cancel, deadline)
        except EngineError as exc:
            return GenerationResult(
                plan=plan, duration_s=elapsed(), failure=str(exc), failure_code="engine_failed"
            )
        if not loaded:
            return GenerationResult(
                plan=plan,
                duration_s=elapsed(),
                canceled=cancel.is_set(),
                timed_out=not cancel.is_set(),
            )
        details: dict[str, Any] = {"load_seconds": self.engine.last_load_s, "clips": []}
        outputs: list[Path] = []
        for index, seed in enumerate(request.seeds):
            out = run_dir / f"raw-{index}.wav"
            outcome = self.engine.run_clip(
                request.job_id,
                {
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "duration": request.duration_s,
                    "seed": seed,
                    "out": str(out.resolve()),
                },
                cancel=cancel,
                deadline=deadline,
                on_progress=_scaled(on_progress, index, request.count),
            )
            details["clips"].append(
                {
                    "index": index,
                    "seconds": round(outcome.seconds, 3),
                    **{
                        key: outcome.reply[key]
                        for key in ("gain_db", "cfg", "steps", "latent_seconds")
                        if key in outcome.reply
                    },
                }
            )
            if not outcome.ok:
                return GenerationResult(
                    plan=plan,
                    duration_s=elapsed(),
                    outputs=tuple(outputs),
                    canceled=outcome.canceled,
                    timed_out=outcome.timed_out,
                    failure=outcome.failure,
                    details=details,
                )
            if not out.is_file() or out.stat().st_size == 0:
                return GenerationResult(
                    plan=plan,
                    duration_s=elapsed(),
                    outputs=tuple(outputs),
                    failure=f"clip {index}: the engine wrote no audio",
                    details=details,
                )
            outputs.append(out)
            event(
                self.log,
                "engine.clip",
                engine=self.engine.config.name,
                job_id=request.job_id,
                index=index,
                seconds=round(outcome.seconds, 3),
            )
        return GenerationResult(
            plan=plan, duration_s=elapsed(), outputs=tuple(outputs), details=details
        )

    def health(self) -> list[Check]:
        return self.checks()

    @abstractmethod
    def checks(self) -> list[Check]:
        """Checkout, Python and model files (for doctor and the startup log)."""

    def unload(self, reason: str) -> None:
        if self.engine.running():
            self.engine.stop(reason)

    def startup(self) -> str | None:
        return self.engine.reap_leftover()

    def shutdown(self) -> None:
        self.engine.shutdown()

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        self.engine.set_peer_check(check)


def _scaled(
    on_progress: ProgressCallback | None, index: int, count: int
) -> Callable[[float], None] | None:
    if on_progress is None:
        return None

    def report(fraction: float) -> None:
        on_progress(min(max((index + fraction) / count, 0.0), 1.0))

    return report


def parse_options[T: BaseModel](model: type[T], settings: Settings, name: str) -> T:
    """Validate the ``[backends.<name>]`` table of the config file."""
    try:
        return model.model_validate(dict(settings.backend_options(name)))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise SettingsError(f"invalid [backends.{name}] options: {problems}") from exc

"""One backend in front of several engines: each request names its engine (set by its
workflow), and the router hands it to that engine's backend. The job queue stays a single
line, so the engines take turns on the GPU like every other job, and only the engine of the
running job keeps its models in memory."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

from sfx_api.backends.base import (
    EngineStatus,
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
    SoundBackend,
)
from sfx_api.checks import Check


class EngineRouter(SoundBackend):
    name: ClassVar[str] = "engines"
    supports_cancel: ClassVar[bool] = True  # every engine kills its worker on cancel

    def __init__(self, engines: Mapping[str, SoundBackend]) -> None:
        if not engines:
            raise ValueError("no engines")
        self.engines = dict(engines)

    def _engine(self, name: str) -> SoundBackend:
        try:
            return self.engines[name]
        except KeyError:
            raise ValueError(f"unknown engine {name!r}") from None

    def typical_seconds(self, duration_s: float) -> float:
        return max(b.typical_seconds(duration_s) for b in self.engines.values())

    def typical_seconds_for(self, engine: str, duration_s: float) -> float:
        return self._engine(engine).typical_seconds(duration_s)

    def uses_negative_prompt_for(self, engine: str) -> bool:
        return self._engine(engine).uses_negative_prompt

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        return self._engine(request.engine).plan(request)

    def engine_status(self, engine: str) -> EngineStatus:
        if engine not in self.engines:
            return EngineStatus(available=False, reason=f"unknown engine {engine!r}")
        return self.engines[engine].engine_status(engine)

    def health(self) -> list[Check]:
        return [check for backend in self.engines.values() for check in backend.health()]

    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        backend = self._engine(request.engine)
        # one engine in memory at a time: an idle one still loaded would crowd the next
        for name, other in self.engines.items():
            if name != request.engine:
                other.unload(f"engine {request.engine} takes the GPU")
        return backend.generate(
            request, run_dir, timeout_s=timeout_s, cancel=cancel, on_progress=on_progress
        )

    def recover(self, run_dir: Path) -> str | None:
        done = [r for b in self.engines.values() if (r := b.recover(run_dir))]
        return "; ".join(done) or None

    def startup(self) -> str | None:
        done = [r for b in self.engines.values() if (r := b.startup())]
        return "; ".join(done) or None

    def shutdown(self) -> None:
        errors: list[Exception] = []
        for backend in self.engines.values():
            try:
                backend.shutdown()
            except Exception as exc:  # stop the others first, then report
                errors.append(exc)
        if errors:
            raise errors[0]

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        for backend in self.engines.values():
            backend.set_peer_check(check)

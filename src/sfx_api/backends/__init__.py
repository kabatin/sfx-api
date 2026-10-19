"""Backends: the engines that make the sounds (Stable Audio 3 sm-sfx and medium,
MOSS-SoundEffect v2.0)."""

from __future__ import annotations

from sfx_api.backends.base import SoundBackend
from sfx_api.backends.router import EngineRouter
from sfx_api.settings import Settings


def build_backend(settings: Settings) -> EngineRouter:
    from sfx_api.backends.moss import MossBackend
    from sfx_api.backends.sa3 import Sa3Backend, Sa3MediumBackend

    return EngineRouter(
        {
            Sa3Backend.name: Sa3Backend.from_settings(settings),
            Sa3MediumBackend.name: Sa3MediumBackend.from_settings(settings),
            MossBackend.name: MossBackend.from_settings(settings),
        }
    )


__all__ = ["EngineRouter", "SoundBackend", "build_backend"]

"""Built-in workflows. Registration is explicit: add new workflows to ``build_registry``."""

from __future__ import annotations

from sfx_api.backends.router import EngineRouter
from sfx_api.workflows.base import Workflow, WorkflowRegistry
from sfx_api.workflows.music import StableAudio3MusicWorkflow
from sfx_api.workflows.sfx import (
    MossWorkflow,
    SfxWorkflow,
    StableAudio3MediumWorkflow,
    StableAudio3Workflow,
)


def build_registry(backend: EngineRouter) -> WorkflowRegistry:
    kinds: list[type[SfxWorkflow]] = [
        StableAudio3Workflow,
        StableAudio3MediumWorkflow,
        StableAudio3MusicWorkflow,
        MossWorkflow,
    ]
    workflows: list[Workflow] = [
        kind(
            typical_seconds_2s=backend.typical_seconds_for(kind.engine, 2.0),
            uses_negative_prompt=backend.uses_negative_prompt_for(kind.engine),
        )
        for kind in kinds
    ]
    return WorkflowRegistry(workflows)


def workflow_engine(workflow: Workflow) -> str | None:
    """The engine a workflow runs on (None for workflows without one, e.g. test doubles)."""
    return workflow.engine if isinstance(workflow, SfxWorkflow) else None


__all__ = ["WorkflowRegistry", "build_registry", "workflow_engine"]

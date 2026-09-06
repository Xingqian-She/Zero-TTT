"""Pure definitions and state aggregation for the three finite workflows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from zero_ttt_contracts import (
    JobState,
    ResourceClass,
    WorkerCapability,
    WorkflowState,
    WorkflowTemplate,
)


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    kind: str
    capability: WorkerCapability
    resource_class: ResourceClass


def workflow_steps(template: WorkflowTemplate) -> tuple[WorkflowStep, ...]:
    if template is WorkflowTemplate.DATA_BOOTSTRAP:
        return tuple(
            WorkflowStep(kind, WorkerCapability.DATA, ResourceClass.DATA_WRITER)
            for kind in (
                "data.scan",
                "data.trial-import",
                "data.verify-trial",
                "data.full-import",
                "data.verify",
                "data.snapshot-train",
                "data.snapshot-validation",
            )
        )
    if template is WorkflowTemplate.COLD_START:
        return (
            WorkflowStep(
                "trainer.cold-start", WorkerCapability.TRAINER, ResourceClass.GPU_EXCLUSIVE
            ),
        )
    if template is WorkflowTemplate.ALPHA_ZERO_ROUND:
        return (
            WorkflowStep(
                "selfplay.collect", WorkerCapability.SELFPLAY, ResourceClass.GPU_EXCLUSIVE
            ),
            WorkflowStep("data.admit-selfplay", WorkerCapability.DATA, ResourceClass.DATA_WRITER),
            WorkflowStep(
                "data.snapshot-selfplay", WorkerCapability.DATA, ResourceClass.DATA_WRITER
            ),
            WorkflowStep("trainer.mixture", WorkerCapability.TRAINER, ResourceClass.GPU_EXCLUSIVE),
        )
    raise ValueError(f"unsupported workflow template: {template}")


def workflow_state(job_states: Iterable[JobState]) -> WorkflowState:
    states = set(job_states)
    if states == {JobState.SUCCEEDED}:
        return WorkflowState.SUCCEEDED
    if JobState.FAILED in states:
        return WorkflowState.FAILED
    if JobState.CANCELLED in states:
        return WorkflowState.CANCELLED
    if states == {JobState.QUEUED}:
        return WorkflowState.QUEUED
    return WorkflowState.RUNNING

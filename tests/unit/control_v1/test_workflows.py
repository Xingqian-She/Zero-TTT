from __future__ import annotations

import pytest
from zero_ttt_contracts import JobState, ResourceClass, WorkflowState, WorkflowTemplate
from zero_ttt_control.workflows import workflow_state, workflow_steps


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([JobState.QUEUED], WorkflowState.QUEUED),
        ([JobState.SUCCEEDED], WorkflowState.SUCCEEDED),
        ([JobState.SUCCEEDED, JobState.QUEUED], WorkflowState.RUNNING),
        ([JobState.CANCEL_REQUESTED, JobState.QUEUED], WorkflowState.RUNNING),
        ([JobState.CANCELLED, JobState.QUEUED], WorkflowState.CANCELLED),
        ([JobState.FAILED, JobState.CANCELLED], WorkflowState.FAILED),
    ],
)
def test_workflow_terminal_state_precedence(states, expected):
    assert workflow_state(states) is expected


def test_alpha_round_preserves_steps_and_resource_ownership():
    steps = workflow_steps(WorkflowTemplate.ALPHA_ZERO_ROUND)
    assert [step.kind for step in steps] == [
        "selfplay.collect",
        "data.admit-selfplay",
        "data.snapshot-selfplay",
        "trainer.mixture",
    ]
    assert [step.resource_class for step in steps] == [
        ResourceClass.GPU_EXCLUSIVE,
        ResourceClass.DATA_WRITER,
        ResourceClass.DATA_WRITER,
        ResourceClass.GPU_EXCLUSIVE,
    ]

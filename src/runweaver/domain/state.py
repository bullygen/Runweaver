"""Centralized domain state transition rules."""

from __future__ import annotations

from runweaver.exceptions import StateTransitionError

from .models import RunState

TERMINAL_STATES = {
    RunState.COMPLETED,
    RunState.CANCELLED,
    RunState.PRUNED,
    RunState.SKIPPED,
    RunState.CACHED,
}

ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PLANNED: frozenset({RunState.QUEUED, RunState.CANCELLED, RunState.SKIPPED}),
    RunState.QUEUED: frozenset({RunState.RUNNING, RunState.CANCELLED, RunState.ORPHANED}),
    RunState.RUNNING: frozenset({
        RunState.COMPLETED, RunState.FAILED, RunState.PAUSING, RunState.CANCELLED,
        RunState.PRUNED, RunState.RETRY_WAIT, RunState.STALE, RunState.CACHED,
    }),
    RunState.PAUSING: frozenset({RunState.PAUSED, RunState.CANCELLED, RunState.FAILED}),
    RunState.PAUSED: frozenset({RunState.QUEUED, RunState.CANCELLED}),
    RunState.FAILED: frozenset({RunState.QUEUED, RunState.CANCELLED}),
    RunState.RETRY_WAIT: frozenset({RunState.QUEUED, RunState.CANCELLED}),
    RunState.STALE: frozenset({RunState.QUEUED, RunState.ORPHANED, RunState.CANCELLED}),
    RunState.ORPHANED: frozenset({RunState.QUEUED, RunState.CANCELLED}),
    RunState.COMPLETED: frozenset({RunState.STALE}),
    RunState.CACHED: frozenset({RunState.STALE}),
    RunState.CANCELLED: frozenset(),
    RunState.PRUNED: frozenset(),
    RunState.SKIPPED: frozenset(),
}


def validate_transition(current: RunState, target: RunState) -> None:
    """Raise if ``current -> target`` violates domain lifecycle semantics."""

    if target == current:
        return
    if target not in ALLOWED_TRANSITIONS[current]:
        raise StateTransitionError(f"invalid state transition: {current.value} -> {target.value}")

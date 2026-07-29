from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from runweaver.domain.models import RunKind, RunState, utc_now
from runweaver.exceptions import StateTransitionError
from runweaver.persistence import SqlAlchemyStateStore


def test_state_store_optimistic_lock_and_stale_lease(tmp_path: Path) -> None:
    store = SqlAlchemyStateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.initialize()
    record = store.create_run(logical_key="trial:one", kind=RunKind.TRIAL)
    queued = store.transition(record.id, RunState.QUEUED, expected_version=record.version)
    with pytest.raises(StateTransitionError, match="optimistic lock"):
        store.transition(record.id, RunState.RUNNING, expected_version=record.version)
    running = store.transition(queued.id, RunState.RUNNING)
    assert store.acquire_lease(running.id, "worker-1", duration_s=1)
    stale = store.reconcile_stale(now=utc_now() + timedelta(seconds=2))
    assert stale == [running.id]
    assert store.get_run(running.id).state == RunState.STALE


def test_logical_key_makes_creation_idempotent(tmp_path: Path) -> None:
    store = SqlAlchemyStateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.initialize()
    first = store.create_run(logical_key="same", kind=RunKind.BLOCK)
    second = store.create_run(logical_key="same", kind=RunKind.BLOCK)
    assert first.id == second.id

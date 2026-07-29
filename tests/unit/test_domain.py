from __future__ import annotations

import pytest
from pydantic import ValidationError
from runweaver import (
    CacheMode,
    FloatParameter,
    ParameterSpace,
    RetryPolicy,
    RunState,
)
from runweaver.domain.models import BlockSpec, SideEffect
from runweaver.domain.state import validate_transition
from runweaver.exceptions import StateTransitionError


def test_domain_models_are_immutable() -> None:
    space = ParameterSpace(parameters=(FloatParameter(name="x", low=0, high=1),))
    with pytest.raises(ValidationError):
        space.version = "2"  # type: ignore[misc]


def test_non_deterministic_cache_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-deterministic"):
        BlockSpec(
            name="random",
            version="1",
            input_schema={},
            output_schema={},
            code_fingerprint="abc",
            deterministic=False,
            cache_policy=CacheMode.READ_WRITE,
        )


def test_irreversible_non_idempotent_retry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-idempotent"):
        BlockSpec(
            name="charge-card",
            version="1",
            input_schema={},
            output_schema={},
            code_fingerprint="abc",
            idempotent=False,
            cache_policy=CacheMode.DISABLED,
            retry_policy=RetryPolicy(max_attempts=2),
            side_effects=(SideEffect(name="charge", description="charge once"),),
        )


def test_state_machine_accepts_resume_and_rejects_terminal_reentry() -> None:
    validate_transition(RunState.PAUSED, RunState.QUEUED)
    validate_transition(RunState.QUEUED, RunState.RUNNING)
    with pytest.raises(StateTransitionError):
        validate_transition(RunState.COMPLETED, RunState.RUNNING)

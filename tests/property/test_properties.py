from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
from runweaver import FloatParameter, IntegerParameter
from runweaver.planning.space import from_unit, to_unit


@given(
    low=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=1e-3, max_value=100, allow_nan=False, allow_infinity=False),
    unit=st.floats(min_value=0, max_value=0.999999, allow_nan=False, allow_infinity=False),
)
def test_continuous_transform_round_trip(low: float, width: float, unit: float) -> None:
    parameter = FloatParameter(name="x", low=low, high=low + width)
    assert abs(to_unit(parameter, from_unit(parameter, unit)) - unit) < 1e-9


@given(unit=st.floats(min_value=0, max_value=0.999999, allow_nan=False, allow_infinity=False))
def test_integer_transform_stays_in_bounds(unit: float) -> None:
    parameter = IntegerParameter(name="n", low=2, high=9)
    value = from_unit(parameter, unit)
    assert 2 <= value <= 9

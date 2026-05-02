"""Unit tests for the rating -> target weight mapping."""

from __future__ import annotations

import math

import pytest

from tradingagents.execution.targets import (
    HOLD,
    RATING_TO_SLOT_FRACTION,
    compute_target_weights,
)


def test_buy_consumes_full_slot_in_equal_weight_basket():
    targets = compute_target_weights({"AAPL": "Buy", "MSFT": "Buy"}, equal_weight_cap=1.0)
    # Two names, equal-weight cap 1.0 -> slot = 0.5; Buy = 1.0 of slot
    assert math.isclose(targets["AAPL"].target_weight, 0.5)
    assert math.isclose(targets["MSFT"].target_weight, 0.5)


def test_overweight_and_underweight_scale_correctly():
    targets = compute_target_weights(
        {"AAPL": "Overweight", "MSFT": "Underweight"}, equal_weight_cap=1.0
    )
    slot = 0.5
    assert math.isclose(targets["AAPL"].target_weight, slot * RATING_TO_SLOT_FRACTION["Overweight"])
    assert math.isclose(targets["MSFT"].target_weight, slot * RATING_TO_SLOT_FRACTION["Underweight"])


def test_sell_zeros_target():
    targets = compute_target_weights({"AAPL": "Sell"})
    assert targets["AAPL"].target_weight == 0.0


def test_hold_carries_no_target_weight():
    targets = compute_target_weights({"AAPL": HOLD})
    assert targets["AAPL"].target_weight is None
    assert targets["AAPL"].slot_fraction is None


def test_unknown_rating_raises():
    with pytest.raises(ValueError):
        compute_target_weights({"AAPL": "Strong Buy"})


def test_cap_below_one_caps_basket_exposure():
    # 7 names all Buy with cap 0.5 -> each ~7.14%, total 50%
    ratings = {t: "Buy" for t in "AAPL MSFT GOOGL AMZN META NVDA TSLA".split()}
    targets = compute_target_weights(ratings, equal_weight_cap=0.5)
    total = sum(t.target_weight for t in targets.values())
    assert math.isclose(total, 0.5, rel_tol=1e-9)

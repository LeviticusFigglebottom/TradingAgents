"""Map a 5-tier portfolio rating to a target weight in the basket.

The framework's Portfolio Manager outputs one of: Buy / Overweight / Hold /
Underweight / Sell. We translate that into a target dollar weight relative
to an equal-weight slot in the watchlist, then the executor reconciles
each ticker's actual position against that target.

For a 7-name basket and ``equal_weight_cap = 1.0``:
    Buy         -> 1.00 of slot   = ~14.3% of equity
    Overweight  -> 0.66 of slot   = ~9.4% of equity
    Hold        -> keep current weight (no trade unless drifted past tolerance)
    Underweight -> 0.33 of slot   = ~4.7% of equity
    Sell        -> 0.0 of slot    = flat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


HOLD = "Hold"

# Fraction of an equal-weight slot to allocate per rating.
# Hold is special: it means "do nothing" (target = current), not a numeric weight.
RATING_TO_SLOT_FRACTION: Dict[str, float] = {
    "Buy": 1.00,
    "Overweight": 0.66,
    "Underweight": 0.33,
    "Sell": 0.0,
}

VALID_RATINGS = set(RATING_TO_SLOT_FRACTION) | {HOLD}


@dataclass(frozen=True)
class TargetWeight:
    ticker: str
    rating: str
    slot_fraction: Optional[float]  # None when rating == Hold
    target_weight: Optional[float]  # None when rating == Hold (current weight is kept)


def compute_target_weights(
    ratings: Dict[str, str],
    *,
    equal_weight_cap: float = 1.0,
) -> Dict[str, TargetWeight]:
    """Translate per-ticker ratings into target weights.

    Args:
        ratings: ticker -> rating string from the PM (Buy/Overweight/Hold/...).
        equal_weight_cap: fraction of total equity to deploy across the basket.
            1.0 = fully invested at max conviction; 0.5 = at most half equity in
            equities even if every name is Buy.

    Returns:
        ticker -> TargetWeight. Hold entries carry ``target_weight=None``;
        the executor must read the current weight from Alpaca and pass it
        through unchanged.
    """
    if not 0.0 < equal_weight_cap <= 1.0:
        raise ValueError(f"equal_weight_cap must be in (0, 1], got {equal_weight_cap}")

    n = len(ratings)
    if n == 0:
        return {}

    slot = equal_weight_cap / n  # max equity fraction per name
    out: Dict[str, TargetWeight] = {}
    for ticker, rating in ratings.items():
        if rating not in VALID_RATINGS:
            raise ValueError(f"Unknown rating {rating!r} for {ticker}")
        if rating == HOLD:
            out[ticker] = TargetWeight(ticker, rating, None, None)
        else:
            frac = RATING_TO_SLOT_FRACTION[rating]
            out[ticker] = TargetWeight(ticker, rating, frac, slot * frac)
    return out

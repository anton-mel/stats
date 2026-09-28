"""Statistics the harness lives by.

* ``cov`` — coefficient of variation across rounds. A cell whose rounds
  spread more than the policy threshold is NOISY and its number is not read.
* ``decide_repetition`` — adaptive repetition: one round, compared against
  the cell's history; only a result outside ``sigma`` × the historical
  spread triggers confirmation rounds.
* ``regression_verdict`` — noise-aware regression: the threshold is the
  cell's own historical CoV, not a fixed percentage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def median(xs: list[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=float)))


def cov(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    a = np.asarray(xs, dtype=float)
    m = a.mean()
    return float(a.std(ddof=1) / m) if m else 0.0


@dataclass
class AdaptiveDecision:
    more_rounds: int
    reason: str


def decide_repetition(
    first: float,
    history: list[float],
    *,
    sigma: float,
    confirm_rounds: int,
    min_history: int = 3,
    floor_rel: float = 0.01,
) -> AdaptiveDecision:
    """After the first round: 0 more rounds if the value sits inside the
    historical band, otherwise ``confirm_rounds`` more. With too little
    history we always confirm (the band is unknown)."""
    if len(history) < min_history:
        return AdaptiveDecision(confirm_rounds, f"history has {len(history)} < {min_history} points")
    h = np.asarray(history, dtype=float)
    center = float(np.median(h))
    spread = max(float(np.std(h, ddof=1)), floor_rel * center)
    z = abs(first - center) / spread if spread else 0.0
    if z <= sigma:
        return AdaptiveDecision(0, f"within band (z={z:.2f} ≤ {sigma})")
    return AdaptiveDecision(confirm_rounds, f"outside band (z={z:.2f} > {sigma}); confirming")


@dataclass
class RegressionVerdict:
    regressed: bool
    improved: bool
    delta_rel: float
    threshold_rel: float
    note: str


def regression_verdict(
    value: float,
    history: list[float],
    *,
    sigma: float = 3.0,
    floor_rel: float = 0.01,
    higher_is_better: bool = True,
) -> RegressionVerdict:
    if not history:
        return RegressionVerdict(False, False, 0.0, 0.0, "no history")
    h = np.asarray(history, dtype=float)
    center = float(np.median(h))
    if center == 0:
        return RegressionVerdict(False, False, 0.0, 0.0, "zero baseline")
    hist_cov = cov(list(h)) if len(h) >= 2 else floor_rel
    thr = max(sigma * hist_cov, floor_rel)
    delta = (value - center) / center
    worse = delta < -thr if higher_is_better else delta > thr
    better = delta > thr if higher_is_better else delta < -thr
    return RegressionVerdict(worse, better, delta, thr, f"center={center:.4g} hist_cov={hist_cov:.4f}")

"""Order schedules: how a parent order is split across time buckets.

Every function returns weights w with w >= 0 and sum(w) = 1 along the last axis.
w[k] is the fraction of the parent order traded in bucket k.
All functions broadcast over leading axes, so a (days, buckets) array works.
"""
from __future__ import annotations

import numpy as np


def _normalise(curve) -> np.ndarray:
    c = np.asarray(curve, dtype=float)
    if np.any(c < 0):
        raise ValueError("curve must be non-negative")
    total = c.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("curve must have positive total")
    return c / total


def twap(n_buckets: int) -> np.ndarray:
    """Equal slice in every bucket."""
    if n_buckets < 1:
        raise ValueError("need at least one bucket")
    return np.full(n_buckets, 1.0 / n_buckets)


def vwap(volume_curve) -> np.ndarray:
    """Slices proportional to a volume curve.

    Pass a FORECAST curve for a tradable schedule. Passing the realised curve
    gives the oracle schedule, which uses information not available at the time.
    """
    return _normalise(volume_curve)


def almgren_chriss(kappa: float, clock) -> np.ndarray:
    """Front-loaded schedule from the Almgren-Chriss (2000) trajectory.

    Remaining holdings follow h(tau) = sinh(kappa * (1 - tau)) / sinh(kappa),
    where tau in [0, 1] is elapsed "clock". The clock is the cumulative sum of
    `clock`: pass equal weights for calendar time, or a volume curve for
    volume time.

    kappa is the urgency (kappa * T in the original paper). kappa = 0 returns
    the clock itself, so on a volume clock it nests VWAP. Larger kappa trades
    earlier. This trajectory is the exact optimum only under LINEAR temporary
    impact; here it is used as a one-parameter family of front-loaded schedules.
    """
    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    c = _normalise(clock)
    if kappa == 0:
        return c
    tau = np.cumsum(c, axis=-1)
    tau[..., -1] = 1.0  # remove float drift so holdings end at exactly zero
    tau_prev = tau - c
    tau_prev[..., 0] = 0.0

    def holdings(t):
        return np.sinh(kappa * (1.0 - t)) / np.sinh(kappa)

    w = np.clip(holdings(tau_prev) - holdings(tau), 0.0, None)
    return w / w.sum(axis=-1, keepdims=True)

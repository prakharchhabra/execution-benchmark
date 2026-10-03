"""Market impact model. These are ASSUMPTIONS, not estimates from this data.

Bar data shows prices that already happened. A simulated order does not move
them, so impact has to be modelled explicitly and swept in sensitivity tests.

Temporary impact (paid only on the slice that causes it):
    temp_k = eta * sigma * (x_k / V_k) ** beta
Permanent impact (shifts the price for all later slices), linear in size:
    perm after q shares = gamma * sigma * q / ADV

x_k   shares traded in bucket k
V_k   market volume actually traded in bucket k
sigma daily volatility (fraction, e.g. 0.02)
ADV   average daily volume in shares

Both are fractions of the arrival price. Functional form follows
Almgren, Thum, Hauptmann and Li (2005); beta = 0.5 is the square-root law.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImpactModel:
    eta: float = 0.142       # temporary impact coefficient
    beta: float = 0.5        # temporary impact exponent (0.5 = square root)
    gamma: float = 0.314     # permanent impact coefficient
    max_participation: float = 1.0  # cap on x_k / V_k


def participation(shares, bucket_volume, cap: float = 1.0):
    """Return (x_k / V_k clipped to cap, boolean mask of clipped buckets)."""
    x = np.asarray(shares, dtype=float)
    v = np.asarray(bucket_volume, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(x > 0, x / v, 0.0)
    p = np.where(np.isnan(p), 0.0, p)
    capped = p > cap
    return np.minimum(p, cap), capped


def temporary_impact(shares, bucket_volume, sigma, model: ImpactModel):
    """Per-bucket temporary impact as a fraction of price. Returns (impact, capped)."""
    p, capped = participation(shares, bucket_volume, model.max_participation)
    sig = np.asarray(sigma, dtype=float)[..., None]
    return model.eta * sig * p ** model.beta, capped


def permanent_impact(shares, adv, sigma, model: ImpactModel):
    """Average permanent impact felt by each bucket's fills, as a fraction of price.

    Fills inside bucket k see the shift from all earlier buckets plus, on
    average, half of their own bucket.
    """
    x = np.asarray(shares, dtype=float)
    cum_before = np.cumsum(x, axis=-1) - x
    q_mid = cum_before + 0.5 * x
    sig = np.asarray(sigma, dtype=float)[..., None]
    a = np.asarray(adv, dtype=float)[..., None]
    return model.gamma * sig * q_mid / a

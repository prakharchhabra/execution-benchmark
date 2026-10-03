"""Implementation shortfall: execution cost measured against the arrival price.

For a parent order filled with weights w at bucket prices P_k, arrival price P0:

    path   = side * sum_k w_k * (P_k - P0) / P0      price moved while we waited
    impact = sum_k w_k * (temp_k + perm_k)           cost we caused (always >= 0)
    IS     = path + impact

side = +1 for a buy, -1 for a sell. Positive IS is a cost. Everything is
returned in basis points of the arrival price.
"""
from __future__ import annotations

import numpy as np

BPS = 1e4


def path_cost_bps(weights, bucket_prices, arrival_price, side: int = 1):
    """Cost from price drift between arrival and each fill. Model-free."""
    if side not in (1, -1):
        raise ValueError("side must be +1 (buy) or -1 (sell)")
    w = np.asarray(weights, dtype=float)
    p = np.asarray(bucket_prices, dtype=float)
    p0 = np.asarray(arrival_price, dtype=float)[..., None]
    return side * BPS * np.sum(w * (p / p0 - 1.0), axis=-1)


def impact_cost_bps(weights, impact_frac):
    """Weighted average of per-bucket impact (fractions of price)."""
    w = np.asarray(weights, dtype=float)
    return BPS * np.sum(w * np.asarray(impact_frac, dtype=float), axis=-1)


def vwap_slippage_bps(weights, bucket_prices, bucket_volume):
    """Schedule average price versus the market interval VWAP. Model-free.

    Signed, for a buy: positive means we paid more than the market VWAP.
    """
    w = np.asarray(weights, dtype=float)
    p = np.asarray(bucket_prices, dtype=float)
    v = np.asarray(bucket_volume, dtype=float)
    r = v / v.sum(axis=-1, keepdims=True)
    market_vwap = np.sum(r * p, axis=-1)
    ours = np.sum(w * p, axis=-1)
    return BPS * (ours - market_vwap) / market_vwap


def timing_risk_theory(weights, bucket_sigma: float) -> float:
    """Standard deviation of path cost if prices follow an arithmetic random walk.

    If P_k = P0 + sum_{j<=k} e_j with sd(e_j) = bucket_sigma, then
    path = sum_j e_j * (sum_{k>=j} w_k), so
    Var = bucket_sigma^2 * sum_j (sum_{k>=j} w_k)^2.
    Used to check the simulator against a known answer.
    """
    w = np.asarray(weights, dtype=float)
    remaining_incl = np.cumsum(w[::-1])[::-1]
    return float(bucket_sigma * np.sqrt(np.sum(remaining_incl ** 2)))

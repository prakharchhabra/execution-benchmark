"""Synthetic minute bars with a KNOWN volume curve and a driftless random walk.

Used only for testing: because the truth is known, the pipeline's answers can
be checked. Output mimics the real feed (UTC timestamps, same columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES = 390


def true_minute_curve(shape: str = "u") -> np.ndarray:
    """Share of daily volume in each minute. 'u' = busy open and close, 'flat' = equal."""
    t = (np.arange(MINUTES) + 0.5) / MINUTES
    if shape == "flat":
        c = np.ones(MINUTES)
    elif shape == "u":
        c = 1.0 + 3.0 * (2.0 * t - 1.0) ** 2
    else:
        raise ValueError("shape must be 'u' or 'flat'")
    return c / c.sum()


def make_bars(symbols=("AAA", "BBB"), n_days: int = 60, seed: int = 0, shape: str = "u",
              daily_vol: float = 0.02, adv: float = 5e6, day_noise: float = 0.3,
              minute_noise: float = 0.5, start: str = "2025-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    curve = true_minute_curve(shape)
    days = pd.bdate_range(start, periods=n_days)
    minute_sd = daily_vol / np.sqrt(MINUTES)
    frames = []
    for s_idx, sym in enumerate(symbols):
        price = 50.0 * (1 + s_idx)
        for day in days:
            price *= np.exp(rng.normal(0, 0.3 * daily_vol))  # overnight gap
            rets = rng.normal(0, minute_sd, MINUTES)
            close = price * np.exp(np.cumsum(rets))
            open_ = np.concatenate([[price], close[:-1]])
            wiggle = np.abs(rng.normal(0, minute_sd / 2, MINUTES))
            high = np.maximum(open_, close) * (1 + wiggle)
            low = np.minimum(open_, close) * (1 - wiggle)
            day_total = adv * np.exp(rng.normal(0, day_noise))
            vol = day_total * curve * np.exp(rng.normal(0, minute_noise, MINUTES))
            ts = (pd.Timestamp(day.date()).tz_localize("America/New_York")
                  + pd.Timedelta(minutes=570) + pd.to_timedelta(np.arange(MINUTES), unit="min"))
            frames.append(pd.DataFrame({
                "symbol": sym,
                "timestamp": ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": open_, "high": high, "low": low, "close": close,
                "volume": np.round(vol), "vwap": (open_ + close) / 2.0,
            }))
            price = close[-1]
    return pd.DataFrame(pd.concat(frames, ignore_index=True))

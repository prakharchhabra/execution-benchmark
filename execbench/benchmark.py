"""Run every schedule on every stock-day and summarise.

What is MEASURED (model-free, from real prices and volumes):
  - volume curve forecast error, out-of-sample
  - VWAP tracking error of each schedule against the market interval VWAP
  - timing risk: the day-to-day spread of path cost
What is ASSUMED (from the impact model, not from the data):
  - impact cost in basis points
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schedules as sch
from .data import Panel, trailing_inputs
from .impact import ImpactModel, permanent_impact, temporary_impact
from .shortfall import impact_cost_bps, path_cost_bps, vwap_slippage_bps

TWAP, VWAP_F, VWAP_O = "TWAP", "VWAP forecast", "VWAP oracle*"


def total_variation(a, b) -> np.ndarray:
    """Half the L1 distance between two curves: the share of volume in the wrong bucket."""
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(axis=-1)


def run_panel(panel: Panel, sizes=(0.01, 0.05, 0.10), kappas=(1.0, 2.0, 4.0),
              model: ImpactModel = ImpactModel(), lookback: int = 20) -> pd.DataFrame:
    """One row per (day, order size, schedule) for one symbol."""
    inp = trailing_inputs(panel, lookback)
    idx = inp["valid"]
    if len(idx) == 0:
        return pd.DataFrame()
    P, V, P0 = panel.price[idx], panel.volume[idx], panel.arrival[idx]
    adv, sigma, fc = inp["adv"], inp["sigma"], inp["curve"]
    n = P.shape[1]
    realised = V / V.sum(axis=1, keepdims=True)

    weights = {TWAP: np.broadcast_to(sch.twap(n), P.shape),
               VWAP_F: fc,
               VWAP_O: realised}
    for k in kappas:
        weights[f"AC k={k:g}"] = sch.almgren_chriss(k, fc)

    tv_fc = total_variation(fc, realised)
    tv_flat = total_variation(sch.twap(n), realised)

    rows = []
    for name, W in weights.items():
        path = path_cost_bps(W, P, P0, side=1)
        slip = vwap_slippage_bps(W, P, V)
        for size in sizes:
            shares = W * (size * adv)[:, None]
            temp, capped = temporary_impact(shares, V, sigma, model)
            perm = permanent_impact(shares, adv, sigma, model)
            rows.append(pd.DataFrame({
                "symbol": panel.symbol, "date": panel.dates[idx], "size": size,
                "schedule": name, "path_bps": path, "vwap_slip_bps": slip,
                "temp_bps": impact_cost_bps(W, temp), "perm_bps": impact_cost_bps(W, perm),
                "n_capped": capped.sum(axis=1), "tv_forecast": tv_fc, "tv_flat": tv_flat,
            }))
    out = pd.concat(rows, ignore_index=True)
    out["impact_bps"] = out["temp_bps"] + out["perm_bps"]
    return out


def run_all(panels: dict[str, Panel], **kwargs) -> pd.DataFrame:
    frames = [run_panel(p, **kwargs) for p in panels.values()]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError("no stock-days with enough history; fetch more days or lower lookback")
    return pd.concat(frames, ignore_index=True)


# ---------- statistics -------------------------------------------------------

def clustered_mean_ci(df: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0):
    """Mean of `col` with a 95% CI from resampling whole DATES.

    Stocks move together on the same day, so stock-days are not independent.
    Resampling dates (not rows) keeps that dependence inside each draw.
    Returns (mean, lo, hi, number of dates).
    """
    by_date = df.groupby("date")[col].agg(["sum", "count"])
    s, c = by_date["sum"].to_numpy(), by_date["count"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(s.sum() / c.sum()), float(lo), float(hi), len(s)


def frontier_table(res: pd.DataFrame) -> pd.DataFrame:
    """Per order size and schedule: modelled impact, measured timing risk, measured drift."""
    rows = []
    for (size, name), g in res.groupby(["size", "schedule"], sort=False):
        drift, lo, hi, _ = clustered_mean_ci(g, "path_bps")
        rows.append({"size": size, "schedule": name,
                     "impact_bps": g["impact_bps"].mean(),
                     "temp_bps": g["temp_bps"].mean(), "perm_bps": g["perm_bps"].mean(),
                     "timing_risk_bps": g["path_bps"].std(ddof=1),
                     "drift_bps": drift, "drift_lo": lo, "drift_hi": hi,
                     "capped_buckets": int(g["n_capped"].sum())})
    return pd.DataFrame(rows)


def volume_forecast_table(res: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample volume curve error: trailing-mean forecast versus a flat curve."""
    one = res[(res["schedule"] == TWAP) & (res["size"] == res["size"].min())].copy()
    one["gain"] = one["tv_flat"] - one["tv_forecast"]
    gain, lo, hi, n_dates = clustered_mean_ci(one, "gain")
    flat, fcst = one["tv_flat"].mean(), one["tv_forecast"].mean()
    return pd.DataFrame([{
        "stock_days": len(one), "dates": n_dates,
        "misallocated_flat": flat, "misallocated_forecast": fcst,
        "reduction": gain, "reduction_lo": lo, "reduction_hi": hi,
        "skill": 1.0 - fcst / flat,
    }])


def capture_table(res: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """How much of the gap between TWAP and the unreachable oracle the forecast closes.

    capture = (TWAP cost - forecast VWAP cost) / (TWAP cost - oracle cost), on
    temporary impact. eta and order size cancel out of this ratio, so it depends
    only on the data and on the exponent beta. 0 = forecast is useless, 1 = perfect.
    """
    base = res[res["size"] == res["size"].min()]
    wide = base.pivot_table(index=["symbol", "date"], columns="schedule", values="temp_bps").reset_index()
    by_date = wide.groupby("date")[[TWAP, VWAP_F, VWAP_O]].sum().to_numpy()

    def ratio(m):
        t, f, o = m.sum(axis=-2).T
        return (t - f) / (t - o)

    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(by_date), size=(n_boot, len(by_date)))
    lo, hi = np.percentile(ratio(by_date[pick]), [2.5, 97.5])
    return pd.DataFrame([{"capture": float(ratio(by_date)), "capture_lo": float(lo),
                          "capture_hi": float(hi), "dates": len(by_date)}])


def tracking_table(res: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute slippage against market VWAP: TWAP versus forecast VWAP."""
    base = res[res["size"] == res["size"].min()]
    wide = base.pivot_table(index=["symbol", "date"], columns="schedule",
                            values="vwap_slip_bps").abs().reset_index()
    wide["gain"] = wide[TWAP] - wide[VWAP_F]
    gain, lo, hi, n_dates = clustered_mean_ci(wide, "gain")
    return pd.DataFrame([{
        "stock_days": len(wide), "dates": n_dates,
        "abs_slip_twap_bps": wide[TWAP].mean(), "abs_slip_vwap_bps": wide[VWAP_F].mean(),
        "reduction_bps": gain, "reduction_lo": lo, "reduction_hi": hi,
    }])

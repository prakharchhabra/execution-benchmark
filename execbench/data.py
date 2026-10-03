"""Minute bars -> intraday buckets -> point-in-time inputs.

Input bars need columns: symbol, timestamp (UTC), open, high, low, close, volume, vwap.
Only the regular US session (09:30 to 16:00 New York) is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_OPEN_MIN = 9 * 60 + 30
SESSION_CLOSE_MIN = 16 * 60
REQUIRED = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class Panel:
    """One symbol. Rows are complete trading days, columns are intraday buckets."""
    symbol: str
    dates: np.ndarray     # (D,)
    price: np.ndarray     # (D, n) bucket VWAP
    volume: np.ndarray    # (D, n) bucket volume
    arrival: np.ndarray   # (D,) first open of the session
    logret: np.ndarray    # (D,) close-to-close log return vs previous trading day
    n_dropped: int        # incomplete days removed (early closes, gaps)
    dropped_dates: tuple = ()   # which days were removed, for auditing


def load_bars(path) -> pd.DataFrame:
    """Read one CSV or every *.csv / *.csv.gz in a directory."""
    p = Path(path)
    files = sorted(list(p.glob("*.csv")) + list(p.glob("*.csv.gz"))) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no CSV files found in {p}")
    bars = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = [c for c in REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {missing}")
    return bars


def bucketize(bars: pd.DataFrame, bucket_minutes: int = 5, tz: str = "America/New_York",
              min_coverage: float = 0.95, min_late_share: float = 0.05) -> dict[str, Panel]:
    """Aggregate minute bars into intraday buckets, one Panel per symbol.

    A day is kept only if it is a full regular session:
      - the first and last buckets traded, and at least `min_coverage` of buckets traded
      - the final hour holds at least `min_late_share` of the day's volume
    The second rule removes early-close days (13:00 close). On those days liquid
    stocks still print thin after-hours trades until 16:00, so the first rule alone
    lets them through with near-empty afternoon buckets. A normal final hour holds
    roughly a fifth of the day's volume; after-hours trading holds well under 1%.
    """
    session_len = SESSION_CLOSE_MIN - SESSION_OPEN_MIN
    if session_len % bucket_minutes:
        raise ValueError("bucket_minutes must divide the 390-minute session")
    n = session_len // bucket_minutes

    # one row per symbol-minute; a re-fetched page must not double the volume
    df = bars.drop_duplicates(["symbol", "timestamp"]).reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
    minute = ts.dt.hour * 60 + ts.dt.minute
    in_session = (minute >= SESSION_OPEN_MIN) & (minute < SESSION_CLOSE_MIN)
    df, ts, minute = df[in_session].copy(), ts[in_session], minute[in_session]
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["bucket"] = (minute - SESSION_OPEN_MIN) // bucket_minutes
    df["ts"] = ts
    if "vwap" not in df.columns:
        df["vwap"] = np.nan
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = df["vwap"].fillna(typical)
    df["notional"] = df["vwap"] * df["volume"]
    df = df.sort_values(["symbol", "ts"])

    out: dict[str, Panel] = {}
    for symbol, g in df.groupby("symbol", sort=True):
        agg = g.groupby(["date", "bucket"]).agg(
            volume=("volume", "sum"), notional=("notional", "sum"), last=("close", "last"))
        vol = agg["volume"].unstack("bucket").reindex(columns=range(n)).fillna(0.0)
        notional = agg["notional"].unstack("bucket").reindex(columns=range(n)).fillna(0.0)
        last = agg["last"].unstack("bucket").reindex(columns=range(n))
        with np.errstate(divide="ignore", invalid="ignore"):
            price = notional / vol.where(vol > 0)
        # empty bucket: carry the previous bucket's last traded price forward
        price = price.fillna(last.ffill(axis=1).shift(1, axis=1)).ffill(axis=1)

        daily = g.groupby("date").agg(arrival=("open", "first"), close=("close", "last"))
        daily["logret"] = np.log(daily["close"] / daily["close"].shift(1))  # all days, incl. dropped

        v = vol.to_numpy()
        late = max(1, 60 // bucket_minutes)
        late_share = v[:, -late:].sum(axis=1) / np.maximum(v.sum(axis=1), 1e-12)
        complete = ((v[:, 0] > 0) & (v[:, -1] > 0) & ((v > 0).mean(axis=1) >= min_coverage)
                    & (late_share >= min_late_share))
        keep = vol.index[complete]
        out[symbol] = Panel(
            symbol=symbol,
            dates=keep.to_numpy(),
            price=price.loc[keep].to_numpy(dtype=float),
            volume=vol.loc[keep].to_numpy(dtype=float),
            arrival=daily.loc[keep, "arrival"].to_numpy(dtype=float),
            logret=daily.loc[keep, "logret"].to_numpy(dtype=float),
            n_dropped=int((~complete).sum()),
            dropped_dates=tuple(vol.index[~complete]),
        )
    return out


def trailing_inputs(panel: Panel, lookback: int = 20) -> dict:
    """Everything a trader would know BEFORE day d opens. No lookahead.

    For each valid day d, using days d-lookback .. d-1 only:
      adv    mean session volume
      sigma  standard deviation of daily log returns
      curve  mean share of the day's volume in each bucket (the forecast)
    Returns arrays aligned to `valid`, the indices of days with a full history.
    """
    D, n = panel.volume.shape
    first = lookback + 1 if np.isnan(panel.logret[0]) else lookback
    if D <= first:
        return {"valid": np.array([], dtype=int), "adv": np.array([]),
                "sigma": np.array([]), "curve": np.empty((0, n))}
    total = panel.volume.sum(axis=1)
    frac = panel.volume / total[:, None]
    valid = np.arange(first, D)
    adv = np.array([total[d - lookback:d].mean() for d in valid])
    sigma = np.array([panel.logret[d - lookback:d].std(ddof=1) for d in valid])
    curve = np.array([frac[d - lookback:d].mean(axis=0) for d in valid])
    return {"valid": valid, "adv": adv, "sigma": sigma, "curve": curve}

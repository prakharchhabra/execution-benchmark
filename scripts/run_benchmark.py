"""Run the benchmark on a folder of minute bars and print every result table.

    python scripts/run_benchmark.py --data data --out results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execbench.benchmark import (TWAP, VWAP_F, VWAP_O, capture_table, frontier_table,  # noqa: E402
                                 run_all, tracking_table, volume_forecast_table)
from execbench.data import bucketize, load_bars  # noqa: E402
from execbench.impact import ImpactModel  # noqa: E402

pd.set_option("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format)


def plot_frontier(frontier: pd.DataFrame, size: float, path: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface, blue = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6"
    f = frontier[frontier["size"] == size].set_index("schedule")
    family = [VWAP_F] + [s for s in f.index if s.startswith("AC")]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(f.loc[family, "timing_risk_bps"], f.loc[family, "impact_bps"], color=blue, lw=2,
            marker="o", ms=8, mec=surface, mew=2, zorder=3)
    ax.plot(f.loc[TWAP, "timing_risk_bps"], f.loc[TWAP, "impact_bps"], color=blue, marker="s",
            ms=8, mec=surface, mew=2, ls="none", zorder=3)
    ax.plot(f.loc[VWAP_O, "timing_risk_bps"], f.loc[VWAP_O, "impact_bps"], marker="o", ms=8,
            mfc=surface, mec=blue, mew=2, ls="none", zorder=3)
    offsets = {TWAP: (0, 12, "center", "bottom"), VWAP_O: (0, -12, "center", "top")}
    for name in f.index:
        dx, dy, ha, va = offsets.get(name, (10, 6, "left", "bottom"))
        text = "VWAP oracle (uses future data)" if name == VWAP_O else name
        ax.annotate(text, (f.loc[name, "timing_risk_bps"], f.loc[name, "impact_bps"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=9, color=ink)
    ax.set_xlabel("Timing risk: std of path cost across stock-days (bps), measured", color=muted, fontsize=9)
    ax.set_ylabel("Impact cost (bps), modelled", color=muted, fontsize=9)
    ax.set_title(f"Cost versus risk by schedule, order = {size:.0%} of ADV{label}",
                 loc="left", color=ink, fontsize=11, pad=12)
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.18)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bucket-minutes", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.142)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.314)
    ap.add_argument("--synthetic", action="store_true", help="label outputs as synthetic test data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    label = "  [SYNTHETIC TEST DATA]" if a.synthetic else ""

    panels = bucketize(load_bars(a.data), bucket_minutes=a.bucket_minutes)
    model = ImpactModel(eta=a.eta, beta=a.beta, gamma=a.gamma)
    res = run_all(panels, model=model, lookback=a.lookback)

    n_days = res.groupby("symbol")["date"].nunique()
    print(f"\n=== SAMPLE{label} ===")
    print(f"symbols: {len(panels)}   stock-days used: {int(n_days.sum())}   "
          f"dates: {res['date'].nunique()}   range: {res['date'].min()} to {res['date'].max()}")
    print(f"incomplete days dropped: {sum(p.n_dropped for p in panels.values())}   "
          f"bucket: {a.bucket_minutes} min   lookback: {a.lookback} days")
    dropped = pd.Series([d for p in panels.values() for d in p.dropped_dates]).value_counts().sort_index()
    for date, count in dropped.items():
        print(f"  dropped {date}: {count} of {len(panels)} symbols")
    capped = res[res["n_capped"] > 0].groupby("date")["n_capped"].sum()
    print(f"buckets where the order exceeded market volume (participation capped): {int(capped.sum())}")
    for date, count in capped.sort_values(ascending=False).head(5).items():
        print(f"  capped on {date}: {int(count)}")

    vf = volume_forecast_table(res)
    print("\n=== 1. VOLUME CURVE FORECAST, out-of-sample (measured) ===")
    print("share of the day's volume put in the wrong bucket; lower is better")
    print(vf.to_string(index=False))

    tr = tracking_table(res)
    print("\n=== 2. VWAP TRACKING ERROR (measured, no impact model) ===")
    print("mean |schedule average price - market VWAP| in bps")
    print(tr.to_string(index=False))

    fr = frontier_table(res)
    print(f"\n=== 3. COST vs RISK (impact modelled: eta={a.eta}, beta={a.beta}, gamma={a.gamma}; risk measured) ===")
    print("* oracle uses the realised volume curve: a bound, not a tradable schedule")
    print(fr.to_string(index=False))

    print("\n=== 4. SENSITIVITY: share of the TWAP-to-oracle gap closed by the forecast ===")
    print("eta and order size cancel out of this ratio when no bucket is capped; only beta matters")
    rows = []
    for beta in (0.4, 0.5, 0.6, 1.0):
        r = run_all(panels, model=ImpactModel(eta=a.eta, beta=beta, gamma=a.gamma),
                    lookback=a.lookback, sizes=(0.05,))
        c = capture_table(r).iloc[0]
        rows.append({"beta": beta, "capture": c["capture"], "capture_lo": c["capture_lo"],
                     "capture_hi": c["capture_hi"], "dates": int(c["dates"])})
    cap = pd.DataFrame(rows)
    print(cap.to_string(index=False))

    res.to_csv(out / "stock_days.csv.gz", index=False)
    for name, table in (("volume_forecast", vf), ("tracking", tr), ("frontier", fr), ("capture", cap)):
        table.to_csv(out / f"{name}.csv", index=False)
    mid = sorted(res["size"].unique())[len(res["size"].unique()) // 2]
    plot_frontier(fr, mid, out / "frontier.png", label)
    print(f"\nsaved tables and frontier.png to {out}/")


if __name__ == "__main__":
    main()

"""Write synthetic bars to data_synth/ so the pipeline can be run without an API key.

These are NOT market data. Results from them must never be quoted as findings.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execbench.synthetic import make_bars  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--shape", default="u", choices=["u", "flat"])
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bars = make_bars(("AAA", "BBB", "CCC"), n_days=a.days, seed=a.seed, shape=a.shape)
    for sym, g in bars.groupby("symbol"):
        g.to_csv(out / f"{sym}.csv.gz", index=False)
    print(f"wrote {len(bars):,} synthetic bars for 3 symbols x {a.days} days to {out}/")

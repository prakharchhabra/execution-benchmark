"""Download 1-minute US equity bars from Alpaca's free market data API.

Setup (free, paper account is enough):
  1. Sign up at alpaca.markets and generate API keys for the paper account.
  2. export APCA_API_KEY_ID=...   and   export APCA_API_SECRET_KEY=...
  3. python scripts/fetch_alpaca.py --start 2025-09-01 --end 2026-09-30

Uses feed=sip (all US exchanges). On the free plan this is allowed for data
older than 15 minutes. Do NOT switch to feed=iex: that is one small exchange
and its volume curve is not the market's volume curve.

Writes one data/<SYMBOL>.csv.gz per symbol. Already-downloaded symbols are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.alpaca.markets/v2/stocks/bars"
# Fixed in advance, before looking at any result: 12 large, liquid names across sectors.
DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,JPM,XOM,JNJ,KO,PFE,BAC,WMT,CVX"
COLUMNS = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
           "v": "volume", "vw": "vwap", "n": "trades"}


def parse_page(payload: dict, symbol: str) -> list[dict]:
    """Rows from one API response. Handles both response shapes:
    multi-symbol {"bars": {"AAPL": [...]}} and single-symbol {"bars": [...]}."""
    bars = payload.get("bars") or {}
    raw = bars.get(symbol, []) if isinstance(bars, dict) else bars
    return [{"symbol": symbol, **{COLUMNS[k]: b[k] for k in COLUMNS if k in b}} for b in raw]


def fetch_symbol(symbol: str, start: str, end: str, headers: dict, feed: str) -> pd.DataFrame:
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "timeframe": "1Min", "start": start, "end": end,
                  "limit": 10000, "adjustment": "split", "feed": feed, "sort": "asc"}
        if token:
            params["page_token"] = token
        for attempt in range(5):
            r = requests.get(URL, params=params, headers=headers, timeout=60)
            if r.status_code == 429:           # rate limit: wait and retry
                time.sleep(15 * (attempt + 1))
                continue
            break
        if r.status_code != 200:
            raise RuntimeError(f"{symbol}: HTTP {r.status_code}: {r.text[:300]}")
        payload = r.json()
        rows.extend(parse_page(payload, symbol))
        token = payload.get("next_page_token")
        if not token:
            break
        time.sleep(0.35)                       # stay under 200 requests per minute
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (must be at least a day ago)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--feed", default="sip")
    args = ap.parse_args()

    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY first (see the top of this file).")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        target = out / f"{symbol}.csv.gz"
        if target.exists():
            print(f"{symbol}: already downloaded, skipping")
            continue
        df = fetch_symbol(symbol, args.start, args.end, headers, args.feed)
        if df.empty:
            print(f"{symbol}: NO DATA returned, check the symbol and dates")
            continue
        df.to_csv(target, index=False)
        days = pd.to_datetime(df["timestamp"], utc=True).dt.date.nunique()
        print(f"{symbol}: {len(df):,} bars over {days} days -> {target}")


if __name__ == "__main__":
    main()

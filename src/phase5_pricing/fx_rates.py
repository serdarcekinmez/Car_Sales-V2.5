"""
fx_rates.py  —  USD/TRY daily bars from Yahoo Finance, fetched on demand and kept as
versioned, never-overwritten snapshots (Phase-6 contract §5.1 H4).

The model works in USD; the as-of rule that turns a bar table into a rate lives in
price_features.asof_fx_rate (open of the latest bar strictly before the auction day).
This module only obtains and stores the bars:

  fetch_usdtry(start, end)            -> FxFetch (bars, raw bytes, provenance)   [network]
  parse_yahoo_chart(raw)              -> DataFrame bar_date, bar_start_utc, open, high, low, close
  save_fx_snapshot(fetch, directory)  -> Path of the new CSV (raw JSON and provenance beside it)
  load_fx(path)                       -> validated DataFrame (input of price_features)
  compare_fx(old, new)                -> dict: overlap, changed / added / missing bars

Dates are DYNAMIC: a later fetch simply covers a longer range. Every snapshot records
the request, the fetch time and SHA-256 digests, and runners record the exact file they
used, so earlier results stay reproducible. Yahoo may revise past values; compare_fx
reports such revisions instead of hiding them.

    python3 src/phase5_pricing/fx_rates.py fetch --start 2024-11-01 [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SYMBOL = "USDTRY=X"
SOURCE = "Yahoo Finance chart API"
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "Datasets" / "external" / "fx"
COLUMNS = ["bar_date", "bar_start_utc", "open", "high", "low", "close"]


def _fail(msg: str) -> None:
    raise ValueError(f"[fx_rates] {msg}")


@dataclass(frozen=True)
class FxFetch:
    bars: pd.DataFrame
    raw: bytes
    provenance: dict


def parse_yahoo_chart(raw: bytes) -> pd.DataFrame:
    """
    Daily bars from a Yahoo chart response; bar_date is the exchange-timezone date.
    A daily bar starts at local midnight. Yahoo appends the current live quote as an
    extra row with a later timestamp; such a trailing row is dropped (attrs records it).
    A non-midnight row anywhere else, or a repeated bar date, stops the parse.
    """
    try:
        res = json.loads(raw)["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        tz = res["meta"]["exchangeTimezoneName"]
        ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"[fx_rates] not a Yahoo chart response: {e}") from None
    df = pd.DataFrame({"bar_date": ts.date,
                       "bar_start_utc": ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]})
    local = pd.Series(ts)
    midnight = ((local.dt.hour == 0) & (local.dt.minute == 0) & (local.dt.second == 0)).to_numpy()
    dropped = 0
    if len(df) and not midnight[-1]:
        df, midnight, dropped = df.iloc[:-1], midnight[:-1], 1        # trailing live quote
    if not midnight.all():
        _fail("a non-midnight row inside the series; not a plain daily-bar response")
    if pd.Series(df["bar_date"]).duplicated().any():
        _fail("the response holds repeated bar dates")
    df = df[COLUMNS].reset_index(drop=True)
    df.attrs["live_quote_rows_dropped"] = dropped
    return df


def fetch_usdtry(start, end=None, *, symbol: str = SYMBOL, timeout: int = 30) -> FxFetch:
    """Download daily bars for [start, end] (end defaults to today, UTC)."""
    start = pd.Timestamp(start).normalize()
    end = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if end is None
           else pd.Timestamp(end).normalize())
    if start > end:
        _fail("start must not be after end")
    p1 = int(start.tz_localize("UTC").timestamp())
    p2 = int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
    url = URL.format(symbol=symbol, p1=p1, p2=p2)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    bars = parse_yahoo_chart(raw)
    prov = {"source": SOURCE, "symbol": symbol, "url": url,
            "requested": [str(start.date()), str(end.date())],
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(bars), "rows_missing_open": int(bars["open"].isna().sum()),
            "live_quote_rows_dropped": bars.attrs.get("live_quote_rows_dropped", 0),
            "covered": [str(bars["bar_date"].min()), str(bars["bar_date"].max())]
            if len(bars) else None,
            "note": "Retrospective download; the provider may revise past values."}
    return FxFetch(bars=bars, raw=raw, provenance=prov)


def save_fx_snapshot(fetch: FxFetch, directory=DEFAULT_DIR) -> Path:
    """Write CSV + raw JSON + provenance under a new versioned name; never overwrites."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stamp = fetch.provenance["fetched_at_utc"].replace(":", "").replace("-", "")[:15]
    lo, hi = fetch.provenance["covered"] or ["empty", "empty"]
    base = f"USDTRY_yahoo_daily_{lo}_{hi}_fetched{stamp}"
    paths = {"csv": d / f"{base}.csv", "raw": d / f"{base}_raw.json",
             "prov": d / f"{base}_provenance.json"}
    for p in paths.values():
        if p.exists():
            _fail(f"{p} already exists; snapshots are never overwritten")
    csv = fetch.bars.to_csv(index=False).encode("utf-8")
    with open(paths["raw"], "xb") as fh:
        fh.write(fetch.raw)
    with open(paths["csv"], "xb") as fh:
        fh.write(csv)
    prov = {**fetch.provenance, "csv_sha256": hashlib.sha256(csv).hexdigest(),
            "files": {k: p.name for k, p in paths.items()}}
    with open(paths["prov"], "x", encoding="utf-8") as fh:
        fh.write(json.dumps(prov, indent=2) + "\n")
    return paths["csv"]


def load_fx(path) -> pd.DataFrame:
    """A bar CSV, validated: unique dates, positive opens where present, sorted."""
    df = pd.read_csv(path)
    missing = [c for c in ("bar_date", "open") if c not in df.columns]
    if missing:
        _fail(f"{path} lacks {missing}")
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.normalize()
    if df["bar_date"].isna().any() or df["bar_date"].duplicated().any():
        _fail(f"{path} has missing or repeated bar_date values")
    o = pd.to_numeric(df["open"], errors="coerce")
    if (o[o.notna()] <= 0).any():
        _fail(f"{path} has non-positive open rates")
    return df.sort_values("bar_date").reset_index(drop=True)


def compare_fx(old: pd.DataFrame, new: pd.DataFrame, *, rel_tol: float = 1e-6) -> dict:
    """Overlap check between two bar tables (open rates): revisions are reported, not hidden."""
    a = old.assign(bar_date=pd.to_datetime(old["bar_date"]).dt.normalize()).set_index("bar_date")["open"]
    b = new.assign(bar_date=pd.to_datetime(new["bar_date"]).dt.normalize()).set_index("bar_date")["open"]
    both = a.index.intersection(b.index)
    av, bv = a.loc[both].astype("float64"), b.loc[both].astype("float64")
    nan_change = av.isna() != bv.isna()
    rel = (bv - av).abs() / av.abs()
    changed = nan_change | (rel > rel_tol).fillna(False)
    return {"overlap_bars": int(len(both)), "changed_bars": int(changed.sum()),
            "max_rel_change": None if rel.dropna().empty else float(rel.max()),
            "changed_dates": [str(d.date()) for d in both[changed.to_numpy()]][:20],
            "only_in_old": int(len(a.index.difference(b.index))),
            "only_in_new": int(len(b.index.difference(a.index))),
            "old_range": [str(a.index.min().date()), str(a.index.max().date())],
            "new_range": [str(b.index.min().date()), str(b.index.max().date())]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="USD/TRY snapshots from Yahoo Finance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end")
    f.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    f.add_argument("--compare-with", type=Path, help="earlier CSV to compare overlapping bars")
    args = ap.parse_args(argv)
    fetch = fetch_usdtry(args.start, args.end)
    path = save_fx_snapshot(fetch, args.dir)
    out = {"saved": str(path), "provenance": fetch.provenance}
    if args.compare_with:
        out["comparison"] = compare_fx(load_fx(args.compare_with), load_fx(path))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

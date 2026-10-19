#!/usr/bin/env python3
"""
check_price_anchor.py  —  does the gap after the sales history lower Model B's prices?

  python pipeline/check_price_anchor.py [--root DIR] [--days 60] [--gap 137]

Holds out the realised sales of the last --days days, builds the price history from the
sales BEFORE them only (cut-off C), and prices each held-out sale THREE ways with the
frozen lgbm_noR, always at the sale's own USDTRY rate, comparing the prediction with the
real sale price in USD (so FX plays no role):

  normal    full history, features measured at the sale date (how the model was trained)
  gap       cut history, measured --gap days after C, as an October prediction does
            WITHOUT the anchor fix (30 / 90-day market windows empty: the old behaviour)
  anchored  cut history, measured at C + 1 day (the fix in model_b)

The held-out sales lie 1..--days days after C, like a real prediction made after the
history ends, and none of them is in the cut history.

If "gap" is clearly lower than "normal" while "anchored" stays close to it, the empty
market windows explain the low prices. Read-only; nothing is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    import numpy as np
    import pandas as pd

    from phase5_pricing import price_features as PF
    from phase5_pricing import price_model as PM
    from phase5_pricing import target_population as TP
    from phase7_inference import bundle as BD

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--days", type=int, default=60, help="last N days of sales to test")
    parser.add_argument("--gap", type=int, default=137,
                        help="days between the end of the history and the prediction date "
                             "(4 June -> 19 October 2026 = 137)")
    args = parser.parse_args(argv)

    bundle = BD.load_bundle(BD.BundleConfig(root=Path(args.root).resolve()))
    mb = bundle.model_b
    if mb is None:
        print("Model B could not be loaded:", bundle.component_errors, file=sys.stderr)
        return 2
    full, _ = TP.build_price_population(mb.clean_sales)
    last = PF.history_last_day(mb.history)
    cut = last - pd.Timedelta(days=args.days)
    held = pd.to_datetime(full["aa_tarihi"]) > cut
    cut_history = PF.build_price_history(full.loc[~held].reset_index(drop=True), mb.fx,
                                         clean_sales=mb.clean_sales)
    pop = full.loc[held].reset_index(drop=True)
    print(f"history ends {last.date()}; cut-off {cut.date()}; {len(pop)} held-out sales")

    def run(history, history_day):
        f = PF.attach_price_features(pop, mb.clean_sales, history, mb.fx,
                                     use_min_fiyat=False, use_lineup=False,
                                     history_day=history_day)
        return f, PM.predict(mb.model, f)

    base, _ = run(mb.history, None)
    actual_usd = pop["satis_tutari"].astype("float64").to_numpy() / base["fx_rate"].to_numpy()
    rows = []
    for name, hist, hday in (("normal", mb.history, None),
                             ("gap", cut_history, cut + pd.Timedelta(days=args.gap)),
                             ("anchored", cut_history, cut + pd.Timedelta(days=1))):
        f, p = run(hist, hday)
        ratio = p["pred_usd"].to_numpy() / actual_usd
        ape = np.abs(p["pred_usd"].to_numpy() - actual_usd)
        rows.append({"mode": name,
                     "median pred/actual": round(float(np.median(ratio)), 3),
                     "WAPE %": round(100 * float(ape.sum() / actual_usd.sum()), 1),
                     "market 30d missing %": round(100 * float(f["mkt_logmed_30d"].isna().mean()), 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

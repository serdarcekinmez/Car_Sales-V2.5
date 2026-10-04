"""
model_b.py  —  Phase 7: the frozen Model B price predictor served on one vehicle.

"What is this vehicle's estimated auction sale price?" The primary model is the frozen
Phase-5 lgbm_noR (no Min.Fiyat, no line-up): a price independent of the reserve. This is
a product choice, not a claim that it won Phase 6; its recorded retrospective WAPE is
about 8.9 %. The output is the estimated satis_tutari of an in-auction sale to a
non-internal buyer, CONDITIONAL on the vehicle selling, within the data's auction scope —
not a proven market-wide fair value and not a probability of sale. No interval is
returned: lgbm_R's calibrated bands belong to lgbm_R and are never transferred.

HISTORY. PriceHistory is built ONCE per bundle from the original eligible historical
sales only (target_population.build_price_population over the historical clean_sales).
A new vehicle is appended with vehicle_input.append_for_inference to a lookup-only frame;
the history is never rebuilt from it. A historical vehicle is described at its original
clean_sales position. Features use sales strictly before the prediction date.

CURRENCY. log-USD model; the rate is the open of the latest USDTRY bar dated strictly
before the prediction date, at most MAX_FX_LAG_DAYS old (price_features.asof_fx_rate).
With no admissible bar the price is UNAVAILABLE: no stale rate is substituted and nothing
is fetched.

PUBLIC API
  load_model_b(*, clean_sales, fx, model_path)    -> ModelB
  fx_status(model, day)                            -> JSON-safe dict
  estimate_price(model, rows, *, source_rows=None) -> JSON-safe dict
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phase5_pricing import price_features as PF
from phase5_pricing import price_model as PM
from phase5_pricing import target_population as TP
from phase7_inference import vehicle_input as VI

MODEL_NAME = "lgbm_noR"
LABEL = "Tahmini ihale satış fiyatı"
MEANING = ("Aracın veri kapsamındaki ihale kanalında, iç grup dışı bir alıcıya satılması "
           "KOŞULUYLA beklenen satış tutarı (Satış Tutarı). Satılma olasılığı değildir; "
           "tüm piyasa için kanıtlanmış adil değer değildir. Min.Fiyat kullanılmaz.")
REQUIRED_FIELDS = ("marka", "model", "model_yili")


def _fail(msg: str) -> None:
    raise ValueError(f"[model_b] {msg}")


@dataclass(frozen=True)
class ModelB:
    model: PM.PriceModel
    fx: pd.DataFrame
    history: PF.PriceHistory
    clean_sales: pd.DataFrame           # the historical frame the history was built from
    fit_date_range: tuple | None


def load_model_b(*, clean_sales: pd.DataFrame, fx: pd.DataFrame, model_path) -> ModelB:
    model = PM.load_price_model(model_path)
    enc = model.encoders
    if model.kind != "lgbm" or enc is None or enc.use_min_fiyat or enc.use_lineup:
        _fail(f"{MODEL_NAME} must be an lgbm model without Min.Fiyat and line-up inputs")
    pop, _ = TP.build_price_population(clean_sales)
    history = PF.build_price_history(pop, fx, clean_sales=clean_sales)
    rng = (model.provenance or {}).get("fit_date_range")
    return ModelB(model=model, fx=fx, history=history, clean_sales=clean_sales,
                  fit_date_range=tuple(rng) if rng else None)


def fx_status(model: ModelB, day: pd.Timestamp) -> dict:
    """Admissible as-of rate for `day`, or the reason it is unavailable."""
    bars = pd.to_datetime(model.fx.loc[pd.to_numeric(model.fx["open"], errors="coerce").notna(),
                                       "bar_date"])
    last = bars.max() if len(bars) else None
    out = {"rule": PF.FX_RULE, "max_lag_days": PF.MAX_FX_LAG_DAYS,
           "snapshot_last_bar": None if last is None else str(last.date())}
    try:
        r = PF.asof_fx_rate([day], model.fx)
    except ValueError:
        return {**out, "available": False, "reason": "fx_unavailable"}
    bar = pd.Timestamp(r["bar_date"].iloc[0])
    return {**out, "available": True, "rate": float(r["rate"].iloc[0]),
            "bar_date": str(bar.date()), "lag_days": int((day - bar).days)}


def estimate_price(model: ModelB, rows: pd.DataFrame, *, source_rows=None) -> dict:
    """
    One vehicle row (FROZEN_SALES_COLUMNS). source_rows=None: a new vehicle, appended to a
    lookup-only frame; otherwise the vehicle's position in the historical clean_sales
    (historical replay). Call fx_status first; this raises when no admissible rate exists.
    """
    if not isinstance(model, ModelB):
        raise TypeError("[model_b] model must be a ModelB")
    if len(rows) != 1:
        _fail("estimate_price scores exactly one vehicle")
    if source_rows is None:
        lookup, src = VI.append_for_inference(model.clean_sales, rows)
    else:
        lookup, src = model.clean_sales, np.asarray(source_rows, dtype="int64")
    population = pd.DataFrame({
        "query_id": (rows["plaka"].astype("string") + VI.KEY_SEPARATOR
                     + rows["aa_kodu"].astype("string")).to_numpy(),
        "source_row": src, "plaka": rows["plaka"].to_numpy(),
        "aa_kodu": rows["aa_kodu"].to_numpy(), "aa_tarihi": rows["aa_tarihi"].to_numpy()})
    feats = PF.attach_price_features(population, lookup, model.history, model.fx,
                                     use_min_fiyat=False, use_lineup=False)
    pred = PM.predict(model.model, feats)
    f = feats.iloc[0]
    support = {f"{lvl}_n": int(f[f"hist_{lvl}_n"]) for lvl in PF.HISTORY_LEVELS}
    support.update({f"market_median_{w}d_present": bool(np.isfinite(f[f"mkt_logmed_{w}d"]))
                    for w in PF.MARKET_WINDOWS})
    return {"price_tl": float(pred["pred_tl"].iloc[0]),
            "price_usd": float(pred["pred_usd"].iloc[0]),
            "fx_rate": float(f["fx_rate"]),
            "fx_bar_date": str(pd.Timestamp(f["fx_bar_date"]).date()),
            "history_support": support}

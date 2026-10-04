"""
price_model.py  —  Model B point predictors, selection metric and conformal intervals.

Every predictor works on log USD price, y = log(satis_tutari / fx_rate), with fx_rate the
as-of USDTRY rate of the row's prediction day (price_features.asof_fx_rate). A prediction
returns to TL with the SAME rate: pred_tl = exp(pred_log_usd) * fx_rate. Inputs are the
rows of price_features.attach_price_features; nothing here reads a file except the
artifact save / load, and nothing here knows about periods: the caller passes only the
rows it is allowed to use (fit rows to fit, calibration rows to calibrate).

CANDIDATES (one fixed configuration each; no tuning sweep)
  b0_static  train-window median log USD price by segment, finest level with >= min_n fit
             rows: marka+model+model_yili -> marka+model -> marka -> all.
  b0_asof    no fitting: the as-of history median of the row itself (hist_S2 -> S3 -> S4
             with n >= min_n, else mkt_logmed_90d). Drift-aware reference.
  lgbm       LightGBM regression on y (L2 on log price = relative error), with the
             encoders' feature list. Run once without and once with Min.Fiyat (group R).

SELECTION METRIC (proposal, pre-registered): WAPE in TL = sum|pred_tl - y_tl| / sum y_tl.
point_metrics also reports MAE, MAPE, MdAPE, share within +-10 % and median signed error.
select_candidate takes the lowest WAPE; ties go to the earlier candidate in the order given
(simplest first).

INTERVALS (split conformal, asymmetric). On calibration rows that the model never saw,
r = y - pred_log_usd. With alpha = 1 - nominal coverage and n residuals,
    q_lo = the floor((n + 1) * alpha / 2)-th smallest residual,
    q_hi = the ceil((n + 1) * (1 - alpha / 2))-th smallest residual,
and the interval is [pred + q_lo, pred + q_hi] in log USD, mapped to TL with fx_rate.
Calibration never changes the point model. Under price drift the nominal coverage is NOT
guaranteed over time; report empirical coverage and width.
GROUP-WISE (Mondrian) calibration, group_by="min_fiyat_present" (Serdar, 2026-10-02): the
same quantiles are taken separately for cars with and without a Min.Fiyat, because their
errors differ (select-window check: coverage 91 % vs 72 % with one global interval). The
grouping column is a pre-auction input. A group whose calibration rows are too few for
the coverage falls back to the global quantiles, and the fallback is recorded.

ESTIMAND. Price conditional on a completed in-auction sale to a non-internal buyer
(target_population). Nothing here estimates the probability that a car sells.

PUBLIC API
  log_usd_target(population, features)                     -> np.ndarray
  LgbmConfig, fit_lgbm(features, y, encoders, config)      -> PriceModel
  fit_static_median(features, y, *, min_n=5)                -> PriceModel
  asof_median_model(*, min_n=5)                             -> PriceModel
  predict_log_usd(model, features)                          -> np.ndarray
  calibrate(model, features, y, *, coverage=0.8, group_by=None) -> PriceModel (intervals)
  predict(model, features)                                  -> DataFrame (TL / USD / interval)
  point_metrics(y_tl, pred_tl)                              -> dict
  select_candidate(scores, order)                           -> str
  save_price_model(model, path) / load_price_model(path)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from phase5_pricing import price_features as PF

FORMAT = "phase5-price-model-v2"
CALIBRATION_GROUPS = ("min_fiyat_present",)
KINDS = ("b0_static", "b0_asof", "lgbm")
STATIC_LEVELS = (("marka", "model", "model_yili"), ("marka", "model"), ("marka",))
ASOF_LEVELS = ("S2", "S3", "S4")
ASOF_FALLBACK = "mkt_logmed_90d"
_SEP = "\x1f"


def _fail(msg: str) -> None:
    raise ValueError(f"[price_model] {msg}")


@dataclass(frozen=True)
class LgbmConfig:
    """The single reviewed LightGBM configuration (no sweep)."""
    num_boost_round: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    lambda_l2: float = 1.0
    min_data_per_group: int = 20
    cat_smooth: float = 10.0
    num_threads: int = 4
    random_state: int = 20261002


@dataclass(frozen=True)
class Calibration:
    coverage: float
    n: int
    q_lo: float                      # global quantiles (always kept)
    q_hi: float
    group_by: str | None = None
    groups: dict | None = None       # str(group value) -> {"n", "q_lo", "q_hi", "source"}


@dataclass(frozen=True)
class PriceModel:
    kind: str
    min_n: int | None = None
    static_table: dict | None = None        # level index -> {key: (median, n)}, plus "all"
    encoders: PF.PriceEncoders | None = None
    config: LgbmConfig | None = None
    model_string: str | None = None
    provenance: dict | None = None
    calibration: Calibration | None = None
    booster: object = dataclasses.field(default=None, compare=False, repr=False)


# ----------------------------------------------------------------------------
# Target.
# ----------------------------------------------------------------------------
def _check_features(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("[price_model] features must be a DataFrame")
    for c in ("query_id", "fx_rate"):
        if c not in features.columns:
            _fail(f"features lack {c}; use price_features.attach_price_features")
    r = features["fx_rate"].to_numpy(dtype="float64")
    if not (np.isfinite(r) & (r > 0)).all():
        _fail("fx_rate must be finite and > 0")


def log_usd_target(population: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    """y = log(satis_tutari / fx_rate), rows aligned by query_id (same order required)."""
    _check_features(features)
    if not (population["query_id"].astype(str).to_numpy()
            == features["query_id"].astype(str).to_numpy()).all():
        _fail("population and features must hold the same query_id in the same order")
    p = population["satis_tutari"].to_numpy(dtype="float64")
    if not (np.isfinite(p) & (p > 0)).all():
        _fail("satis_tutari must be finite and > 0")
    return np.log(p / features["fx_rate"].to_numpy(dtype="float64"))


def _check_y(y, features) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    if y.shape != (len(features),) or not np.isfinite(y).all():
        _fail("y must be a finite array aligned with features")
    return y


# ----------------------------------------------------------------------------
# Baselines.
# ----------------------------------------------------------------------------
def _key(features: pd.DataFrame, cols) -> pd.Series:
    parts = [features[c].astype("string") for c in cols]
    bad = np.zeros(len(features), dtype=bool)
    for p in parts:
        bad |= p.isna().to_numpy()
    key = parts[0].fillna("")
    for p in parts[1:]:
        key = key + _SEP + p.fillna("")
    return key.where(~bad)


def _check_min_n(min_n) -> int:
    if not isinstance(min_n, (int, np.integer)) or isinstance(min_n, bool) or min_n < 1:
        _fail("min_n must be an int >= 1")
    return int(min_n)


def fit_static_median(features: pd.DataFrame, y, *, min_n: int = 5) -> PriceModel:
    """b0_static from the given (fit) rows only."""
    _check_features(features)
    y = _check_y(y, features)
    min_n = _check_min_n(min_n)
    table = {}
    for i, cols in enumerate(STATIC_LEVELS):
        k = _key(features, cols)
        g = pd.DataFrame({"k": k, "y": y}).dropna(subset=["k"]).groupby("k")["y"]
        stats = pd.DataFrame({"m": g.median(), "n": g.size()})
        table[str(i)] = {str(key): [float(r.m), int(r.n)] for key, r in stats.iterrows()}
    table["all"] = [float(np.median(y)), int(len(y))]
    return PriceModel(kind="b0_static", min_n=min_n, static_table=table,
                      provenance={"n_fit_rows": int(len(y))})


def asof_median_model(*, min_n: int = 5) -> PriceModel:
    """b0_asof: nothing to fit; reads the row's own as-of history columns."""
    return PriceModel(kind="b0_asof", min_n=_check_min_n(min_n), provenance={"n_fit_rows": 0})


def _predict_static(model: PriceModel, features: pd.DataFrame) -> np.ndarray:
    out = np.full(len(features), np.nan)
    for i, cols in enumerate(STATIC_LEVELS):
        todo = np.isnan(out)
        if not todo.any():
            break
        lvl = model.static_table[str(i)]
        k = _key(features, cols).to_numpy(dtype=object)
        for j in np.flatnonzero(todo):
            hit = lvl.get(k[j]) if isinstance(k[j], str) else None
            if hit is not None and hit[1] >= model.min_n:
                out[j] = hit[0]
    out[np.isnan(out)] = model.static_table["all"][0]
    return out


def _predict_asof(model: PriceModel, features: pd.DataFrame) -> np.ndarray:
    need = [f"hist_{l}_{s}" for l in ASOF_LEVELS for s in ("n", "logmed")] + [ASOF_FALLBACK]
    missing = [c for c in need if c not in features.columns]
    if missing:
        _fail(f"features lack {missing}")
    out = np.full(len(features), np.nan)
    for lvl in ASOF_LEVELS:
        n = features[f"hist_{lvl}_n"].to_numpy(dtype="float64")
        m = features[f"hist_{lvl}_logmed"].to_numpy(dtype="float64")
        take = np.isnan(out) & (n >= model.min_n) & np.isfinite(m)
        out[take] = m[take]
    fb = features[ASOF_FALLBACK].to_numpy(dtype="float64")
    take = np.isnan(out)
    out[take] = fb[take]
    if np.isnan(out).any():
        _fail("b0_asof has no history at all for some rows (no market median either)")
    return out


# ----------------------------------------------------------------------------
# LightGBM.
# ----------------------------------------------------------------------------
def _check_config(config: LgbmConfig) -> LgbmConfig:
    if not isinstance(config, LgbmConfig):
        raise TypeError("[price_model] config must be an LgbmConfig")
    for f in ("num_boost_round", "num_leaves", "min_data_in_leaf", "min_data_per_group",
              "num_threads", "random_state"):
        v = getattr(config, f)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            _fail(f"config.{f} must be an int >= 1")
    for f in ("learning_rate", "lambda_l2", "cat_smooth"):
        v = getattr(config, f)
        if not isinstance(v, float) or not math.isfinite(v) or v < 0:
            _fail(f"config.{f} must be a finite float >= 0")
    return config


def _params(config: LgbmConfig) -> dict:
    return {"objective": "regression", "metric": "None",
            "learning_rate": config.learning_rate, "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf, "lambda_l2": config.lambda_l2,
            "min_data_per_group": config.min_data_per_group, "cat_smooth": config.cat_smooth,
            "bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 1.0,
            "num_threads": config.num_threads, "seed": config.random_state,
            "deterministic": True, "force_col_wise": True, "verbosity": -1}


def fit_lgbm(features: pd.DataFrame, y, encoders: PF.PriceEncoders,
             config: LgbmConfig = LgbmConfig()) -> PriceModel:
    """LightGBM on the given (fit) rows; inputs are exactly encoders.feature_names."""
    _check_features(features)
    y = _check_y(y, features)
    if not isinstance(encoders, PF.PriceEncoders):
        raise TypeError("[price_model] encoders must be PriceEncoders")
    config = _check_config(config)
    X, names = PF.encode_price_features(features, encoders)
    cat_idx = [names.index(c) for c in encoders.categorical]
    data = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=cat_idx,
                       free_raw_data=True)
    booster = lgb.train(_params(config), data, num_boost_round=config.num_boost_round)
    if booster.num_trees() != config.num_boost_round:
        _fail(f"training stopped after {booster.num_trees()} of {config.num_boost_round} rounds")
    prov = {"n_fit_rows": int(len(y)), "lightgbm_version": lgb.__version__,
            "num_trees": booster.num_trees(),
            "fit_date_range": [str(pd.to_datetime(features["aa_tarihi"]).min().date()),
                               str(pd.to_datetime(features["aa_tarihi"]).max().date())]
            if "aa_tarihi" in features.columns and len(features) else None}
    return PriceModel(kind="lgbm", encoders=encoders, config=config,
                      model_string=booster.model_to_string(), provenance=prov, booster=booster)


def _booster(model: PriceModel):
    if model.booster is None:
        object.__setattr__(model, "booster", lgb.Booster(model_str=model.model_string))
    return model.booster


# ----------------------------------------------------------------------------
# Prediction, calibration, metrics.
# ----------------------------------------------------------------------------
def _check_model(model) -> PriceModel:
    if not isinstance(model, PriceModel) or model.kind not in KINDS:
        raise TypeError("[price_model] model must be a PriceModel")
    return model


def predict_log_usd(model: PriceModel, features: pd.DataFrame) -> np.ndarray:
    """Point prediction of log USD price, one per row, row order kept."""
    model = _check_model(model)
    if not isinstance(features, pd.DataFrame):
        raise TypeError("[price_model] features must be a DataFrame")
    if model.kind == "b0_static":
        return _predict_static(model, features)
    if model.kind == "b0_asof":
        return _predict_asof(model, features)
    X, _ = PF.encode_price_features(features, model.encoders)
    if len(X) == 0:
        return np.empty(0)
    return _booster(model).predict(X, num_threads=model.config.num_threads).astype("float64")


def _order_stat(sorted_r: np.ndarray, k: int) -> float:
    """k-th smallest (1-based), clipped to the sample; beyond it the interval is infinite."""
    if k < 1:
        return -math.inf
    if k > len(sorted_r):
        return math.inf
    return float(sorted_r[k - 1])


def _quantiles(r: np.ndarray, coverage: float) -> tuple[float, float]:
    r = np.sort(r)
    n, alpha = len(r), 1.0 - coverage
    eps = 1e-9                                         # 1 - 0.8 is 0.19999..., not 0.2
    return (_order_stat(r, math.floor((n + 1) * alpha / 2 + eps)),
            _order_stat(r, math.ceil((n + 1) * (1 - alpha / 2) - eps)))


def _group_values(features: pd.DataFrame, group_by: str) -> np.ndarray:
    if group_by not in features.columns:
        _fail(f"features lack the calibration group column {group_by}")
    v = pd.to_numeric(features[group_by], errors="coerce")
    if v.isna().any():
        _fail(f"{group_by} has missing values")
    return np.array([str(int(x)) for x in v.to_numpy()], dtype=object)


def calibrate(model: PriceModel, features: pd.DataFrame, y, *,
              coverage: float = 0.8, group_by: str | None = None) -> PriceModel:
    """
    Split-conformal residual quantiles from rows the model never saw, globally and, with
    group_by, per group value. Returns a copy of the model with .calibration; the point
    model is unchanged.
    """
    model = _check_model(model)
    if model.calibration is not None:
        _fail("model is already calibrated; calibration is done once")
    if not isinstance(coverage, float) or not 0 < coverage < 1:
        _fail("coverage must be a float in (0, 1)")
    _check_features(features)
    y = _check_y(y, features)
    if len(y) == 0:
        _fail("calibration needs at least one row")
    if group_by is not None and group_by not in CALIBRATION_GROUPS:
        _fail(f"group_by must be None or one of {CALIBRATION_GROUPS}")
    r = y - predict_log_usd(model, features)
    n = len(r)
    q_lo, q_hi = _quantiles(r, coverage)
    if not (math.isfinite(q_lo) and math.isfinite(q_hi)):
        _fail(f"{n} calibration rows are too few for coverage {coverage}")
    groups = None
    if group_by is not None:
        g = _group_values(features, group_by)
        groups = {}
        for val in sorted(set(g)):
            lo, hi = _quantiles(r[g == val], coverage)
            ok = math.isfinite(lo) and math.isfinite(hi)
            groups[val] = {"n": int((g == val).sum()), "q_lo": lo if ok else q_lo,
                           "q_hi": hi if ok else q_hi, "source": "group" if ok else "global"}
    cal = Calibration(coverage=coverage, n=n, q_lo=q_lo, q_hi=q_hi, group_by=group_by,
                      groups=groups)
    return dataclasses.replace(model, calibration=cal, booster=model.booster)


def _interval_offsets(cal: Calibration, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (q_lo, q_hi); an unseen group value uses the global quantiles."""
    lo = np.full(len(features), cal.q_lo)
    hi = np.full(len(features), cal.q_hi)
    if cal.group_by is not None:
        g = _group_values(features, cal.group_by)
        for val, q in cal.groups.items():
            lo[g == val], hi[g == val] = q["q_lo"], q["q_hi"]
    return lo, hi


def predict(model: PriceModel, features: pd.DataFrame) -> pd.DataFrame:
    """query_id, pred_log_usd, pred_usd, pred_tl and, if calibrated, lo_tl / hi_tl."""
    _check_features(features)
    p = predict_log_usd(model, features)
    rate = features["fx_rate"].to_numpy(dtype="float64")
    out = pd.DataFrame({"query_id": features["query_id"].to_numpy(), "pred_log_usd": p,
                        "pred_usd": np.exp(p), "pred_tl": np.exp(p) * rate})
    if model.calibration is not None:
        lo, hi = _interval_offsets(model.calibration, features)
        out["lo_tl"] = np.exp(p + lo) * rate
        out["hi_tl"] = np.exp(p + hi) * rate
    return out


def point_metrics(y_tl, pred_tl) -> dict:
    """WAPE (selection), MAE, MAPE, MdAPE, within +-10 % and median signed % error, in TL."""
    y = np.asarray(y_tl, dtype="float64")
    p = np.asarray(pred_tl, dtype="float64")
    if y.shape != p.shape or y.ndim != 1:
        _fail("y_tl and pred_tl must be 1-D arrays of the same length")
    if len(y) == 0:
        return {"n": 0, "wape": None, "mae": None, "mape": None, "mdape": None,
                "within_10": None, "median_signed_pct": None}
    if not (np.isfinite(y).all() and (y > 0).all() and np.isfinite(p).all()):
        _fail("metrics need finite inputs and positive actual prices")
    err = p - y
    ape = np.abs(err) / y
    return {"n": int(len(y)), "wape": float(np.abs(err).sum() / y.sum()),
            "mae": float(np.abs(err).mean()), "mape": float(ape.mean()),
            "mdape": float(np.median(ape)), "within_10": float((ape <= 0.10).mean()),
            "median_signed_pct": float(np.median(err / y))}


def select_candidate(scores: dict, order) -> str:
    """Lowest WAPE; ties (exactly equal) -> earliest in `order`. Missing WAPE never wins."""
    order = list(order)
    if not order or set(order) != set(scores):
        _fail("order must list exactly the scored candidates")
    valid = [c for c in order if scores[c].get("wape") is not None
             and math.isfinite(scores[c]["wape"])]
    if not valid:
        _fail("no candidate has a finite WAPE")
    return min(valid, key=lambda c: (scores[c]["wape"], order.index(c)))


# ----------------------------------------------------------------------------
# Artifact.
# ----------------------------------------------------------------------------
def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _enc_dict(enc: PF.PriceEncoders) -> dict:
    d = dataclasses.asdict(enc)
    d["vocabulary"] = {c: list(v) for c, v in sorted(enc.vocabulary.items())}
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _enc_from(d: dict) -> PF.PriceEncoders:
    enc = PF.PriceEncoders(**{**d, "feature_names": tuple(d["feature_names"]),
                              "categorical": tuple(d["categorical"]),
                              "vocabulary": {c: tuple(v) for c, v in d["vocabulary"].items()},
                              "market_windows": tuple(d["market_windows"])})
    if (list(enc.feature_names) != PF.feature_columns(use_min_fiyat=enc.use_min_fiyat,
                                                      use_lineup=enc.use_lineup)
            or enc.history_days != PF.DEFAULT_HISTORY_DAYS
            or tuple(enc.market_windows) != PF.MARKET_WINDOWS or enc.fx_rule != PF.FX_RULE):
        _fail("stored encoders do not match this price_features version (features, history "
              "window, market windows or FX rule)")
    return enc


def _payload(model: PriceModel) -> dict:
    return {"format": FORMAT, "kind": model.kind, "min_n": model.min_n,
            "static_table": model.static_table,
            "encoders": None if model.encoders is None else _enc_dict(model.encoders),
            "config": None if model.config is None else dataclasses.asdict(model.config),
            "model_string": model.model_string,
            "model_sha256": None if model.model_string is None
            else hashlib.sha256(model.model_string.encode("utf-8")).hexdigest(),
            "provenance": model.provenance,
            "calibration": None if model.calibration is None
            else dataclasses.asdict(model.calibration)}


def save_price_model(model: PriceModel, path) -> str:
    """Canonical JSON {payload, payload_sha256}; never overwrites. Returns the digest."""
    model = _check_model(model)
    payload = _payload(model)
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    with open(Path(path), "x", encoding="utf-8") as fh:
        fh.write(_canonical({"payload": payload, "payload_sha256": digest}).decode() + "\n")
    return digest


def load_price_model(path) -> PriceModel:
    """Read an artifact and re-verify both digests; raises on any mismatch."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"payload", "payload_sha256"}:
        _fail("artifact must hold exactly payload and payload_sha256")
    p = doc["payload"]
    if hashlib.sha256(_canonical(p)).hexdigest() != doc["payload_sha256"]:
        _fail("artifact payload digest mismatch")
    if p.get("format") != FORMAT or p.get("kind") not in KINDS:
        _fail("artifact format or kind is not supported")
    ms = p["model_string"]
    if ms is not None and hashlib.sha256(ms.encode("utf-8")).hexdigest() != p["model_sha256"]:
        _fail("artifact model_string digest mismatch")
    if p["kind"] == "lgbm" and (ms is None or p["encoders"] is None or p["config"] is None):
        _fail("an lgbm artifact needs model_string, encoders and config")
    return PriceModel(kind=p["kind"], min_n=p["min_n"], static_table=p["static_table"],
                      encoders=None if p["encoders"] is None else _enc_from(p["encoders"]),
                      config=None if p["config"] is None else LgbmConfig(**p["config"]),
                      model_string=ms, provenance=p["provenance"],
                      calibration=None if p["calibration"] is None
                      else Calibration(**p["calibration"]))

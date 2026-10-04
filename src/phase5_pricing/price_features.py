"""
price_features.py  —  Model B (price) inputs at prediction time, in USD.

For every population row (target_population.build_price_population) this module builds
the inputs a price model may see at PREDICTION TIME = 00:00 on aa_tarihi, plus the as-of
USD/TRY rate used to move prices between TL and USD. It trains nothing and reads no file:
clean_sales, the population and the FX table are caller-provided DataFrames.

CURRENCY (Serdar, 2026-10-02). TL lost value quickly, so the model works in USD and
results are returned in TL. One rate per prediction day d, used for EVERY conversion of
that row (label TL->USD in price_model, prediction USD->TL, TL inputs such as tramer and
Min.Fiyat):
    rate(d) = `open` of the most recent daily USDTRY bar dated STRICTLY before d.
That bar opened during the previous day(s), so the value is published before 00:00 on d.
Weekends and holidays roll back to the last bar; bars with a missing open are skipped; a
lag over MAX_FX_LAG_DAYS raises. Same-day or later rates are never used.

ADMISSIBILITY. A feature uses only evidence dated strictly before d. Outcome fields
(price, tamamlama, buyer, sale flags) and identifiers never become inputs; see
FORBIDDEN_INPUTS, enforced on every output.

LABEL INDEPENDENCE (2026-10-03). build_price_history indexes REALISED past sales and so
requires satis_tutari finite and > 0. attach_price_features describes rows at prediction
time and needs only query_id, source_row (position in clean_sales), plaka, aa_kodu and
aa_tarihi: a vehicle not yet sold can be scored without a satis_tutari column, and a
supplied value is never read.

FEATURE GROUPS
  V  vehicle (always)  listing / inspection attributes (Model A's V1 assumption):
                       categorical VEHICLE_CATEGORICAL, numeric VEHICLE_NUMERIC.
                       tramer_tutari is converted to USD at rate(d).
  H  history (always)  log USD sale prices of POPULATION rows dated in [d - 365, d - 1]:
                       per segment level (HISTORY_LEVELS, finest to coarsest) the count,
                       the median and the days since the latest such sale, plus market
                       medians over 30 and 90 days. Each past price is converted at ITS
                       OWN rate(date). A segment with no history stays missing (n = 0);
                       nothing is backed off or invented. Excluded rows (off-auction,
                       internal, repeated keys) never enter the history.
  R  reserve           Min.Fiyat in USD (use_min_fiyat=True; Serdar allowed its use on
                       2026-10-02). A missing or <= 0 Min.Fiyat stays MISSING with
                       min_fiyat_present = 0. It is never filled from the sale price:
                       that value does not exist at prediction time (target leakage).
  L  line-up           relative_position, hat_size, aa_girisi_sira,
                       gunler_onceki_denemeden, aa_day_of_week (use_lineup=True).
                       Line-up timing and the meaning of "AA Girişi" are unconfirmed;
                       off unless a caller opts in for an ablation.

FITTED STATE. fit_price_encoders learns category vocabularies from the caller's fit rows
only (values seen >= min_count times; rarer or unseen values encode as missing) and fixes
the feature list and flags. No numeric imputation: LightGBM handles NaN. Encoders are
saved as canonical JSON with a payload SHA-256 and re-verified on load, so inference
reproduces training-time encoding exactly.

PUBLIC API
  asof_fx_rate(dates, fx)                                -> DataFrame (rate, bar_date)
  build_price_history(population, fx, *, clean_sales)   -> PriceHistory  (sold rows only)
  attach_price_features(population, clean_sales, history, fx, *, use_min_fiyat,
                        use_lineup)                      -> DataFrame     (no label needed)
  feature_columns(*, use_min_fiyat, use_lineup)          -> list[str]
  fit_price_encoders(features, *, fit_mask, use_min_fiyat, use_lineup, min_count=5)
                                                         -> PriceEncoders
  encode_price_features(features, encoders)              -> (X float64, names)
  save_price_encoders(encoders, path) / load_price_encoders(path)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT = "phase5-price-encoders-v1"
MAX_FX_LAG_DAYS = 7
DEFAULT_HISTORY_DAYS = 365
MARKET_WINDOWS = (30, 90)
FX_RULE = "open of the latest USDTRY daily bar dated strictly before aa_tarihi"

VEHICLE_CATEGORICAL = ["marka", "model", "model_uzantisi", "yakit_tipi", "vites_tipi",
                       "kasa_tipi", "renk", "lokasyon", "satici_tipi"]
VEHICLE_NUMERIC_SOURCE = ["model_yili", "car_age", "km_clean", "km_per_year",
                          "eks_puani_clean", "hasar_compound", "tramer_adedi", "tramer_tutari"]
VEHICLE_NUMERIC = ["model_yili", "car_age", "km_clean", "km_per_year", "eks_puani_clean",
                   "hasar_compound", "tramer_adedi", "tramer_tutari_usd"]
HISTORY_LEVELS = {
    "S1": ("marka", "model", "model_uzantisi", "model_yili"),
    "S2": ("marka", "model", "model_yili"),
    "S3": ("marka", "model"),
    "S4": ("marka",),
}
HISTORY_COLUMNS = [f"hist_{lvl}_{s}" for lvl in HISTORY_LEVELS for s in ("n", "logmed", "days")]
MARKET_COLUMNS = [f"mkt_logmed_{w}d" for w in MARKET_WINDOWS]
RESERVE_COLUMNS = ["min_fiyat_usd", "log_min_fiyat_usd", "min_fiyat_present"]
LINEUP_COLUMNS = ["relative_position", "hat_size", "aa_girisi_sira",
                  "gunler_onceki_denemeden", "aa_day_of_week"]
KEY_COLUMNS = ["query_id", "aa_tarihi"]
AUX_COLUMNS = ["fx_rate", "fx_bar_date"]          # conversion support, never a model input

FORBIDDEN_INPUTS = frozenset({
    "satis_tutari", "net_satis", "tamamlama", "target_value", "is_sold", "model_b_eligible",
    "sold_price_anomaly_flag", "internal_buyer_flag", "off_auction_sale_flag",
    "alici_firma_kisa_unvani", "alici_id", "split", "period", "dataset_version_id",
    "plaka", "aa_kodu", "query_id", "source_row", "sira_no", "aa_tarihi",
    "fx_rate", "fx_bar_date", "price_usd", "log_price_usd",
})

_ROW_REQUIRED = ["query_id", "source_row", "plaka", "aa_kodu", "aa_tarihi"]
_LABEL = "satis_tutari"                           # past sales only; never read for features
_SEP = "\x1f"


def _fail(msg: str) -> None:
    raise ValueError(f"[price_features] {msg}")


# ----------------------------------------------------------------------------
# FX.
# ----------------------------------------------------------------------------
def _fx_table(fx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fx, pd.DataFrame) or not {"bar_date", "open"} <= set(fx.columns):
        _fail("fx must be a DataFrame with bar_date and open")
    t = pd.DataFrame({"bar_date": pd.to_datetime(fx["bar_date"]).dt.normalize(),
                      "open": pd.to_numeric(fx["open"], errors="coerce").astype("float64")})
    if t["bar_date"].isna().any():
        _fail("fx has a missing bar_date")
    if t["bar_date"].duplicated().any():
        _fail("fx has repeated bar_date values")
    t = t.loc[t["open"].notna()].sort_values("bar_date")          # missing opens are skipped
    if (t["open"] <= 0).any():
        _fail("fx open rates must be positive")
    return t["bar_date"].to_numpy("datetime64[ns]"), t["open"].to_numpy()


def asof_fx_rate(dates, fx: pd.DataFrame) -> pd.DataFrame:
    """
    rate(d) for each date: the open of the latest bar dated strictly before d. Returns
    DataFrame(rate, bar_date) aligned with `dates`. Raises when no earlier bar exists or
    the bar is more than MAX_FX_LAG_DAYS days old.
    """
    bars, opens = _fx_table(fx)
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if d.isna().any():
        _fail("asof_fx_rate needs non-missing dates")
    pos = np.searchsorted(bars, d.to_numpy("datetime64[ns]"), side="left") - 1
    if (pos < 0).any():
        _fail(f"no USDTRY bar before {d[pos < 0].min().date()}")
    used = bars[pos]
    lag = (d.to_numpy("datetime64[ns]") - used) / np.timedelta64(1, "D")
    if (lag > MAX_FX_LAG_DAYS).any():
        _fail(f"USDTRY bar older than {MAX_FX_LAG_DAYS} days for "
              f"{d[lag > MAX_FX_LAG_DAYS].min().date()}")
    return pd.DataFrame({"rate": opens[pos], "bar_date": used})


# ----------------------------------------------------------------------------
# History.
# ----------------------------------------------------------------------------
def _day(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s).dt.normalize().to_numpy("datetime64[D]")
            .astype("int64"))


def _segment_key(frame: pd.DataFrame, cols) -> pd.Series:
    parts = [frame[c].astype("string") for c in cols]
    missing = np.zeros(len(frame), dtype=bool)
    for p in parts:
        missing |= p.isna().to_numpy() | p.fillna("").str.strip().eq("").to_numpy()
    key = parts[0].fillna("")
    for p in parts[1:]:
        key = key + _SEP + p.fillna("")
    return key.where(~missing).astype("string")


@dataclass(frozen=True)
class PriceHistory:
    """Read-only past-sale index: per level, key -> (day numbers ascending, log USD price)."""
    levels: dict
    market_days: np.ndarray
    market_logp: np.ndarray
    n_entries: int
    window_days: int = DEFAULT_HISTORY_DAYS
    source_rows: frozenset = field(default_factory=frozenset)


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _vehicle_frame(population: pd.DataFrame, clean_sales: pd.DataFrame) -> pd.DataFrame:
    """clean_sales rows of the population (by source_row), identity verified."""
    if not isinstance(clean_sales, pd.DataFrame):
        raise TypeError("[price_features] clean_sales must be a DataFrame")
    need = ["plaka", "aa_kodu", "aa_tarihi"] + VEHICLE_CATEGORICAL + VEHICLE_NUMERIC_SOURCE \
        + ["min_fiyat_clean"] + [c for c in LINEUP_COLUMNS]
    missing = [c for c in need if c not in clean_sales.columns]
    if missing:
        _fail(f"clean_sales lacks columns {missing}")
    rows = population["source_row"].to_numpy()
    if len(rows) and (rows.min() < 0 or rows.max() >= len(clean_sales)):
        _fail("population source_row is outside clean_sales; pass the SAME clean_sales frame")
    v = clean_sales.iloc[rows][need].reset_index(drop=True)
    same = (v["plaka"].astype("string").to_numpy() == population["plaka"].astype("string").to_numpy())
    same &= (v["aa_kodu"].astype("string").to_numpy() == population["aa_kodu"].astype("string").to_numpy())
    same &= (pd.to_datetime(v["aa_tarihi"]).to_numpy() == pd.to_datetime(population["aa_tarihi"]).to_numpy())
    if not same.all():
        _fail("clean_sales rows do not match the population (plaka / aa_kodu / aa_tarihi); "
              "pass the SAME clean_sales frame the population was built from")
    return v


def _check_rows(population: pd.DataFrame) -> None:
    """Keys and dates of the rows to describe; no outcome field is required."""
    if not isinstance(population, pd.DataFrame):
        raise TypeError("[price_features] population must be a DataFrame")
    missing = [c for c in _ROW_REQUIRED if c not in population.columns]
    if missing:
        _fail(f"population lacks columns {missing}")
    if population["query_id"].duplicated().any():
        _fail("population query_id must be unique")
    if population["aa_tarihi"].isna().any():
        _fail("population has a missing aa_tarihi")


def _check_sold(population: pd.DataFrame) -> None:
    """Past sales indexed as history must carry a realised, positive sale price."""
    _check_rows(population)
    if _LABEL not in population.columns:
        _fail(f"population lacks columns {[_LABEL]}")
    p = pd.to_numeric(population[_LABEL], errors="coerce")
    if not (np.isfinite(p) & (p > 0)).all():
        _fail("population satis_tutari must be finite and > 0")


def build_price_history(population: pd.DataFrame, fx: pd.DataFrame, *,
                        clean_sales: pd.DataFrame) -> PriceHistory:
    """
    Index the population's past sales as log USD prices (each at its own rate(date)).
    Segment keys come from clean_sales via source_row. Only population rows enter, and
    each must be a realised sale (satis_tutari finite and > 0).
    """
    _check_sold(population)
    v = _vehicle_frame(population, clean_sales)
    days = _day(population["aa_tarihi"])
    rate = asof_fx_rate(population["aa_tarihi"], fx)["rate"].to_numpy()
    logp = np.log(population["satis_tutari"].astype("float64").to_numpy() / rate)
    levels = {}
    for lvl, cols in HISTORY_LEVELS.items():
        key = _segment_key(v, cols)
        ok = key.notna().to_numpy()
        frame = pd.DataFrame({"k": key[ok].astype(str).to_numpy(), "d": days[ok], "p": logp[ok]})
        frame = frame.sort_values(["k", "d", "p"], kind="mergesort")
        idx = {}
        for k, g in frame.groupby("k", sort=False):
            idx[k] = (_readonly(g["d"].to_numpy()), _readonly(g["p"].to_numpy()))
        levels[lvl] = idx
    order = np.lexsort((logp, days))
    return PriceHistory(levels=levels, market_days=_readonly(days[order]),
                        market_logp=_readonly(logp[order]), n_entries=len(population),
                        source_rows=frozenset(population["source_row"].astype(int).tolist()))


def _window_stats(d_arr: np.ndarray, p_arr: np.ndarray, day: int, window: int):
    lo = np.searchsorted(d_arr, day - window, side="left")
    hi = np.searchsorted(d_arr, day, side="left")             # strictly before `day`
    n = hi - lo
    if n == 0:
        return 0, np.nan, np.nan
    return n, float(np.median(p_arr[lo:hi])), float(day - d_arr[hi - 1])


def _history_features(v: pd.DataFrame, days: np.ndarray, history: PriceHistory) -> pd.DataFrame:
    out = {}
    W = history.window_days
    for lvl, cols in HISTORY_LEVELS.items():
        key = _segment_key(v, cols)
        n = np.zeros(len(v), dtype="int64")
        med = np.full(len(v), np.nan)
        since = np.full(len(v), np.nan)
        pairs = pd.DataFrame({"k": key, "d": days}).dropna(subset=["k"])
        for (k, d), rows in pairs.groupby(["k", "d"], sort=False).groups.items():
            entry = history.levels[lvl].get(str(k))
            if entry is None:
                continue
            res = _window_stats(entry[0], entry[1], int(d), W)
            n[rows], med[rows], since[rows] = res
        out[f"hist_{lvl}_n"], out[f"hist_{lvl}_logmed"], out[f"hist_{lvl}_days"] = n, med, since
    for w in MARKET_WINDOWS:
        col = np.full(len(v), np.nan)
        for d in np.unique(days):
            col[days == d] = _window_stats(history.market_days, history.market_logp, int(d), w)[1]
        out[f"mkt_logmed_{w}d"] = col
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Features.
# ----------------------------------------------------------------------------
def feature_columns(*, use_min_fiyat: bool, use_lineup: bool) -> list[str]:
    cols = VEHICLE_CATEGORICAL + VEHICLE_NUMERIC + HISTORY_COLUMNS + MARKET_COLUMNS
    if use_min_fiyat:
        cols += RESERVE_COLUMNS
    if use_lineup:
        cols += LINEUP_COLUMNS
    return cols


def _check_flags(use_min_fiyat, use_lineup) -> None:
    for name, v in (("use_min_fiyat", use_min_fiyat), ("use_lineup", use_lineup)):
        if not isinstance(v, (bool, np.bool_)):
            raise TypeError(f"[price_features] {name} must be a bool")


def _assert_no_forbidden(names) -> None:
    bad = sorted(set(names) & FORBIDDEN_INPUTS)
    if bad:
        _fail(f"forbidden inputs {bad} in the feature list")


def attach_price_features(population: pd.DataFrame, clean_sales: pd.DataFrame,
                          history: PriceHistory, fx: pd.DataFrame, *,
                          use_min_fiyat: bool, use_lineup: bool) -> pd.DataFrame:
    """
    One row per population row, same order: KEY_COLUMNS, AUX_COLUMNS (fx_rate, fx_bar_date)
    and feature_columns(...). Works on any subset of rows with identical values.

    `population` needs only query_id, source_row, plaka, aa_kodu and aa_tarihi; source_row
    is the row's position in `clean_sales`. satis_tutari is neither required nor read, so
    an unsold vehicle can be described; if the column is present its value (missing, 0 or
    any other) does not change the output.
    """
    _check_flags(use_min_fiyat, use_lineup)
    _check_rows(population)
    if not isinstance(history, PriceHistory):
        raise TypeError("[price_features] history must be a PriceHistory")
    v = _vehicle_frame(population, clean_sales)
    days = _day(population["aa_tarihi"])
    fxr = asof_fx_rate(population["aa_tarihi"], fx)
    rate = fxr["rate"].to_numpy()
    out = pd.DataFrame({"query_id": population["query_id"].astype("string").to_numpy(),
                        "aa_tarihi": pd.to_datetime(population["aa_tarihi"]).to_numpy(),
                        "fx_rate": rate, "fx_bar_date": fxr["bar_date"].to_numpy()})
    for c in VEHICLE_CATEGORICAL:
        out[c] = v[c].astype("string").str.strip().replace("", pd.NA).to_numpy()
        out[c] = out[c].astype("string")
    for c in VEHICLE_NUMERIC_SOURCE:
        num = pd.to_numeric(v[c], errors="coerce").astype("float64").to_numpy()
        if c == "tramer_tutari":
            out["tramer_tutari_usd"] = num / rate
        else:
            out[c] = num
    hist = _history_features(v, days, history)
    out = pd.concat([out, hist], axis=1)
    if use_min_fiyat:
        mf = pd.to_numeric(v["min_fiyat_clean"], errors="coerce").astype("float64").to_numpy()
        present = np.isfinite(mf) & (mf > 0)
        usd = np.where(present, mf / rate, np.nan)
        out["min_fiyat_usd"] = usd
        out["log_min_fiyat_usd"] = np.log(usd)
        out["min_fiyat_present"] = present.astype("int8")
    if use_lineup:
        for c in LINEUP_COLUMNS:
            out[c] = pd.to_numeric(v[c], errors="coerce").astype("float64").to_numpy()
    cols = KEY_COLUMNS + AUX_COLUMNS + feature_columns(use_min_fiyat=use_min_fiyat,
                                                      use_lineup=use_lineup)
    _assert_no_forbidden(feature_columns(use_min_fiyat=use_min_fiyat, use_lineup=use_lineup))
    return out[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Encoders.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceEncoders:
    feature_names: tuple
    categorical: tuple
    vocabulary: dict                 # column -> tuple of kept values (code = position)
    use_min_fiyat: bool
    use_lineup: bool
    min_count: int
    n_fit_rows: int
    history_days: int = DEFAULT_HISTORY_DAYS
    market_windows: tuple = MARKET_WINDOWS
    fx_rule: str = FX_RULE


def fit_price_encoders(features: pd.DataFrame, *, fit_mask, use_min_fiyat: bool,
                       use_lineup: bool, min_count: int = 5) -> PriceEncoders:
    """Vocabularies from the fit rows only (count >= min_count); fixes names and flags."""
    _check_flags(use_min_fiyat, use_lineup)
    if not isinstance(min_count, (int, np.integer)) or isinstance(min_count, bool) or min_count < 1:
        _fail("min_count must be an int >= 1")
    names = feature_columns(use_min_fiyat=use_min_fiyat, use_lineup=use_lineup)
    _assert_no_forbidden(names)
    missing = [c for c in names if c not in features.columns]
    if missing:
        _fail(f"features lack {missing} (were they attached with the same flags?)")
    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != (len(features),):
        _fail("fit_mask must be a boolean array aligned with features")
    if not mask.any():
        _fail("fit_mask selects no rows")
    fit = features.loc[mask]
    vocab = {}
    for c in VEHICLE_CATEGORICAL:
        counts = fit[c].dropna().astype(str).value_counts()
        vocab[c] = tuple(sorted(counts[counts >= min_count].index.tolist()))
    return PriceEncoders(feature_names=tuple(names), categorical=tuple(VEHICLE_CATEGORICAL),
                         vocabulary=vocab, use_min_fiyat=bool(use_min_fiyat),
                         use_lineup=bool(use_lineup), min_count=int(min_count),
                         n_fit_rows=int(mask.sum()))


def encode_price_features(features: pd.DataFrame,
                          encoders: PriceEncoders) -> tuple[np.ndarray, list[str]]:
    """float64 matrix in encoders.feature_names order; categories -> codes, unknown -> NaN."""
    if not isinstance(encoders, PriceEncoders):
        raise TypeError("[price_features] encoders must be PriceEncoders")
    missing = [c for c in encoders.feature_names if c not in features.columns]
    if missing:
        _fail(f"features lack {missing}")
    X = np.empty((len(features), len(encoders.feature_names)), dtype="float64")
    for j, c in enumerate(encoders.feature_names):
        if c in encoders.vocabulary:
            mapping = {v: i for i, v in enumerate(encoders.vocabulary[c])}
            X[:, j] = features[c].astype("string").map(mapping).astype("float64").to_numpy()
        else:
            X[:, j] = pd.to_numeric(features[c], errors="coerce").astype("float64").to_numpy()
    return X, list(encoders.feature_names)


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _payload(enc: PriceEncoders) -> dict:
    return {"format": FORMAT, "feature_names": list(enc.feature_names),
            "categorical": list(enc.categorical),
            "vocabulary": {c: list(v) for c, v in sorted(enc.vocabulary.items())},
            "use_min_fiyat": enc.use_min_fiyat, "use_lineup": enc.use_lineup,
            "min_count": enc.min_count, "n_fit_rows": enc.n_fit_rows,
            "history_days": enc.history_days, "market_windows": list(enc.market_windows),
            "fx_rule": enc.fx_rule}


def save_price_encoders(encoders: PriceEncoders, path) -> str:
    """Canonical JSON {payload, payload_sha256}; never overwrites. Returns the digest."""
    if not isinstance(encoders, PriceEncoders):
        raise TypeError("[price_features] encoders must be PriceEncoders")
    payload = _payload(encoders)
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    with open(Path(path), "x", encoding="utf-8") as fh:
        fh.write(_canonical({"payload": payload, "payload_sha256": digest}).decode() + "\n")
    return digest


def load_price_encoders(path) -> PriceEncoders:
    """Read and re-verify an encoder file; raises on any digest or contract mismatch."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"payload", "payload_sha256"}:
        _fail("encoder file must hold exactly payload and payload_sha256")
    p = doc["payload"]
    if hashlib.sha256(_canonical(p)).hexdigest() != doc["payload_sha256"]:
        _fail("encoder payload digest mismatch")
    if p.get("format") != FORMAT:
        _fail(f"encoder format {p.get('format')!r} is not {FORMAT!r}")
    enc = PriceEncoders(feature_names=tuple(p["feature_names"]),
                        categorical=tuple(p["categorical"]),
                        vocabulary={c: tuple(v) for c, v in p["vocabulary"].items()},
                        use_min_fiyat=p["use_min_fiyat"], use_lineup=p["use_lineup"],
                        min_count=p["min_count"], n_fit_rows=p["n_fit_rows"],
                        history_days=p["history_days"],
                        market_windows=tuple(p["market_windows"]), fx_rule=p["fx_rule"])
    expected = feature_columns(use_min_fiyat=enc.use_min_fiyat, use_lineup=enc.use_lineup)
    if list(enc.feature_names) != expected or enc.history_days != DEFAULT_HISTORY_DAYS \
            or enc.market_windows != MARKET_WINDOWS or enc.fx_rule != FX_RULE:
        _fail("encoder contract differs from this module version")
    _assert_no_forbidden(enc.feature_names)
    return enc

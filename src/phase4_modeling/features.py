"""
features.py  —  the first Phase-4 (Model A) feature set for selected (query, firm) KEYS.

For every key row emitted by query_universe.iter_training_keys / iter_evaluation_keys it
answers: "what did we know about this car, this firm, and this firm's taste for cars like
this one, strictly before the auction day?" It trains no model, fixes no category
vocabulary, performs NO file I/O and WRITES NOTHING. Features are computed only for the
keys passed in; no date-wide car-by-firm table is built.

SOURCES (all caller-provided DataFrames; nothing is loaded here)
  bids       clean_reel_db: query_id, firma_adi, aa_tarihi, bid_timestamp. firma_adi is
             folded with ascii_key_series, exactly as candidate_pool does.
  positives  the Phase-2 grade >= 1 (query_id, firma_key, grade) pairs — the same input
             query_universe.reconcile_positives takes. Used ONLY for the grade-2 (win) flag
             of past engagements; it must match the bid engagements one to one.
  vehicles   clean_sales: plaka, aa_kodu, aa_tarihi and the vehicle source columns below.
  event_index / exclude
             the accepted Phase-3 EventIndex and exclusions — the SAME ones used to
             generate the keys. firm_* pool features come only from candidate_pool_as_of.

CUTOFF. as_of is the key's aa_tarihi (auction-day midnight). Evidence is admissible only
if its effective time is STRICTLY BEFORE it, as in candidate_pool.py. An ENGAGEMENT is one
(query, firm) pair with at least one bid. Its effective time is
    e_rec = max over the firm's bids on that car of max(aa_tarihi, bid_timestamp),
so an engagement — its count, its vehicle and its win flag — is admitted only once ALL its
bids are admissible. The key's own car and every car of the same auction day are therefore
never used. (candidate_pool counts single bids; the two differ only for late-stamped bids.)

AVAILABILITY ASSUMPTIONS (reviewed by Serdar 2026-09-28)
  V1 accepted  vehicle attributes in the sales export equal the listing at auction time.
  V3 accepted  an auction win (grade 2) is known by the end of the auction day.
  V4 accepted  days_since_known inherits the unverified directory-date assumption.
  V2 unknown   whether Min.Fiyat is set before the auction. min_fiyat_clean is therefore
               NOT a Model-A input. Model-B note: price fields are to be used there after a
               USD conversion at that day's rate; nothing price-related is built here.

FEATURES (FEATURE_COLUMNS is the model-input ALLOWLIST; every other column is excluded)
  veh_*   the key's car, from its single sales row. Constant within a query: they matter
          for ranking only through interactions with firm features.
            veh_marka, veh_kasa_tipi, veh_lokasyon   categorical, kept as strings
            veh_car_age, veh_hasar_compound          as cleaned in Phase 1
            veh_km                                   km_clean; <= 0 or > KM_MAX -> NA
  firm_*  the firm's history before as_of.
            firm_days_since_known, firm_days_since_last_bid,
            firm_days_since_last_participation, firm_total_bids_prior,
            firm_is_active_365d                      copied from candidate_pool_as_of
            firm_n_cars_bid_prior                    admitted engagements
            firm_n_wins_prior                        admitted engagements with grade 2
            firm_win_rate_prior                      wins / cars
  aff_*   the firm's past engagements compared with the key's car.
            aff_marka_n_prior                        past cars of the same marka
            aff_marka_share, aff_kasa_share,
            aff_lokasyon_share                       same value / past cars with a known value
            aff_age_gap                              |car_age - mean car_age of past cars|

MISSING VALUES. Nothing is imputed. Counts are 0 when there is no history (a true zero).
days_since_* are NA when that evidence never existed (the pool's convention). A share is NA
when its denominator is 0 or the key's own value is NA; aff_marka_n_prior is NA when the
key's marka is NA. Continuous features use NaN. Categorical features stay strings with NA;
their encoding vocabulary is a training decision (fit period), not made here.

VEHICLE ROWS. A key's query must have exactly one sales row — quarantined duplicate listings
and orphan queries never reach the key streams, and a violation raises. For PAST
engagements, a duplicate listing contributes each attribute only where its rows agree
(otherwise NA), and an orphan query (no sales row) contributes NA attributes: it still
counts as an engagement but enters no share's numerator or denominator.

EXCLUDED FROM MODEL INPUTS: identifiers and ledger fields (query_id, plaka, aa_kodu,
aa_tarihi, firma_key, split, period, dataset_version_id), labels and label audit (grade,
label_provenance), pool metadata (lane, known_via, known_since), query status and counts,
every outcome-side sales column, and the present-day registry snapshot attributes.
attach_features reads only query_id, aa_tarihi and firma_key from a key chunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from common.normalize import ascii_key_series
# The accepted strict date parser; reused so bids are dated exactly as in the pool.
from phase3_firm_universe.candidate_pool import (EventIndex, _check_exclude, _to_ns,
                                                 candidate_pool_as_of)
from phase4_modeling.query_universe import (KEY_DTYPES, POSITIVE_COLUMNS, _dates, _grades,
                                            _strings)

KM_MAX = 1_000_000                       # same bound as anomaly_flags.km_flags

BID_COLUMNS = ["query_id", "firma_adi", "aa_tarihi", "bid_timestamp"]
CATEGORICAL_SOURCES = ["marka", "kasa_tipi", "lokasyon"]
NUMERIC_SOURCES = ["car_age", "km_clean", "hasar_compound"]
VEHICLE_COLUMNS = ["plaka", "aa_kodu", "aa_tarihi", *CATEGORICAL_SOURCES, *NUMERIC_SOURCES]
KEY_INPUT_COLUMNS = ["query_id", "aa_tarihi", "firma_key"]

# pool column -> feature column, copied unchanged from candidate_pool_as_of.
POOL_FEATURES = {
    "days_since_known": "firm_days_since_known",
    "days_since_last_bid": "firm_days_since_last_bid",
    "days_since_last_participation": "firm_days_since_last_participation",
    "total_bids_prior": "firm_total_bids_prior",
    "is_active_365d": "firm_is_active_365d",
}

FEATURE_DTYPES = {
    "veh_marka": "string", "veh_kasa_tipi": "string", "veh_lokasyon": "string",
    "veh_car_age": "float64", "veh_km": "float64", "veh_hasar_compound": "float64",
    "firm_days_since_known": "Int32", "firm_days_since_last_bid": "Int32",
    "firm_days_since_last_participation": "Int32", "firm_total_bids_prior": "Int32",
    "firm_is_active_365d": "bool",
    "firm_n_cars_bid_prior": "int32", "firm_n_wins_prior": "int32",
    "firm_win_rate_prior": "float64",
    "aff_marka_n_prior": "Int32", "aff_marka_share": "float64", "aff_kasa_share": "float64",
    "aff_lokasyon_share": "float64", "aff_age_gap": "float64",
}
FEATURE_COLUMNS = list(FEATURE_DTYPES)
OUTPUT_DTYPES = {**KEY_DTYPES, **FEATURE_DTYPES}

_SHARES = {"marka": "aff_marka_share", "kasa_tipi": "aff_kasa_share",
           "lokasyon": "aff_lokasyon_share"}


@dataclass(frozen=True)
class _FirmHistory:
    """
    One firm's engagements sorted by e_rec. Prefix arrays have length m + 1, so the value
    at hi = searchsorted(times, as_of) covers exactly the admitted engagements.
    packed[f] holds code * (m + 1) + position for engagements whose field f is known.
    """
    times: np.ndarray
    wins: np.ndarray
    age_sum: np.ndarray
    age_n: np.ndarray
    known: tuple[np.ndarray, ...]
    packed: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FeatureIndex:
    """Vehicle table per query_id plus per-firm engagement histories. Build once, read only."""
    vehicles: pd.DataFrame                  # index query_id; n_sales_rows, aa_tarihi, veh_*, codes
    firms: dict[str, _FirmHistory]


# ----------------------------------------------------------------------------
# Input checks — fail loudly; identifiers are checked, never normalized.
# ----------------------------------------------------------------------------
def _fail(message: str) -> None:
    raise ValueError(f"[features] {message}")


def _require_frame(df: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[features] {name} must be a pandas DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(f"{name} is missing required columns: {missing}")


def _category(s: pd.Series, name: str) -> pd.Series:
    """String-or-null column; blank strings become NA. Values are not re-normalized."""
    vals = s.astype(object).to_numpy()
    present = ~pd.isna(vals)
    if not all(isinstance(v, str) for v in vals[present]):
        _fail(f"{name} holds non-string values")
    out = pd.Series(pd.array(vals, dtype="string"))
    return out.mask(out.str.strip().eq("").fillna(False))


def _number(s: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_numeric_dtype(s.dtype):
        _fail(f"{name} must be numeric, got {s.dtype}")
    return s.astype("float64").to_numpy()


def _vehicle_table(vehicles: pd.DataFrame) -> pd.DataFrame:
    """
    One row per query_id = plaka + '::' + aa_kodu. Each attribute is kept only where all
    of the query's sales rows agree (all-NA agrees); otherwise it is NA. The auction date
    must agree. Categorical codes (-1 = NA) share one vocabulary across all queries.
    """
    _require_frame(vehicles, VEHICLE_COLUMNS, "vehicles")
    plaka = _strings(vehicles["plaka"], "vehicles.plaka")
    code = _strings(vehicles["aa_kodu"], "vehicles.aa_kodu")
    frame = pd.DataFrame({
        "query_id": pd.array([p + "::" + c for p, c in zip(plaka, code)], dtype="string"),
        "aa_tarihi": _dates(vehicles["aa_tarihi"], "vehicles.aa_tarihi"),
    })
    for col in CATEGORICAL_SOURCES:
        frame[f"veh_{col}"] = _category(vehicles[col], f"vehicles.{col}").array
    km = _number(vehicles["km_clean"], "vehicles.km_clean")
    frame["veh_km"] = np.where((km <= 0) | (km > KM_MAX), np.nan, km)
    frame["veh_car_age"] = _number(vehicles["car_age"], "vehicles.car_age")
    frame["veh_hasar_compound"] = _number(vehicles["hasar_compound"], "vehicles.hasar_compound")

    frame = frame.sort_values("query_id", kind="mergesort").reset_index(drop=True)
    grouped = frame.groupby("query_id", sort=True)
    first = frame.drop_duplicates("query_id").set_index("query_id")
    if (grouped["aa_tarihi"].nunique() != 1).any():
        _fail("a query's sales rows disagree on aa_tarihi")
    table = pd.DataFrame(index=first.index)
    table["n_sales_rows"] = grouped.size().to_numpy()
    table["aa_tarihi"] = first["aa_tarihi"]
    for col in ["veh_marka", "veh_kasa_tipi", "veh_lokasyon",
                "veh_car_age", "veh_km", "veh_hasar_compound"]:
        agree = grouped[col].nunique(dropna=False).to_numpy() == 1
        table[col] = first[col].where(agree)
    for col in CATEGORICAL_SOURCES:
        table[f"code_{col}"] = pd.factorize(table[f"veh_{col}"], use_na_sentinel=True)[0]
    return table


def _engagements(bids: pd.DataFrame, positives: pd.DataFrame,
                 table: pd.DataFrame) -> pd.DataFrame:
    """One row per (query_id, firma_key) with e_rec, the win flag and the car's attributes."""
    _require_frame(bids, BID_COLUMNS, "bids")
    _require_frame(positives, POSITIVE_COLUMNS, "positives")
    keys = ascii_key_series(bids["firma_adi"])
    keep = keys.ne("").to_numpy()                       # blank firm names name no firm
    qid = _strings(bids["query_id"][keep], "bids.query_id")
    day = _to_ns(bids["aa_tarihi"][keep], "bids.aa_tarihi", allow_null=False,
                 require_midnight=True)
    stamp = _to_ns(bids["bid_timestamp"][keep], "bids.bid_timestamp", allow_null=False)
    raw = pd.DataFrame({"query_id": qid, "firma_key": keys[keep].to_numpy(dtype=object),
                        "day": day, "e": np.maximum(day, stamp)})
    days = raw.groupby("query_id")["day"].agg(["min", "max"])
    if (days["min"] != days["max"]).any():
        _fail("a query's bids disagree on aa_tarihi")
    at = table.index.get_indexer(days.index)
    known = at >= 0
    if (table["aa_tarihi"].to_numpy()[at[known]].view("int64")
            != days["min"].to_numpy()[known]).any():
        _fail("bid aa_tarihi disagrees with the sales aa_tarihi of the same query")

    rec = raw.groupby(["query_id", "firma_key"], sort=True)["e"].max().reset_index()
    pos = pd.DataFrame({
        "query_id": _strings(positives["query_id"], "positives.query_id"),
        "firma_key": _strings(positives["firma_key"], "positives.firma_key"),
        "grade": _grades(positives["grade"], "positives.grade"),
    })
    if pos.duplicated(["query_id", "firma_key"]).any():
        _fail("positives repeat a (query_id, firma_key) pair")
    both = rec.merge(pos, on=["query_id", "firma_key"], how="outer", indicator=True)
    lost = both["_merge"].ne("both")
    if lost.any():
        _fail(f"{int(lost.sum())} bid engagements and positives do not match one to one, "
              f"e.g. {both.loc[lost, ['query_id', 'firma_key', '_merge']].head(3).values.tolist()}")
    rec = both.drop(columns="_merge")
    rec["win"] = rec["grade"].to_numpy() == 2

    at = table.index.get_indexer(rec["query_id"])
    has = at >= 0                                          # orphan queries have no sales row
    for col in CATEGORICAL_SOURCES:
        rec[f"code_{col}"] = np.where(has, table[f"code_{col}"].to_numpy()[at], -1)
    rec["car_age"] = np.where(has, table["veh_car_age"].to_numpy()[at], np.nan)
    return rec


def _firm_history(g: pd.DataFrame) -> _FirmHistory:
    g = g.sort_values(["e", "query_id"], kind="mergesort")
    m = len(g)
    age = g["car_age"].to_numpy(dtype=float)
    has_age = ~np.isnan(age)

    def prefix(values: np.ndarray) -> np.ndarray:
        out = np.concatenate([[0], np.cumsum(values)])
        out.setflags(write=False)
        return out
    known, packed = [], []
    positions = np.arange(m, dtype=np.int64)
    for col in CATEGORICAL_SOURCES:
        codes = g[f"code_{col}"].to_numpy(dtype=np.int64)
        ok = codes >= 0
        p = np.sort(codes[ok] * (m + 1) + positions[ok])
        p.setflags(write=False)
        known.append(prefix(ok.astype(np.int64)))
        packed.append(p)
    times = g["e"].to_numpy(dtype=np.int64, copy=True)
    times.setflags(write=False)
    return _FirmHistory(times=times, wins=prefix(g["win"].to_numpy(dtype=np.int64)),
                        age_sum=prefix(np.where(has_age, age, 0.0)),
                        age_n=prefix(has_age.astype(np.int64)),
                        known=tuple(known), packed=tuple(packed))


# ----------------------------------------------------------------------------
# Public interface.
# ----------------------------------------------------------------------------
def build_feature_index(bids: pd.DataFrame,
                        positives: pd.DataFrame,
                        vehicles: pd.DataFrame) -> FeatureIndex:
    """
    Build the reusable, read-only feature index from the whole history. Nothing here is
    filtered by date: admissibility is decided per key in attach_features. Raises when bid
    engagements and positives do not match one to one, or when dates disagree.
    """
    table = _vehicle_table(vehicles)
    rec = _engagements(bids, positives, table)
    firms = {str(k): _firm_history(g) for k, g in rec.groupby("firma_key", sort=True)}
    return FeatureIndex(vehicles=table, firms=firms)


def extend_feature_index(feature_index: FeatureIndex,
                         vehicles: pd.DataFrame) -> FeatureIndex:
    """
    Inference helper (Phase 7). A NEW FeatureIndex whose vehicle table is the historical
    table followed by the queries of `vehicles` — cars not auctioned yet — and whose firm
    histories are the historical ones, shared unchanged. Nothing is rebuilt and the given
    index is not modified. `vehicles` has the VEHICLE_COLUMNS of build_feature_index.

    A query_id already in the index raises: a historical car is scored with the original
    index, never attached twice. Categories reuse the historical value -> code mapping
    exactly (the packed firm histories are keyed by those codes). A value never seen in
    the history gets a fresh code above every historical one, so it matches no past
    engagement (share 0 over known past cars, aff_marka_n_prior 0 — as for a first-ever
    value in training). NA stays -1.
    """
    if not isinstance(feature_index, FeatureIndex):
        raise TypeError("[features] feature_index must be a FeatureIndex")
    old = feature_index.vehicles
    new = _vehicle_table(vehicles)
    clash = new.index[new.index.isin(old.index)]
    if len(clash):
        _fail(f"vehicles repeat query_ids already in the index, e.g. {list(clash[:3])}; "
              f"score historical cars with the original index")
    for col in CATEGORICAL_SOURCES:
        codes = old[f"code_{col}"].to_numpy(dtype=np.int64)
        known = codes >= 0
        pairs = pd.DataFrame({"v": old[f"veh_{col}"].to_numpy(dtype=object)[known],
                              "c": codes[known]}).drop_duplicates()
        if not (pairs["v"].is_unique and pairs["c"].is_unique):
            _fail(f"the index's {col} codes are not one value per code")
        vals = new[f"veh_{col}"].to_numpy(dtype=object)
        present = new[f"veh_{col}"].notna().to_numpy()
        at = np.full(len(new), -1, dtype=np.int64)
        at[present] = pd.Index(pairs["v"].to_numpy(dtype=object)).get_indexer(vals[present])
        out = np.full(len(new), -1, dtype=np.int64)
        seen = at >= 0
        out[seen] = pairs["c"].to_numpy(dtype=np.int64)[at[seen]]
        unseen = present & ~seen
        if unseen.any():
            start = int(codes.max()) + 1 if len(codes) else 0
            _, inverse = np.unique(vals[unseen].astype(str), return_inverse=True)
            out[unseen] = start + inverse
        new[f"code_{col}"] = out
    table = pd.concat([old, new[old.columns]])
    return FeatureIndex(vehicles=table, firms=feature_index.firms)


def attach_features(feature_index: FeatureIndex,
                    event_index: EventIndex,
                    keys: pd.DataFrame,
                    *,
                    exclude: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Return `keys` unchanged — same rows, order, index and columns — followed by
    FEATURE_COLUMNS with FEATURE_DTYPES. Only query_id, aa_tarihi and firma_key are read.
    Pass the SAME event_index and exclude used to generate the keys: a key firm missing
    from its date's pool raises. Works on any chunk; results do not depend on chunking.
    """
    if not isinstance(feature_index, FeatureIndex):
        raise TypeError("[features] feature_index must be a FeatureIndex")
    if not isinstance(event_index, EventIndex):
        raise TypeError("[features] event_index must be a candidate_pool.EventIndex")
    _require_frame(keys, KEY_INPUT_COLUMNS, "keys")
    clash = [c for c in FEATURE_COLUMNS if c in keys.columns]
    if clash:
        _fail(f"keys already hold feature columns: {clash}")
    excluded = _check_exclude(exclude)
    qid = _strings(keys["query_id"], "keys.query_id")
    firm = _strings(keys["firma_key"], "keys.firma_key")
    dates = _dates(keys["aa_tarihi"], "keys.aa_tarihi")
    if pd.DataFrame({"q": qid, "f": firm}).duplicated().any():
        _fail("keys repeat a (query_id, firma_key) pair")
    n = len(keys)

    # --- vehicle: the key's single sales row ---
    table = feature_index.vehicles
    at = table.index.get_indexer(qid)
    if (at < 0).any():
        _fail(f"keys name queries without a sales row, e.g. {qid[at < 0][:3].tolist()}")
    if (table["n_sales_rows"].to_numpy()[at] != 1).any():
        _fail("keys name a query with several sales rows; quarantined queries have no keys")
    if (table["aa_tarihi"].to_numpy()[at] != dates).any():
        _fail("keys.aa_tarihi disagrees with the sales aa_tarihi of the query")
    veh = table.iloc[at]
    out = {c: veh[c].to_numpy() for c in ["veh_marka", "veh_kasa_tipi", "veh_lokasyon",
                                          "veh_car_age", "veh_km", "veh_hasar_compound"]}

    # --- firm history from the accepted pool, one pool per date ---
    pool_cols = {new: np.empty(n, dtype=object) for new in POOL_FEATURES.values()}
    for day in np.unique(dates):
        rows = np.flatnonzero(dates == day)
        pool = candidate_pool_as_of(event_index, pd.Timestamp(day), exclude=excluded)
        where = pd.Index(pool["firma_key"]).get_indexer(firm[rows])
        if (where < 0).any():
            _fail(f"key firms absent from the {pd.Timestamp(day).date()} pool, e.g. "
                  f"{firm[rows][where < 0][:3].tolist()}; pass the event_index and exclude "
                  f"used to generate the keys")
        for src, new in POOL_FEATURES.items():
            pool_cols[new][rows] = pool[src].to_numpy(dtype=object)[where]

    # --- engagement history and affinity ---
    t = dates.view("int64")
    qcodes = [veh[f"code_{c}"].to_numpy(dtype=np.int64) for c in CATEGORICAL_SOURCES]
    q_age = veh["veh_car_age"].to_numpy(dtype=float)
    n_cars = np.zeros(n, np.int64)
    n_wins = np.zeros(n, np.int64)
    age_sum = np.zeros(n)
    age_n = np.zeros(n, np.int64)
    n_same = [np.zeros(n, np.int64) for _ in CATEGORICAL_SOURCES]
    n_known = [np.zeros(n, np.int64) for _ in CATEGORICAL_SOURCES]
    for key, pos in pd.Series(np.arange(n)).groupby(firm, sort=False).indices.items():
        h = feature_index.firms.get(key)
        if h is None:
            continue                                   # no engagement ever: zeros
        hi = np.searchsorted(h.times, t[pos], side="left")    # strictly before as_of
        n_cars[pos] = hi
        n_wins[pos] = h.wins[hi]
        age_sum[pos] = h.age_sum[hi]
        age_n[pos] = h.age_n[hi]
        stride = len(h.times) + 1
        for f in range(len(CATEGORICAL_SOURCES)):
            q = qcodes[f][pos]
            same = (np.searchsorted(h.packed[f], q * stride + hi, side="left")
                    - np.searchsorted(h.packed[f], q * stride, side="left"))
            n_same[f][pos] = np.where(q >= 0, same, 0)
            n_known[f][pos] = h.known[f][hi]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["firm_win_rate_prior"] = np.where(n_cars > 0, n_wins / n_cars, np.nan)
        mean_age = np.where(age_n > 0, age_sum / age_n, np.nan)
        out["aff_age_gap"] = np.abs(q_age - mean_age)
        for f, col in enumerate(CATEGORICAL_SOURCES):
            valid = (qcodes[f] >= 0) & (n_known[f] > 0)
            out[_SHARES[col]] = np.where(valid, n_same[f] / n_known[f], np.nan)
    out["firm_n_cars_bid_prior"] = n_cars
    out["firm_n_wins_prior"] = n_wins
    out["aff_marka_n_prior"] = pd.arrays.IntegerArray(n_same[0].astype("int32"), qcodes[0] < 0)
    out.update(pool_cols)

    features = pd.DataFrame({c: pd.Series(out[c], index=keys.index).astype(d)
                             for c, d in FEATURE_DTYPES.items()})
    return pd.concat([keys.copy(), features], axis=1)

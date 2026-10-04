"""
vehicle_input.py  —  Phase 7.1: pre-auction vehicle input -> clean_sales-compatible rows.

Phase 7 serves the frozen models on vehicles that have not been auctioned yet. The models
read their inputs through Phase-5 price_features, which looks a vehicle up by POSITION
(source_row) in a clean_sales frame. This module turns a caller's vehicle table into rows
of exactly that frame. It trains nothing, predicts nothing, reads / writes no file and does
nothing at import time: every input is a caller-provided object.

ONE DERIVATION. Vehicle fields go through phase1_preparation.clean_sales.build_clean_sales
unchanged (plate cleanup, model year -> car_age, KM / Eks.Puanı / Tramer / Min.Fiyat
parsing, Ekspertiz -> hasar_compound, seller classification), called with
date_window = (prediction_date, prediction_date) so the historical constants stay as they
are. No cleaning formula is copied here; a Phase-1 change reaches inference automatically.

INPUT SCHEMA (RAW_COLUMNS: exact Turkish export names, every one optional)
  Plaka, Marka, Model, Model Uzantısı, Model Yılı, KM, Eks.Puanı, Ekspertiz, Tramer Adedi,
  Tramer Tutarı, Min.Fiyat, Satıcı Kısa Ünvan, Yakıt Tipi, Vites Tipi, Kasa Tipi, Renk,
  Lokasyon.
  Any other column raises. Outcome columns (Alıcı*, Satış Tutarı / Durumu / Tipi, AA Esnası
  Tamamlama, Şehir = the BUYER's city, and normalized names such as satis_tutari, is_sold,
  model_b_eligible) raise by name: a result is never silently consumed. AA Kodu, AA Tarihi,
  Sıra and AA Girişi raise as well: the adapter supplies the keys and the date, and no
  lineup evidence is accepted in this version.
  A missing column or value stays MISSING. Min.Fiyat is optional (lgbm_noR is the planned
  primary price model) and is never filled from any sale value.

OUTPUT of build_inference_vehicles: FROZEN_SALES_COLUMNS in order, one row per input row in
input order, RangeIndex, split = "inference", dataset_version_id = the caller's version_id.
  keys        plaka = the supplied plate after Phase-1 standardize_plaka, or the synthetic
              token SYN<request>R<i> when none is given (i = 0-based input position). The
              token starts with letters, so plaka_format_valid is honestly 0; no real plate
              is ever invented. aa_kodu = INF<request>N<i>; aa_tarihi = prediction_date.
              query_id = plaka + "::" + aa_kodu is unique inside a request; a repeated
              normalized plate raises. Same request_id + same input order = same keys.
  date facts  aa_day_of_week, aa_month and car_age follow from the prediction date.
  lineup      sira_no, relative_position, hat_size, is_early_in_line, aa_girisi_sira and
              gunler_onceki_denemeden are MISSING: Phase 1 would derive them from this
              batch as if it were one auction line, which it is not.
  outcomes    satis_tutari, net_satis, tamamlama, alici_*, internal_buyer_flag,
              off_auction_sale_flag and sold_price_anomaly_flag are MISSING. Phase 1 needs
              placeholders for the raw buyer / sale columns; their results are redacted.
  sentinels   is_sold = model_b_eligible = 0 ONLY so that an inference row can never enter
              a training population. They are NOT a prediction that the vehicle won't sell.
  dtypes      redacted int8 flags become nullable Int8, hat_size nullable Int64, redacted
              numbers float64 NaN and redacted text string <NA>.

INHERITED PHASE-1 LIMITS (reported by describe_input, not repaired)
  * Missing Ekspertiz text parses to hasar_compound = 0. A zero score is NOT evidence of an
    undamaged vehicle.
  * A missing / unmatched seller name is classified 'diger' (Phase-1 fallback). 'diger' is
    NOT evidence of the seller type.
  * A missing Min.Fiyat sets min_fiyat_was_zero = 1 ("reserve not set"), as in training.

HISTORICAL REPLAY (from_history). Past vehicles are used at their EXISTING clean_sales
positions (source_rows); they are never appended again, so they cannot become duplicate
history. Model-facing rows keep the original features, keys and date; their outcome
columns are redacted and returned separately as recorded outcomes.

APPEND (append_for_inference). extended = clean_sales followed by the inference rows;
source_rows are the appended positions, exactly what price_features expects in
population.source_row. extended is a FEATURE-LOOKUP frame only: price history must be built
from the historical population alone (build_price_history over the original clean_sales),
never from extended. With two dataset_version_ids and missing flags it is not a valid
training population, and target_population's validation rejects it. Concatenation may
promote historical int8 / int64 columns to the nullable Int8 / Int64 of the redacted
columns; values are preserved, byte-identical dtypes / serialization are not promised.

PUBLIC API
  build_inference_vehicles(raw, *, prediction_date, uye, version_id, request_id) -> rows
  from_history(clean_sales, query_ids)    -> (rows, source_rows, recorded_outcomes)
  append_for_inference(clean_sales, rows) -> (extended, source_rows)
  describe_input(rows, *, raw=None)       -> JSON-safe support / missing-field summary
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from common.normalize import ascii_key, ascii_key_series, standardize_plaka
from phase1_preparation.clean_sales import FROZEN_SALES_COLUMNS, build_clean_sales

SPLIT = "inference"
KEY_SEPARATOR = "::"                  # query_id = plaka + "::" + aa_kodu (Phase 2 / Phase 5)
SYNTHETIC_PLATE_PREFIX = "SYN"
SYNTHETIC_CODE_PREFIX = "INF"
WAWASAKY = "WAWASAKY"                 # same marker as pipeline/run_phase6_eval.py

RAW_COLUMNS = ("Plaka", "Marka", "Model", "Model Uzantısı", "Model Yılı", "KM", "Eks.Puanı",
               "Ekspertiz", "Tramer Adedi", "Tramer Tutarı", "Min.Fiyat", "Satıcı Kısa Ünvan",
               "Yakıt Tipi", "Vites Tipi", "Kasa Tipi", "Renk", "Lokasyon")

# Raw columns build_clean_sales reads that a pre-auction input must not carry. They enter
# Phase 1 as all-missing placeholders and everything derived from them is redacted.
_PHASE1_PLACEHOLDERS = ("Sıra", "AA Girişi", "Satış Tutarı", "Alıcı Firma Kısa Ünvanı",
                        "Alıcı Id", "Satış Tipi")

# Column-name screens on ascii_key(name) (Turkish-folded, upper-case, A-Z0-9 only).
_OUTCOME_PREFIXES = ("ALICI",)                      # Alıcı Firma Kısa Ünvanı, Alıcı Id, ...
_OUTCOME_PARTS = ("SATIS", "TAMAMLAMA")             # Satış Tutarı / Durumu / Tipi, net_satis
_OUTCOME_KEYS = frozenset({
    "SEHIR", "ISSOLD", "MODELBELIGIBLE", "INTERNALBUYERFLAG", "OFFAUCTIONSALEFLAG",
    "SOLDPRICEANOMALYFLAG", "TARGETVALUE", "LABEL", "GRADE"})
_ADAPTER_KEYS = frozenset({
    "AAKODU", "AATARIHI", "SIRA", "AAGIRISI", "SIRANO", "RELATIVEPOSITION", "HATSIZE",
    "ISEARLYINLINE", "AAGIRISISIRA", "GUNLERONCEKIDENEMEDEN", "SPLIT", "DATASETVERSIONID",
    "QUERYID", "SOURCEROW"})

# Redacted columns -> output dtype (float64 holds NaN; the nullable dtypes hold <NA>).
REDACTED_OUTCOME_COLUMNS = {
    "satis_tutari": "float64", "net_satis": "float64", "tamamlama": "float64",
    "alici_firma_kisa_unvani": "string", "alici_id": "float64",
    "internal_buyer_flag": "Int8", "off_auction_sale_flag": "Int8",
    "sold_price_anomaly_flag": "Int8",
}
UNKNOWN_LINEUP_COLUMNS = {
    "sira_no": "float64", "relative_position": "float64", "hat_size": "Int64",
    "is_early_in_line": "Int8", "aa_girisi_sira": "float64",
    "gunler_onceki_denemeden": "float64",
}
SENTINEL_COLUMNS = ("is_sold", "model_b_eligible")  # 0 = "never training-eligible" only

_VEHICLE_FIELDS = ["marka", "model", "model_uzantisi", "model_yili", "car_age", "km_clean",
                   "eks_puani_clean", "tramer_adedi", "tramer_tutari", "min_fiyat_clean",
                   "yakit_tipi", "vites_tipi", "kasa_tipi", "renk", "lokasyon"]
_REQUEST_RE = re.compile(r"[A-Za-z0-9]{1,64}")

LIMITS = (
    "hasar_compound = 0 is not evidence of an undamaged vehicle: missing or unparseable "
    "Ekspertiz text also scores 0 (Phase 1).",
    "satici_tipi 'diger' is the Phase-1 fallback for missing / unmatched seller names, not "
    "evidence of the seller type.",
    "is_sold = model_b_eligible = 0 are sentinels that keep rows out of training; they do not "
    "predict that the vehicle will not sell.",
    "Inference rows carry no lineup facts (sira_no, hat_size, relative_position, "
    "is_early_in_line, aa_girisi_sira, gunler_onceki_denemeden); use a no-lineup model.",
    "Min.Fiyat is optional and never filled from a sale value; an absent value has "
    "min_fiyat_was_zero = 1 (Phase 1 'reserve not set').",
    "WAWASAKY rows stay in the cohort; the marker is neither an exclusion nor proof of the "
    "vehicle type.",
)


def _fail(msg: str) -> None:
    raise ValueError(f"[vehicle_input] {msg}")


def _sample(values, k: int = 5) -> list[str]:
    return [str(v) for v in list(values)[:k]]


# ----------------------------------------------------------------------------
# Validation helpers.
# ----------------------------------------------------------------------------
def _check_raw(raw: pd.DataFrame) -> None:
    """Explicit allow-list: outcome, adapter-owned and unknown columns all raise."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("[vehicle_input] raw must be a DataFrame")
    cols = list(raw.columns)
    if not all(isinstance(c, str) for c in cols):
        _fail("raw column names must be strings")
    repeated = pd.Series(cols, dtype="object")
    repeated = repeated[repeated.duplicated()].unique()
    if len(repeated):
        _fail(f"raw has repeated column names {_sample(repeated)}")
    allowed = {ascii_key(c): c for c in RAW_COLUMNS}
    outcome, adapter, unknown = [], [], []
    for c in cols:
        if c in RAW_COLUMNS:
            continue
        k = ascii_key(c)
        if (k.startswith(_OUTCOME_PREFIXES) or any(p in k for p in _OUTCOME_PARTS)
                or k in _OUTCOME_KEYS):
            outcome.append(c)
        elif k in _ADAPTER_KEYS:
            adapter.append(c)
        else:
            unknown.append(f"{c} (expected {allowed[k]!r})" if k in allowed else c)
    if outcome:
        _fail(f"raw carries outcome / buyer / label columns {outcome}; a pre-auction input "
              "must not contain the result")
    if adapter:
        _fail(f"raw carries {adapter}; keys and date are supplied by the adapter and lineup "
              "evidence is not accepted in this version")
    if unknown:
        _fail(f"raw has unknown columns {unknown}; allowed: {list(RAW_COLUMNS)}")


def _prediction_day(value) -> pd.Timestamp:
    """A present, timezone-naive whole date."""
    if value is None or isinstance(value, (bool, int, float, np.number)):
        raise TypeError("[vehicle_input] prediction_date must be a date, datetime, Timestamp "
                        "or date string")
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[vehicle_input] prediction_date {value!r} is not a date") from exc
    if pd.isna(ts):
        _fail("prediction_date is missing")
    if ts.tzinfo is not None:
        _fail("prediction_date must be timezone-naive")
    if ts != ts.normalize():
        _fail(f"prediction_date must be a whole date (midnight), got {ts}")
    return ts


def _request_token(request_id) -> str:
    """Upper-cased ASCII alphanumeric namespace for the synthetic keys."""
    if not isinstance(request_id, str):
        raise TypeError("[vehicle_input] request_id must be a string")
    token = request_id.strip()
    if not _REQUEST_RE.fullmatch(token):
        _fail("request_id must be 1-64 ASCII letters / digits")
    return token.upper()


def _check_version(version_id) -> None:
    if not isinstance(version_id, str) or not version_id.strip() \
            or version_id != version_id.strip():
        _fail("version_id must be a non-blank string without surrounding whitespace")


def _check_schema(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"[vehicle_input] {name} must be a DataFrame")
    if list(frame.columns) != FROZEN_SALES_COLUMNS:
        _fail(f"{name} must have exactly FROZEN_SALES_COLUMNS in order")
    if not pd.api.types.is_datetime64_any_dtype(frame["aa_tarihi"]):
        _fail(f"{name}.aa_tarihi must be a datetime column")
    if getattr(frame["aa_tarihi"].dt, "tz", None) is not None:
        _fail(f"{name}.aa_tarihi must be timezone-naive")


def _blank(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return s.isna() | s.fillna("").str.strip().eq("")


def _query_ids(frame: pd.DataFrame) -> pd.Series:
    """plaka + '::' + aa_kodu by position; <NA> where either key is blank (never matches)."""
    plaka = frame["plaka"].astype("string").reset_index(drop=True)
    kodu = frame["aa_kodu"].astype("string").reset_index(drop=True)
    valid = ~(_blank(plaka) | _blank(kodu))
    return (plaka + KEY_SEPARATOR + kodu).where(valid).astype("string")


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()


def _redact(frame: pd.DataFrame, spec: dict) -> None:
    """Make every column in spec missing, in place on a frame this module owns."""
    for c, dtype in spec.items():
        if dtype == "float64":
            frame[c] = np.full(len(frame), np.nan)
        else:
            frame[c] = pd.Series(pd.NA, index=frame.index, dtype=dtype)


def _set_sentinels(frame: pd.DataFrame) -> None:
    for c in SENTINEL_COLUMNS:
        frame[c] = np.zeros(len(frame), dtype="int8")


def _empty_internal_lookup() -> pd.DataFrame:
    """flag_internal's lookup shape with no group: the buyer is a redacted placeholder."""
    return pd.DataFrame({"group": pd.Series(dtype="string"),
                         "match_pattern": pd.Series(dtype="string")})


# ----------------------------------------------------------------------------
# Public API.
# ----------------------------------------------------------------------------
def build_inference_vehicles(raw: pd.DataFrame, *, prediction_date, uye: pd.DataFrame | None,
                             version_id: str, request_id: str) -> pd.DataFrame:
    """
    Clean pre-auction vehicles for prediction on prediction_date. raw uses RAW_COLUMNS
    (all optional). uye is the in-memory Üye-Kurumsal frame for seller classification, or
    None (regex fallback only). Returns FROZEN_SALES_COLUMNS rows in input order; see the
    module docstring for keys, redaction and sentinels. The input is not modified.
    """
    _check_raw(raw)
    day = _prediction_day(prediction_date)
    req = _request_token(request_id)
    _check_version(version_id)
    if uye is not None and not isinstance(uye, pd.DataFrame):
        raise TypeError("[vehicle_input] uye must be a DataFrame or None")

    n = len(raw)
    idx = pd.RangeIndex(n)
    satis = pd.DataFrame(index=idx)
    for c in RAW_COLUMNS:
        satis[c] = (raw[c].reset_index(drop=True) if c in raw.columns
                    else pd.Series(np.nan, index=idx, dtype="float64"))

    # A plate that standardizes to "" counts as not supplied and gets a synthetic token.
    supplied, _ = standardize_plaka(satis["Plaka"])
    has_plate = supplied.ne("").to_numpy(dtype=bool)
    synthetic = pd.Series([f"{SYNTHETIC_PLATE_PREFIX}{req}R{i}" for i in range(n)],
                          index=idx, dtype="string")
    satis["Plaka"] = supplied.where(has_plate, synthetic)
    satis["AA Kodu"] = pd.Series([f"{SYNTHETIC_CODE_PREFIX}{req}N{i}" for i in range(n)],
                                 index=idx, dtype="string")
    satis["AA Tarihi"] = pd.Series(day, index=idx, dtype="datetime64[ns]")
    for c in _PHASE1_PLACEHOLDERS:
        satis[c] = pd.Series(np.nan, index=idx, dtype="float64")

    flags: list[dict] = []          # Phase-1 audit trail; not part of the inference output
    log: list[dict] = []
    out, _ = build_clean_sales(satis, uye, _empty_internal_lookup(), version_id, flags, log,
                               date_window=(day, day))
    out = out.reset_index(drop=True)

    _redact(out, REDACTED_OUTCOME_COLUMNS)
    _redact(out, UNKNOWN_LINEUP_COLUMNS)
    _set_sentinels(out)
    out["split"] = pd.Series(SPLIT, index=out.index, dtype="string")

    if (out.loc[~has_plate, "plaka_format_valid"] != 0).any():
        _fail("internal error: a synthetic plate passed the plate format check")
    if (out["aa_tarihi_invalid_flag"] != 0).any():
        _fail("internal error: the prediction date fell outside its own date window")
    repeated = out["plaka"].duplicated(keep=False)
    if repeated.any():
        _fail(f"repeated normalized plate(s) in one request: "
              f"{_sample(out.loc[repeated, 'plaka'].unique())}")
    if _query_ids(out).duplicated().any():
        _fail("internal error: repeated query_id in one request")
    return out[FROZEN_SALES_COLUMNS]


def from_history(clean_sales: pd.DataFrame, query_ids
                 ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Historical replay of the requested query_ids, in the requested order.
    Returns (rows, source_rows, recorded_outcomes):
      rows               FROZEN_SALES_COLUMNS copies with original features, keys, date,
                         split and version; outcome columns redacted, sentinels 0.
      source_rows        int64 positions of those rows in clean_sales. Use them with the
                         ORIGINAL clean_sales; never append rows (append_for_inference
                         rejects them).
      recorded_outcomes  query_id, source_row, aa_tarihi and the original outcome values.
    A requested id that is missing, blank-keyed, repeated in the request or occurs on more
    than one clean_sales row raises; no row is ever picked arbitrarily.
    """
    _check_schema(clean_sales, "clean_sales")
    if isinstance(query_ids, (str, bytes)) or not isinstance(
            query_ids, (Sequence, pd.Series, pd.Index, np.ndarray)):
        raise TypeError("[vehicle_input] query_ids must be a list-like of query_id strings")
    ids = list(query_ids)
    bad = [q for q in ids if not isinstance(q, str) or not q.strip()]
    if bad:
        _fail(f"query_ids must be non-blank strings, got {_sample(bad)}")
    requested = pd.Series(ids, dtype="object")
    if requested.duplicated().any():
        _fail(f"query_ids repeats {_sample(requested[requested.duplicated()].unique())}")

    qid = _query_ids(clean_sales)
    found = pd.DataFrame({"query_id": qid.to_numpy(dtype=object),
                          "pos": np.arange(len(clean_sales), dtype="int64")})
    found = found.loc[qid.notna().to_numpy()]
    counts = found["query_id"].value_counts()
    missing = [q for q in ids if counts.get(q, 0) == 0]
    if missing:
        _fail(f"{len(missing)} query_id(s) not in clean_sales, e.g. {_sample(missing)}")
    ambiguous = [q for q in ids if counts.get(q, 0) > 1]
    if ambiguous:
        _fail(f"{len(ambiguous)} query_id(s) occur on more than one clean_sales row, "
              f"e.g. {_sample(ambiguous)}")
    pos = found.set_index("query_id")["pos"]
    source_rows = pos.loc[ids].to_numpy(dtype="int64") if ids else np.empty(0, dtype="int64")

    rows = clean_sales.iloc[source_rows][FROZEN_SALES_COLUMNS].reset_index(drop=True).copy()
    outcome_cols = list(REDACTED_OUTCOME_COLUMNS) + list(SENTINEL_COLUMNS)
    outcomes = rows[["aa_tarihi"] + outcome_cols].copy()
    outcomes.insert(0, "source_row", source_rows)
    outcomes.insert(0, "query_id", pd.Series(ids, dtype="string"))
    _redact(rows, REDACTED_OUTCOME_COLUMNS)
    _set_sentinels(rows)
    return rows, source_rows, outcomes


def append_for_inference(clean_sales: pd.DataFrame, rows: pd.DataFrame
                         ) -> tuple[pd.DataFrame, np.ndarray]:
    """
    (extended, source_rows): the historical clean_sales followed by inference rows from
    build_inference_vehicles, as a NEW frame with a RangeIndex; neither input is modified.
    source_rows are the int64 positions of the appended rows, for price_features'
    population.source_row. extended is for feature lookup only; build price history from the
    historical population (see module docstring). Raises when clean_sales already holds
    inference rows, when rows are not redacted single-version inference rows, or when a
    query_id or the version id collides with clean_sales.
    """
    _check_schema(clean_sales, "clean_sales")
    _check_schema(rows, "rows")
    if clean_sales["split"].astype("string").fillna("").eq(SPLIT).any():
        _fail("clean_sales already holds inference rows; append to the historical "
              "clean_sales only")
    if len(rows) == 0:
        return clean_sales.reset_index(drop=True).copy(), np.empty(0, dtype="int64")

    if not rows["split"].astype("string").fillna("").eq(SPLIT).all():
        _fail(f"rows must all have split == {SPLIT!r} (historical rows are replayed at their "
              "source positions via from_history, never appended)")
    versions = rows["dataset_version_id"].astype("string")
    if _blank(versions).any() or versions.nunique() != 1:
        _fail("rows must carry exactly one non-blank dataset_version_id")
    version = versions.iloc[0]
    if clean_sales["dataset_version_id"].astype("string").fillna("").eq(version).any():
        _fail(f"rows' dataset_version_id {version!r} is already used by clean_sales")
    for c in SENTINEL_COLUMNS:
        v = pd.to_numeric(rows[c], errors="coerce")
        if v.isna().any() or not v.eq(0).all():
            _fail(f"rows.{c} must be the 0 sentinel")
    carried = [c for c in REDACTED_OUTCOME_COLUMNS if rows[c].notna().any()]
    if carried:
        _fail(f"rows carry outcome values in {carried}")
    if rows["aa_tarihi"].isna().any():
        _fail("rows have a missing aa_tarihi")

    new_q = _query_ids(rows)
    if new_q.isna().any():
        _fail("rows have a blank plaka or aa_kodu")
    if new_q.duplicated().any():
        _fail(f"rows repeat query_id {_sample(new_q[new_q.duplicated()].unique())}")
    clash = new_q.isin(_query_ids(clean_sales).dropna().unique())
    if clash.any():
        _fail(f"{int(clash.sum())} row query_id(s) already exist in clean_sales, "
              f"e.g. {_sample(new_q[clash])}; use a new request_id")

    extended = pd.concat([clean_sales.reset_index(drop=True), rows.reset_index(drop=True)],
                         ignore_index=True)
    drift = [c for c in FROZEN_SALES_COLUMNS
             if extended[c].dtype == object and clean_sales[c].dtype != object]
    if drift:
        _fail(f"appending turned {drift} into object columns; rows and clean_sales dtypes "
              "are incompatible")
    source_rows = np.arange(len(clean_sales), len(extended), dtype="int64")
    return extended, source_rows


def describe_input(rows: pd.DataFrame, *, raw: pd.DataFrame | None = None) -> dict:
    """
    JSON-safe support summary for FROZEN_SALES_COLUMNS rows (from build_inference_vehicles
    or from_history). raw, when given, must be the frame the rows were built from (same
    length and order; plates / Marka / Model are checked); only raw can tell an originally
    missing Ekspertiz text or seller name apart from a derived 0 / 'diger', so without it
    those counts are None (unknown). Nothing is excluded; WAWASAKY is a marker only.
    """
    _check_schema(rows, "rows")
    n = len(rows)
    if raw is not None:
        _check_alignment(rows, raw)

    missing = {}
    for c in _VEHICLE_FIELDS:
        s = rows[c]
        miss = _blank(s) if c in ("marka", "model", "model_uzantisi", "yakit_tipi",
                                  "vites_tipi", "kasa_tipi", "renk", "lokasyon") \
            else pd.Series(np.isnan(_num(s)))
        missing[c] = int(miss.sum())

    mf = _num(rows["min_fiyat_clean"])
    mf_present = np.isfinite(mf) & (mf > 0)
    hasar = _num(rows["hasar_compound"])
    zero_damage = hasar == 0
    if raw is None:
        eks_missing = seller_missing = plate_supplied = None
        zero_with_text = None
    else:
        eks_blank = _raw_blank(raw, "Ekspertiz")
        eks_missing = int(eks_blank.sum())
        zero_with_text = int((zero_damage & ~eks_blank).sum())
        seller_missing = int(_raw_blank(raw, "Satıcı Kısa Ünvan").sum())
        plate_supplied = int(_raw_plates(raw).ne("").sum())

    uz = rows["model_uzantisi"].astype("string")
    wawasaky = uz.str.upper().str.contains(WAWASAKY, na=False).to_numpy(dtype=bool)

    def flag_count(col: str) -> int:
        return int((_num(rows[col]) == 1).sum())

    def counts(col: str) -> dict:
        s = rows[col].astype("string").fillna("<NA>")
        return {str(k): int(v) for k, v in s.value_counts(sort=True).items()}

    return {
        "n_rows": n,
        "split": counts("split"),
        "dataset_version_ids": counts("dataset_version_id"),
        "missing": missing,
        "min_fiyat": {"n_present": int(mf_present.sum()), "n_absent": int((~mf_present).sum()),
                      "n_min_fiyat_was_zero_flag": flag_count("min_fiyat_was_zero")},
        "inspection": {"n_hasar_compound_zero": int(zero_damage.sum()),
                       "n_ekspertiz_missing": eks_missing,
                       "n_zero_score_with_text": zero_with_text,
                       "note": LIMITS[0]},
        "seller": {"satici_tipi": counts("satici_tipi"), "n_seller_missing": seller_missing,
                   "note": LIMITS[1]},
        "plate": {"n_format_valid": flag_count("plaka_format_valid"),
                  "n_plate_supplied": plate_supplied},
        "lineup": {"n_sira_no_missing": int(np.isnan(_num(rows["sira_no"])).sum())},
        "outcomes_redacted": bool(all(rows[c].isna().all() for c in REDACTED_OUTCOME_COLUMNS)),
        "anomaly_flags": {c: flag_count(c) for c in ("km_nonpositive_flag", "km_extreme_flag",
                                                     "eks_puani_out_of_range_flag")},
        "wawasaky": {"n": int(wawasaky.sum()),
                     "rows": [int(i) for i in np.flatnonzero(wawasaky)],
                     "note": LIMITS[5]},
        "limits": list(LIMITS) + ([] if raw is not None else [
            "Without raw, whether Ekspertiz text or the seller name was originally missing "
            "is unknown."]),
    }


# ----------------------------------------------------------------------------
# describe_input helpers.
# ----------------------------------------------------------------------------
def _raw_blank(raw: pd.DataFrame, col: str) -> np.ndarray:
    if col not in raw.columns:
        return np.ones(len(raw), dtype=bool)
    return _blank(raw[col].reset_index(drop=True)).to_numpy(dtype=bool)


def _raw_plates(raw: pd.DataFrame) -> pd.Series:
    if "Plaka" not in raw.columns:
        return pd.Series("", index=pd.RangeIndex(len(raw)), dtype="string")
    plate, _ = standardize_plaka(raw["Plaka"].reset_index(drop=True))
    return plate


def _check_alignment(rows: pd.DataFrame, raw: pd.DataFrame) -> None:
    """raw must be the frame rows were built from: same length, plates and Marka / Model."""
    _check_raw(raw)
    if len(raw) != len(rows):
        _fail("raw and rows differ in length; pass the raw frame the rows were built from")
    plate = _raw_plates(raw).to_numpy(dtype=object)
    given = plate != ""
    got = rows["plaka"].astype("string").fillna("").to_numpy(dtype=object)
    if not (got[given] == plate[given]).all():
        _fail("raw plates do not match rows.plaka; raw is not aligned with rows")
    synthetic = pd.Series(got[~given], dtype="string").str.startswith(SYNTHETIC_PLATE_PREFIX)
    if not synthetic.fillna(False).all():
        _fail("rows without a raw plate do not carry a synthetic plate; raw is not aligned")
    for src, col in (("Marka", "marka"), ("Model", "model")):
        if src in raw.columns:
            a = ascii_key_series(raw[src].reset_index(drop=True)).to_numpy(dtype=object)
            b = ascii_key_series(rows[col].reset_index(drop=True)).to_numpy(dtype=object)
            if not (a == b).all():
                _fail(f"raw {src} does not match rows.{col}; raw is not aligned with rows")

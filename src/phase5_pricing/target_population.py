"""
target_population.py  —  the Model B (price) target and training population.

Model B estimates, before its auction, the sale price of one car in one auction. This
module decides WHICH sold rows may carry a price label and WHAT that label is. It builds
no feature, trains no model, converts no currency, applies no log and performs NO file
I/O: every input is a caller-provided DataFrame (Phase-1 clean_sales).

TARGET (Serdar, 2026-10-02)
  target = satis_tutari (raw "Satış Tutarı"), nominal TL as recorded.
  tamamlama ("AA Esnası Tamamlama") is what the company pays from its own margin so that a
  sale closes at satis_tutari when the buyer pays less; the buyer pays about
  net_satis = satis_tutari - tamamlama. The price is satis_tutari, so net_satis is NOT the
  target. satis_tutari, tamamlama, net_satis, buyer fields, is_sold and
  off_auction_sale_flag describe the OUTCOME: they are carried for audit only and must
  never become model inputs.

ESTIMAND. The satis_tutari of one car in one auction, CONDITIONAL on a completed
in-auction sale to a non-internal buyer. It is not the probability that the car sells;
unsold cars (satis_tutari missing or <= 0, the confirmed Phase-1 rule is_sold =
satis_tutari > 0) never become zero-price examples and appear only in the audit.

POPULATION RULES (sold rows only; every sold source row ends in exactly one table)
  query_id = plaka + "::" + aa_kodu, exactly as phase2_labels/labels.py builds it.
  Reasons, in priority order (primary_reason = the first that applies; all_reasons lists
  every one that applies, "|"-joined in this order):
    invalid_key        plaka or aa_kodu missing / blank. Blank keys never form a query_id,
                       so they can never collide with each other.
    invalid_date       aa_tarihi missing or aa_tarihi_invalid_flag == 1
    repeated_query_id  the valid query_id occurs on more than one input row (sold or
                       unsold, equal or different prices). ALL its sold rows are excluded;
                       no first / last row is ever chosen.
    internal_buyer     internal_buyer_flag == 1
    off_auction_scope  off_auction_sale_flag == 1. The first Model B cohort is
                       auction-only; off-auction sales are counted, never scored.
  A sold row with no reason enters the population with target_value = satis_tutari.

KNOWN LIMITS (stated, not repaired)
  * Phase 1 filled a missing / unparseable tamamlama with 0, so clean_sales cannot tell an
    original missing value from a real 0. The audit reports them together as
    n_tamamlama_zero_or_missing; separating them needs raw-data lineage.
  * sold_price_anomaly_flag (sold with satis_tutari > 50M) is counted in the audit but is
    not an exclusion reason: no outlier rule has been decided.
  * Phase-1 model_b_eligible means "sold and not internal" (off-auction undecided there).
    It is checked for that meaning; the population is its auction-only subset.

PERIODS. assign_price_periods takes explicit, contiguous, inclusive date ranges whose
names follow PERIOD_ORDER (train < select < calibrate < retrospective_eval < s6_select <
d6_confirm); a caller declares the periods it uses, in that order, and their dates are
DYNAMIC (Serdar, 2026-10-03: later exports extend the data). A date outside the declared
ranges raises, so newly arriving data can never slip silently into a spent window.
PROPOSED_PRICE_PERIODS is the reviewed Phase-5 proposal; callers pass it on purpose. Every
2026 date up to 2026-06-04 has been inspected before (Phase 1, Model A's test, the legacy
table), so retrospective_eval is NOT an untouched test; s6_select / d6_confirm are the
Phase-6 selection and confirmation windows (Phase-6 contract §5).

PUBLIC API
  audit_price_fields(clean_sales)                      -> DataFrame (AUDIT_COLUMNS)
  build_price_population(clean_sales)                  -> (population, exclusions)
  assign_price_periods(population, *, boundaries)      -> population + `period`
  population_summary(population, exclusions, audit)    -> JSON-safe dict
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

TARGET_COLUMN = "satis_tutari"
TARGET_DECISION = (
    "Serdar 2026-10-02: the price is Satış Tutarı (satis_tutari). Tamamlama is paid by the "
    "company from its own margin when the buyer pays less; net_satis is not the target.")
ESTIMAND = (
    "satis_tutari (nominal TL) of one car in one auction, conditional on a completed "
    "in-auction sale to a non-internal buyer; not the probability of sale.")
TAMAMLAMA_LIMIT = (
    "Phase 1 filled missing tamamlama with 0; zero and missing cannot be told apart "
    "without raw-data lineage.")

KEY_SEPARATOR = "::"
FLAG_COLUMNS = ["is_sold", "internal_buyer_flag", "off_auction_sale_flag",
                "aa_tarihi_invalid_flag", "model_b_eligible", "sold_price_anomaly_flag"]
REQUIRED_COLUMNS = ["plaka", "aa_kodu", "aa_tarihi", "sira_no", "satis_tutari", "tamamlama",
                    "net_satis", *FLAG_COLUMNS, "dataset_version_id"]
REASONS = ("invalid_key", "invalid_date", "repeated_query_id", "internal_buyer",
           "off_auction_scope")

POPULATION_DTYPES = {
    "query_id": "string", "source_row": "int64", "plaka": "string", "aa_kodu": "string",
    "sira_no": "Int64", "aa_tarihi": "datetime64[ns]", "satis_tutari": "float64",
    "tamamlama": "float64", "net_satis": "float64", "target_value": "float64",
    "dataset_version_id": "string",
}
EXCLUSION_DTYPES = {
    "source_row": "int64", "query_id": "string", "plaka": "string", "aa_kodu": "string",
    "sira_no": "Int64", "aa_tarihi": "datetime64[ns]", "satis_tutari": "float64",
    "tamamlama": "float64", "net_satis": "float64", "primary_reason": "string",
    "all_reasons": "string", "dataset_version_id": "string",
}
AUDIT_KEYS = ["aa_month", "is_sold", "off_auction_sale_flag", "internal_buyer_flag"]
AUDIT_COUNTS = ["n_rows", "n_satis_na", "n_satis_zero", "n_satis_negative",
                "n_tamamlama_positive", "n_tamamlama_negative", "n_tamamlama_zero_or_missing",
                "n_net_nonpositive", "n_net_differs_from_satis", "n_net_gt_satis",
                "n_model_b_eligible", "n_sold_price_anomaly"]
AUDIT_QUANTILES = ["satis_q50", "satis_q90", "tamamlama_share_q50", "tamamlama_share_q90"]
AUDIT_COLUMNS = AUDIT_KEYS + AUDIT_COUNTS + AUDIT_QUANTILES

PERIOD_ORDER = ("train", "select", "calibrate", "retrospective_eval", "s6_select",
                "d6_confirm")
PERIOD_NAMES = PERIOD_ORDER[:4]                     # the Phase-5 protocol
PROPOSED_PRICE_PERIODS = {
    "train": ("2025-01-03", "2025-09-30"),
    "select": ("2025-10-01", "2025-12-31"),
    "calibrate": ("2026-01-01", "2026-02-28"),
    "retrospective_eval": ("2026-03-01", "2026-06-04"),
}


def _fail(msg: str) -> None:
    raise ValueError(f"[target_population] {msg}")


# ----------------------------------------------------------------------------
# Validation and shared row facts.
# ----------------------------------------------------------------------------
def _validate(clean_sales: pd.DataFrame) -> pd.DataFrame:
    """Checked copy of the required columns; source_row = 0-based input position."""
    if not isinstance(clean_sales, pd.DataFrame):
        raise TypeError("[target_population] clean_sales must be a DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in clean_sales.columns]
    if missing:
        _fail(f"clean_sales lacks required columns {missing}")
    df = clean_sales[REQUIRED_COLUMNS].copy()
    df.index = pd.RangeIndex(len(df))
    df.insert(0, "source_row", np.arange(len(df), dtype="int64"))
    for c in FLAG_COLUMNS:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or not v.isin([0, 1]).all():
            _fail(f"{c} must be binary 0/1 without missing values")
        df[c] = v.astype("int8")
    for c in ("satis_tutari", "tamamlama", "net_satis"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            _fail(f"{c} must be numeric")
        df[c] = df[c].astype("float64")
    if not pd.api.types.is_datetime64_any_dtype(df["aa_tarihi"]):
        _fail("aa_tarihi must be a datetime column")
    if getattr(df["aa_tarihi"].dt, "tz", None) is not None:
        _fail("aa_tarihi must be timezone-naive")
    df["aa_tarihi"] = df["aa_tarihi"].astype("datetime64[ns]")
    sold = np.isfinite(df["satis_tutari"]) & (df["satis_tutari"] > 0)
    if not (df["is_sold"].eq(1) == sold).all():
        _fail("is_sold disagrees with the confirmed rule is_sold = satis_tutari > 0")
    eligible = sold & df["internal_buyer_flag"].eq(0)
    if not (df["model_b_eligible"].eq(1) == eligible).all():
        _fail("model_b_eligible no longer means 'sold and not internal'; review Phase 1")
    versions = df["dataset_version_id"].astype("string").dropna().str.strip()
    versions = versions[versions != ""].unique()
    if len(df) and (len(versions) != 1 or df["dataset_version_id"].isna().any()):
        _fail(f"clean_sales must carry exactly one dataset_version_id, got {list(versions)}")
    for c in ("plaka", "aa_kodu", "dataset_version_id"):
        df[c] = df[c].astype("string")
    df["sira_no"] = pd.to_numeric(df["sira_no"], errors="coerce").astype("Int64")
    df["sold"] = sold.to_numpy()
    df["valid_key"] = ~(_blank(df["plaka"]) | _blank(df["aa_kodu"])).to_numpy(dtype=bool)
    df["query_id"] = (df["plaka"] + KEY_SEPARATOR + df["aa_kodu"]).where(df["valid_key"])
    df["query_id"] = df["query_id"].astype("string")
    df["valid_date"] = df["aa_tarihi"].notna() & df["aa_tarihi_invalid_flag"].eq(0)
    return df


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.fillna("").str.strip().eq("")


def _reason_matrix(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.loc[df["valid_key"], "query_id"].value_counts()
    repeated = df["query_id"].map(counts).fillna(0).gt(1) & df["valid_key"]
    return pd.DataFrame({
        "invalid_key": ~df["valid_key"],
        "invalid_date": ~df["valid_date"],
        "repeated_query_id": repeated,
        "internal_buyer": df["internal_buyer_flag"].eq(1),
        "off_auction_scope": df["off_auction_sale_flag"].eq(1),
    }, index=df.index)[list(REASONS)].astype(bool)


# ----------------------------------------------------------------------------
# Public API.
# ----------------------------------------------------------------------------
def audit_price_fields(clean_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Price-field counts for EVERY input row, grouped by aa_month ('YYYY-MM', or
    'invalid_date') x is_sold x off_auction_sale_flag x internal_buyer_flag. Quantiles use
    sold rows only; tamamlama_share = tamamlama / satis_tutari over sold rows with
    tamamlama > 0. n_tamamlama_zero_or_missing is ONE combined count (see TAMAMLAMA_LIMIT).
    """
    df = _validate(clean_sales)
    month = df["aa_tarihi"].dt.strftime("%Y-%m")
    df["aa_month"] = month.where(df["valid_date"], "invalid_date").astype(str)
    satis, tam, net = df["satis_tutari"], df["tamamlama"], df["net_satis"]
    sold = df["sold"]
    facts = pd.DataFrame({
        "n_rows": 1,
        "n_satis_na": satis.isna(),
        "n_satis_zero": satis.eq(0),
        "n_satis_negative": satis.lt(0),
        "n_tamamlama_positive": tam.gt(0),
        "n_tamamlama_negative": tam.lt(0),
        "n_tamamlama_zero_or_missing": tam.eq(0) | tam.isna(),
        "n_net_nonpositive": net.le(0),
        "n_net_differs_from_satis": ~(net.eq(satis) | (net.isna() & satis.isna())),
        "n_net_gt_satis": net.gt(satis),
        "n_model_b_eligible": df["model_b_eligible"].eq(1),
        "n_sold_price_anomaly": df["sold_price_anomaly_flag"].eq(1),
    }).astype("int64")
    keys = df[AUDIT_KEYS[1:]].astype("int8").assign(aa_month=df["aa_month"])[AUDIT_KEYS]
    out = pd.concat([keys, facts], axis=1).groupby(AUDIT_KEYS, sort=True).sum().reset_index()
    share = (tam / satis).where(sold & tam.gt(0))
    q = pd.DataFrame({"satis": satis.where(sold), "share": share}).assign(**{
        k: keys[k] for k in AUDIT_KEYS}).groupby(AUDIT_KEYS, sort=True)
    quant = pd.DataFrame({
        "satis_q50": q["satis"].quantile(0.5), "satis_q90": q["satis"].quantile(0.9),
        "tamamlama_share_q50": q["share"].quantile(0.5),
        "tamamlama_share_q90": q["share"].quantile(0.9)}).reset_index()
    out = out.merge(quant, on=AUDIT_KEYS, how="left", validate="one_to_one")
    for c in AUDIT_KEYS[1:]:
        out[c] = out[c].astype("int8")
    for c in AUDIT_QUANTILES:
        out[c] = out[c].astype("float64")
    return out[AUDIT_COLUMNS].reset_index(drop=True)


def build_price_population(clean_sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the SOLD rows of clean_sales into (population, exclusions). population: one row
    per query_id, target_value = satis_tutari, sorted by (aa_tarihi, query_id).
    exclusions: one row per rejected sold source row, sorted by source_row. Unsold rows
    appear in neither table. The input is not modified.
    """
    df = _validate(clean_sales)
    reasons = _reason_matrix(df)
    sold = df["sold"]
    rejected = sold & reasons.any(axis=1)
    kept = sold & ~rejected

    pop = df.loc[kept].copy()
    pop["target_value"] = pop[TARGET_COLUMN]
    pop = pop.sort_values(["aa_tarihi", "query_id"], kind="mergesort")
    pop = pop[list(POPULATION_DTYPES)].astype(POPULATION_DTYPES).reset_index(drop=True)
    if pop["query_id"].duplicated().any():
        _fail("internal error: a query_id occurs twice in the population")

    exc = df.loc[rejected].copy()
    r = reasons.loc[rejected]
    names = np.array(REASONS, dtype=object)
    exc["primary_reason"] = names[r.to_numpy().argmax(axis=1)] if len(r) else []
    exc["all_reasons"] = ["|".join(names[row]) for row in r.to_numpy()]
    exc = exc.sort_values("source_row", kind="mergesort")
    exc = exc[list(EXCLUSION_DTYPES)].astype(EXCLUSION_DTYPES).reset_index(drop=True)
    return pop, exc


def _parse_boundaries(boundaries) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(boundaries, Mapping):
        raise TypeError("[target_population] boundaries must be a mapping name -> (start, end)")
    names = tuple(boundaries)
    if not names or any(n not in PERIOD_ORDER for n in names) \
            or list(names) != sorted(names, key=PERIOD_ORDER.index) or len(set(names)) != len(names):
        _fail(f"boundaries must name periods from {PERIOD_ORDER}, each once, in this order")
    out, prev_end = {}, None
    for name, span in boundaries.items():
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            _fail(f"period {name} must be a (start, end) pair")
        try:
            start, end = (pd.Timestamp(x) for x in span)
        except (TypeError, ValueError):
            _fail(f"period {name} has an unparseable date")
        if pd.isna(start) or pd.isna(end) or start != start.normalize() or end != end.normalize():
            _fail(f"period {name} must use whole dates")
        if start.tz is not None or end.tz is not None:
            _fail(f"period {name} must use timezone-naive dates")
        if start > end:
            _fail(f"period {name} starts after it ends")
        if prev_end is not None and start != prev_end + pd.Timedelta(days=1):
            _fail(f"period {name} must start the day after the previous period ends "
                  f"(no gap, no overlap)")
        out[name] = (start, end)
        prev_end = end
    return out


def assign_price_periods(population: pd.DataFrame, *, boundaries) -> pd.DataFrame:
    """
    A copy of population with `period` (string) from explicit inclusive date ranges.
    Raises on a malformed boundary set or on any aa_tarihi outside the declared ranges.
    """
    if not isinstance(population, pd.DataFrame) or "aa_tarihi" not in population.columns:
        raise TypeError("[target_population] population must be a DataFrame with aa_tarihi")
    spans = _parse_boundaries(boundaries)
    day = population["aa_tarihi"].dt.normalize()
    if day.isna().any():
        _fail("population has a missing aa_tarihi")
    period = pd.Series(pd.NA, index=population.index, dtype="string")
    for name, (start, end) in spans.items():
        period = period.mask(day.between(start, end), name)
    if period.isna().any():
        outside = sorted(day[period.isna()].dt.date.astype(str).unique())
        _fail(f"{int(period.isna().sum())} rows fall outside the declared periods "
              f"(first dates {outside[:3]}); extend the boundaries explicitly")
    out = population.copy()
    out["period"] = period
    return out


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if obj is pd.NA or obj is None:
        return None
    return obj


def population_summary(population: pd.DataFrame, exclusions: pd.DataFrame,
                       audit: pd.DataFrame) -> dict:
    """
    JSON-safe summary. Raises unless every sold row in the audit is in exactly one of
    population / exclusions (counts and source_row identity).
    """
    for name, frame, cols in (("population", population, POPULATION_DTYPES),
                              ("exclusions", exclusions, EXCLUSION_DTYPES),
                              ("audit", audit, dict.fromkeys(AUDIT_COLUMNS))):
        if not isinstance(frame, pd.DataFrame) or [c for c in cols if c not in frame.columns]:
            _fail(f"{name} does not have the {name} schema")
    n_sold = int(audit.loc[audit["is_sold"].eq(1), "n_rows"].sum())
    rows = pd.concat([population["source_row"], exclusions["source_row"]])
    if rows.duplicated().any():
        _fail("a source_row appears in both population and exclusions")
    if len(rows) != n_sold:
        _fail(f"sold-row reconciliation failed: audit {n_sold} != population "
              f"{len(population)} + exclusions {len(exclusions)}")
    reasons_all = exclusions["all_reasons"].astype(str).str.split("|")
    tam = population["tamamlama"]
    versions = sorted(set(population["dataset_version_id"].dropna().astype(str))
                      | set(exclusions["dataset_version_id"].dropna().astype(str)))
    summary = {
        "target_column": TARGET_COLUMN,
        "target_decision": TARGET_DECISION,
        "estimand": ESTIMAND,
        "dataset_version_id": versions,
        "n_input_rows": int(audit["n_rows"].sum()),
        "n_sold_rows": n_sold,
        "n_population": len(population),
        "n_excluded_sold_rows": len(exclusions),
        "excluded_by_primary_reason": {r: int(exclusions["primary_reason"].eq(r).sum())
                                       for r in REASONS},
        "rows_with_reason": {r: int(reasons_all.apply(lambda xs, r=r: r in xs).sum())
                             for r in REASONS},
        "n_model_b_eligible": int(audit["n_model_b_eligible"].sum()),
        "n_sold_price_anomaly": int(audit["n_sold_price_anomaly"].sum()),
        "population_tamamlama": {
            "n_positive": int(tam.gt(0).sum()),
            "n_negative": int(tam.lt(0).sum()),
            "n_zero_or_missing": int((tam.eq(0) | tam.isna()).sum()),
            "share_q50": (tam / population["satis_tutari"]).where(tam.gt(0)).quantile(0.5),
        },
        "population_date_range": ([str(population["aa_tarihi"].min().date()),
                                   str(population["aa_tarihi"].max().date())]
                                  if len(population) else None),
        "tamamlama_limitation": TAMAMLAMA_LIMIT,
        "scope": "auction-only; off-auction sales counted, never scored",
    }
    if "period" in population.columns:
        present = [PERIOD_ORDER.index(p) for p in population["period"].dropna().unique()
                   if p in PERIOD_ORDER]
        last = max(present + [len(PERIOD_NAMES) - 1])  # Phase-5 periods always listed
        summary["population_by_period"] = {
            p: int(population["period"].eq(p).sum()) for p in PERIOD_ORDER[:last + 1]}
    return _clean(summary)

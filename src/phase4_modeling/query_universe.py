"""
query_universe.py  —  the Phase-4 query ledger and bounded candidate KEY streams.

For every car-in-auction (a "query") this module answers three questions, in memory:
  * which observed positive firms could the pre-auction pool reach?   reconcile_positives
  * what does each query look like before any model sees it?          build_query_ledger
  * which (query, firm) KEYS go to training and to evaluation?        iter_training_keys /
                                                                      iter_evaluation_keys
It computes no feature, trains no model, scores no metric and performs NO file I/O.

SOURCES (all caller-provided DataFrames; nothing is loaded here)
  queries       query_id, plaka, aa_kodu, aa_tarihi, split — the Phase-2 query table
                (phase2_labels/labels.py build_query_table, QUERY_COLUMNS) with the original
                source split (clean_sales.py SPLIT_BOUNDARY, labels.py add_split) — plus
                is_duplicate_listing / is_orphan_query from Phase-2 query_audit
                (labels.py QUERY_AUDIT_COLUMNS). No vehicle rule is inferred here.
  positives     the Phase-2 grade >= 1 (query_id, firma_key, grade) pairs
                (labels.py build_bidder_pairs / assign_grade). Their date comes from queries.
  participants  cleaned AA participation keys (phase2_labels/participation.py
                clean_participation: aa_kodu, firma_key). Used ONLY for label provenance and
                diagnostic counts — never to choose a key.
  index         the accepted Phase-3 EventIndex (phase3_firm_universe/candidate_pool.py).
Pool membership and reachability come ONLY from candidate_pool_as_of and classify_pairs.
This module has no membership or timing logic of its own.

LABELS
  grade 2  phase2_win   observed bid + supported auction win (Phase-2 rule, unchanged)
  grade 1  phase2_bid   observed bid without a qualifying win (off-auction stays 1)
  grade 0  phase2_participant_no_bid   pool firm that attended the auction, no recorded bid
           pool_only_no_bid            pool firm with no recorded bid, not a participant
  Grade 0 means NO RECORDED BID, and only under the ASSUMPTION that clean_reel_db completely
  records the platform bids for that query. It is never proven disinterest, and pool firms
  missing from Phase 2 had no pre-approved label: this is a derived zero. label_provenance
  is AUDIT METADATA and must never be used as a model feature. Labels are attached only
  AFTER keys are selected, so participation can never change which keys are selected.
  GAIN_BY_GRADE (0/1/3) is the gain for the future ranker objective and for NDCG.

  An observed positive that is NOT in the pool is a coverage miss. It stays in the
  reconciled table and in the ledger counts, and it is NEVER inserted into any key stream.

PERIODS. fit: aa_tarihi < validation_start (default 2025-10-01); validation: up to
SPLIT_BOUNDARY (2026-01-01); test: from SPLIT_BOUNDARY on. The original `split` column is
carried unchanged next to `period`. Test is held out: training never sees it and
evaluation emits it only with allow_test=True. Training on ('fit', 'validation') is the
pre-2026 refit, with settings frozen by the caller.

CALLER CONTRACT. Every call for one ledger must receive the SAME immutable EventIndex and
the SAME exclude values. Each public call materializes `exclude` once as a frozenset, so a
one-shot generator is safe. The module only DETECTS a date whose pool size or reachable
positive set disagrees with the ledger / reconciled table. Two different pools can agree
on both, so this is not a full identity check.

CHUNKS ARE TRANSPORT UNITS, NOT RANKING GROUPS. Both iterators yield DataFrames of at most
max_rows rows, ordered by (aa_tarihi, query_id, firma_key). A query larger than a chunk
continues in the next one: nothing is dropped and nothing is re-sampled per chunk, and the
concatenated stream is identical for every max_rows. Consumers that rank, group or score
MUST reassemble each complete query_id group across chunk boundaries (for example, carry
group state until query_id changes or the iterator ends). An iterator with nothing to
emit yields no chunk; every yielded chunk has 1..max_rows rows. Memory is bounded by the
inputs, one date's pool and one chunk: pools are never cached and no date-wide car-by-firm
table is built.

METRIC CONTRACT (documented here; implemented in a later file, not in this module)
  P = observed positive firms of a query, R = P in the pool, W = the grade-2 firm,
  T = first min(K, pool_size) pool firms by score, ties by firma_key ascending.
  Q_elig = non-quarantined queries; Q_pos = Q_elig with |P| > 0; Q_reach = Q_elig, |R| > 0.
  Conditional Recall@K = |T intersection R| / |R| and NDCG@K (gains 0/1/3, ideal ranking
  from the full pool) are arithmetic means over COUNT(Q_reach). Micro conditional recall
  sum(|T intersection R|) / sum(|R|) is reported separately. Over Q_pos: query coverage =
  #Q_reach / #Q_pos; pair coverage = sum(|R|) / sum(|P|); winner coverage likewise on W;
  end-to-end Recall@K = sum(|T intersection P|) / sum(|P|), where zero-reachable and
  empty-pool queries count as misses. Zero-positive queries are excluded from those
  denominators but kept in counts and evaluation. Conservative all-query variants add
  quarantined positives to denominators with a zero numerator. An empty denominator gives
  NaN with count 0.
"""
from __future__ import annotations

import hashlib
import json
import numbers
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from common.normalize import ascii_key_series
from phase1_preparation.clean_sales import SPLIT_BOUNDARY
# _check_exclude is the accepted exclusion validator; reused so both modules agree.
from phase3_firm_universe.candidate_pool import (LANE_DTYPE, EventIndex, _check_exclude,
                                                 candidate_pool_as_of, classify_pairs)

if SPLIT_BOUNDARY != pd.Timestamp("2026-01-01"):
    raise ValueError(f"[query_universe] SPLIT_BOUNDARY must be 2026-01-01, got {SPLIT_BOUNDARY}")

VALIDATION_START = pd.Timestamp("2025-10-01")

PERIODS = ("fit", "validation", "test")
STATUSES = ("quarantined", "zero_positive", "zero_reachable", "reachable_positive")
PROVENANCES = ("phase2_win", "phase2_bid", "phase2_participant_no_bid", "pool_only_no_bid")
TRAINING_PERIODS = (("fit",), ("fit", "validation"))
GAIN_BY_GRADE = {0: 0, 1: 1, 2: 3}

PERIOD_DTYPE = pd.CategoricalDtype(list(PERIODS))
STATUS_DTYPE = pd.CategoricalDtype(list(STATUSES))
PROVENANCE_DTYPE = pd.CategoricalDtype(list(PROVENANCES))

QUERY_COLUMNS = ["query_id", "plaka", "aa_kodu", "aa_tarihi", "split",
                 "is_duplicate_listing", "is_orphan_query"]
POSITIVE_COLUMNS = ["query_id", "firma_key", "grade"]
PARTICIPANT_COLUMNS = ["aa_kodu", "firma_key"]

RECONCILED_DTYPES = {
    "query_id": "string", "firma_key": "string", "aa_tarihi": "datetime64[ns]",
    "grade": "int8", "reachable": "bool", "lane": LANE_DTYPE,     # lane missing = unreachable
}
COUNT_COLUMNS = ["pool_size", "n_pos", "n_reach_pos", "n_win", "n_reach_win",
                 "n_phase2_zero_in_pool", "n_phase2_zero_out_pool"]
LEDGER_DTYPES = {
    "query_id": "string", "plaka": "string", "aa_kodu": "string",
    "aa_tarihi": "datetime64[ns]", "split": "string", "period": PERIOD_DTYPE,
    "is_duplicate_listing": "bool", "is_orphan_query": "bool", "quarantine_reason": "string",
    **{c: "int32" for c in COUNT_COLUMNS},
    "status": STATUS_DTYPE,
}
KEY_DTYPES = {
    "query_id": "string", "aa_tarihi": "datetime64[ns]", "firma_key": "string",
    "grade": "int8", "label_provenance": PROVENANCE_DTYPE, "split": "string",
    "period": PERIOD_DTYPE,
}
COVERAGE_DTYPES = {
    "period": PERIOD_DTYPE, "status": STATUS_DTYPE, "n_queries": "int64", "n_pos": "int64",
    "n_reach_pos": "int64", "n_win": "int64", "n_reach_win": "int64",
    "n_pool_pairs": "int64",                                      # sum of pool_size
    "n_phase2_zero_in_pool": "int64", "n_phase2_zero_out_pool": "int64",
}

_SPLITS = ("train", "test")


# ----------------------------------------------------------------------------
# Input checks — fail loudly; identifiers are checked, never normalized.
# ----------------------------------------------------------------------------
def _fail(message: str) -> None:
    raise ValueError(f"[query_universe] {message}")


def _require_frame(df: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[query_universe] {name} must be a pandas DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(f"{name} is missing required columns: {missing}")


def _check_index(index: EventIndex) -> None:
    if not isinstance(index, EventIndex):
        raise TypeError("[query_universe] index must be a candidate_pool.EventIndex")


def _strings(s: pd.Series, name: str, *, allow_blank: bool = False) -> np.ndarray:
    """Non-null str values -> object array. Blank means empty after stripping."""
    if s.isna().any():
        _fail(f"{name} has {int(s.isna().sum())} null values")
    vals = s.to_numpy(dtype=object)
    if not all(isinstance(v, str) for v in vals):
        _fail(f"{name} holds non-string values")
    if not allow_blank:
        blank = np.fromiter((v.strip() == "" for v in vals), dtype=bool, count=len(vals))
        if blank.any():
            _fail(f"{name} has {int(blank.sum())} blank values")
    return vals


def _dates(s: pd.Series, name: str) -> np.ndarray:
    """Naive, non-null, midnight datetime64 column -> datetime64[ns] array."""
    if isinstance(s.dtype, pd.DatetimeTZDtype) or not pd.api.types.is_datetime64_dtype(s.dtype):
        _fail(f"{name} must be a timezone-naive datetime64 column, got {s.dtype}")
    if s.isna().any():
        _fail(f"{name} has {int(s.isna().sum())} null dates")
    s = s.astype("datetime64[ns]")
    if not s.eq(s.dt.normalize()).all():
        _fail(f"{name} holds non-midnight values; it is an auction-day field")
    return s.to_numpy(dtype="datetime64[ns]")


def _bools(s: pd.Series, name: str) -> np.ndarray:
    if not pd.api.types.is_bool_dtype(s.dtype) or s.isna().any():
        _fail(f"{name} must be a non-null boolean column, got {s.dtype}")
    return s.to_numpy(dtype=bool)


def _grades(s: pd.Series, name: str) -> np.ndarray:
    """Observed positive grades: integers in {1, 2}. Never coerced."""
    if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_integer_dtype(s.dtype):
        _fail(f"{name} must be an integer column, got {s.dtype}")
    if s.isna().any():
        _fail(f"{name} has null grades")
    bad = ~s.isin([1, 2])
    if bad.any():
        _fail(f"{name} holds grades outside {{1, 2}}: {sorted(set(s[bad].tolist()))[:5]}")
    return s.to_numpy(dtype=np.int8)


def _counts(s: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_integer_dtype(s.dtype):
        _fail(f"{name} must be an integer column, got {s.dtype}")
    if s.isna().any() or (s < 0).any():
        _fail(f"{name} must be non-null and non-negative")
    return s.to_numpy(dtype=np.int64)


def _labels(s: pd.Series, allowed: tuple[str, ...], name: str) -> np.ndarray:
    vals = s.astype(object).to_numpy()
    if pd.isna(vals).any() or not np.isin(vals, np.array(allowed, dtype=object)).all():
        _fail(f"{name} must only hold {list(allowed)}")
    return vals


def _int_arg(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"[query_universe] {name} must be an int, got {value!r}")
    if value < minimum:
        _fail(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_validation_start(value) -> pd.Timestamp:
    if value is None or isinstance(value, (bool, np.bool_, numbers.Number)):
        _fail(f"validation_start must be a date, got {value!r}")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"[query_universe] validation_start is not a valid date: {value!r}") from exc
    if ts is pd.NaT:
        _fail("validation_start is NaT")
    if ts.tz is not None:
        _fail("validation_start must be timezone-naive")
    if ts != ts.normalize():
        _fail(f"validation_start must be a midnight, got {ts}")
    if ts >= SPLIT_BOUNDARY:
        _fail(f"validation_start must be before {SPLIT_BOUNDARY.date()}, got {ts}")
    return ts


def _validate_queries(queries: pd.DataFrame) -> pd.DataFrame:
    _require_frame(queries, QUERY_COLUMNS, "queries")
    qid = _strings(queries["query_id"], "queries.query_id")
    plaka = _strings(queries["plaka"], "queries.plaka")
    code = _strings(queries["aa_kodu"], "queries.aa_kodu")
    split = _strings(queries["split"], "queries.split")
    dates = _dates(queries["aa_tarihi"], "queries.aa_tarihi")
    q = pd.DataFrame({
        "query_id": pd.array(qid, dtype="string"), "plaka": pd.array(plaka, dtype="string"),
        "aa_kodu": pd.array(code, dtype="string"), "aa_tarihi": dates,
        "split": pd.array(split, dtype="string"),
        "is_duplicate_listing": _bools(queries["is_duplicate_listing"], "queries.is_duplicate_listing"),
        "is_orphan_query": _bools(queries["is_orphan_query"], "queries.is_orphan_query"),
    })
    dup = q["query_id"].duplicated()
    if dup.any():
        _fail(f"queries.query_id is not unique, e.g. {q.loc[dup, 'query_id'].head(3).tolist()}")
    bad = q["query_id"].ne(q["plaka"] + "::" + q["aa_kodu"])
    if bad.any():
        _fail(f"query_id differs from plaka + '::' + aa_kodu, e.g. {q.loc[bad, 'query_id'].head(3).tolist()}")
    if not np.isin(split, np.array(_SPLITS, dtype=object)).all():
        _fail(f"queries.split must only hold {list(_SPLITS)}")
    expected = np.where(dates < SPLIT_BOUNDARY.to_datetime64(), "train", "test")
    bad = split != expected
    if bad.any():
        _fail(f"queries.split disagrees with aa_tarihi vs {SPLIT_BOUNDARY.date()}, "
              f"e.g. {qid[bad][:3].tolist()}")
    return q


def _validate_positives(positives: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    _require_frame(positives, POSITIVE_COLUMNS, "positives")
    qid = _strings(positives["query_id"], "positives.query_id")
    key = _strings(positives["firma_key"], "positives.firma_key")
    grade = _grades(positives["grade"], "positives.grade")
    pos = pd.DataFrame({"query_id": pd.array(qid, dtype="string"),
                        "firma_key": pd.array(key, dtype="string"), "grade": grade})
    dup = pos.duplicated(["query_id", "firma_key"])
    if dup.any():
        _fail(f"positives repeat a (query_id, firma_key) pair, "
              f"e.g. {pos.loc[dup, ['query_id', 'firma_key']].head(3).values.tolist()}")
    at = pd.Index(q["query_id"]).get_indexer(pos["query_id"])
    if (at < 0).any():
        _fail(f"positives name queries absent from queries, e.g. {qid[at < 0][:3].tolist()}")
    _check_one_winner(pos, "positives")
    pos["aa_tarihi"] = q["aa_tarihi"].to_numpy()[at]      # the date comes from queries
    return pos


def _check_one_winner(frame: pd.DataFrame, name: str) -> None:
    winners = frame.loc[frame["grade"] == 2].groupby("query_id")["firma_key"].size()
    many = winners[winners > 1].index
    if len(many):
        _fail(f"{name} has more than one grade-2 firm for a query, e.g. {list(many[:3])}")


def _validate_participants(participants: pd.DataFrame) -> pd.DataFrame:
    _require_frame(participants, PARTICIPANT_COLUMNS, "participants")
    code = _strings(participants["aa_kodu"], "participants.aa_kodu")
    key = _strings(participants["firma_key"], "participants.firma_key")
    parts = pd.DataFrame({"aa_kodu": pd.array(code, dtype="string"),
                          "firma_key": pd.array(key, dtype="string")})
    bad = parts["firma_key"].ne(ascii_key_series(parts["firma_key"]))
    if bad.any():
        _fail(f"participants.firma_key holds keys not in ascii_key form, "
              f"e.g. {parts.loc[bad, 'firma_key'].head(3).tolist()}")
    dup = parts.duplicated()
    if dup.any():
        _fail(f"participants repeat an (aa_kodu, firma_key) row, "
              f"e.g. {parts.loc[dup].head(3).values.tolist()}")
    return parts


def _check_reconciled(reconciled: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Validate a reconcile_positives output against a query_id/aa_tarihi reference."""
    _require_frame(reconciled, list(RECONCILED_DTYPES), "reconciled")
    qid = _strings(reconciled["query_id"], "reconciled.query_id")
    key = _strings(reconciled["firma_key"], "reconciled.firma_key")
    dates = _dates(reconciled["aa_tarihi"], "reconciled.aa_tarihi")
    grade = _grades(reconciled["grade"], "reconciled.grade")
    reach = _bools(reconciled["reachable"], "reconciled.reachable")
    lane = reconciled["lane"].astype(object).to_numpy()
    has_lane = ~pd.isna(lane)
    if not np.isin(lane[has_lane], np.array(["A", "B", "C"], dtype=object)).all() \
            or (has_lane != reach).any():
        _fail("reconciled.lane must be A/B/C exactly where reachable is True")
    rec = pd.DataFrame({"query_id": pd.array(qid, dtype="string"),
                        "firma_key": pd.array(key, dtype="string"),
                        "aa_tarihi": dates, "grade": grade, "reachable": reach})
    if rec.duplicated(["query_id", "firma_key"]).any():
        _fail("reconciled repeats a (query_id, firma_key) pair")
    at = pd.Index(ref["query_id"]).get_indexer(rec["query_id"])
    if (at < 0).any():
        _fail(f"reconciled names unknown queries, e.g. {qid[at < 0][:3].tolist()}")
    if (ref["aa_tarihi"].to_numpy()[at] != dates).any():
        _fail("reconciled.aa_tarihi disagrees with the query date")
    _check_one_winner(rec, "reconciled")
    return rec


def _derive_reason_status(dup: np.ndarray, orph: np.ndarray,
                          n_pos: np.ndarray, n_reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reason = np.select([dup & orph, dup, orph],
                       ["duplicate_listing|orphan_query", "duplicate_listing", "orphan_query"],
                       default="").astype(object)
    status = np.select([dup | orph, n_pos == 0, n_reach == 0],
                       list(STATUSES[:3]), default=STATUSES[3]).astype(object)
    return reason, status


def _validate_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """A build_query_ledger output, re-checked so the iterators can trust it."""
    _require_frame(ledger, list(LEDGER_DTYPES), "ledger")
    cols = {c: _strings(ledger[c], f"ledger.{c}") for c in ("query_id", "plaka", "aa_kodu", "split")}
    reason = _strings(ledger["quarantine_reason"], "ledger.quarantine_reason", allow_blank=True)
    dup = _bools(ledger["is_duplicate_listing"], "ledger.is_duplicate_listing")
    orph = _bools(ledger["is_orphan_query"], "ledger.is_orphan_query")
    counts = {c: _counts(ledger[c], f"ledger.{c}") for c in COUNT_COLUMNS}
    period = _labels(ledger["period"], PERIODS, "ledger.period")
    status = _labels(ledger["status"], STATUSES, "ledger.status")
    split = cols["split"]
    if not np.isin(split, np.array(_SPLITS, dtype=object)).all():
        _fail(f"ledger.split must only hold {list(_SPLITS)}")
    if pd.Series(cols["query_id"]).duplicated().any():
        _fail("ledger.query_id is not unique")
    exp_reason, exp_status = _derive_reason_status(dup, orph, counts["n_pos"], counts["n_reach_pos"])
    if (exp_reason != reason).any() or (exp_status != status).any():
        _fail("ledger quarantine_reason/status disagree with its flags and counts")
    if ((period == "test") != (split == "test")).any():
        _fail("ledger.period 'test' must coincide with split 'test'")
    out = pd.DataFrame({
        **{c: pd.array(v, dtype="string") for c, v in cols.items()},
        "aa_tarihi": _dates(ledger["aa_tarihi"], "ledger.aa_tarihi"),
        "period": pd.Categorical(period, dtype=PERIOD_DTYPE),
        "is_duplicate_listing": dup, "is_orphan_query": orph,
        "quarantine_reason": pd.array(reason, dtype="string"),
        **counts,
        "status": pd.Categorical(status, dtype=STATUS_DTYPE),
    })
    return out[list(LEDGER_DTYPES)]


def _positive_counts(rec: pd.DataFrame, qids: pd.Series) -> dict[str, np.ndarray]:
    """n_pos / n_reach_pos / n_win / n_reach_win per query, in qids order."""
    n = len(qids)
    at = pd.Index(qids).get_indexer(rec["query_id"])
    reach = rec["reachable"].to_numpy(dtype=bool)
    win = rec["grade"].to_numpy() == 2

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(at[mask], minlength=n).astype(np.int64)
    all_rows = np.ones(len(rec), dtype=bool)
    return {"n_pos": count(all_rows), "n_reach_pos": count(reach),
            "n_win": count(win), "n_reach_win": count(win & reach)}


def _pool_keys(index: EventIndex, date, excluded: frozenset[str]) -> np.ndarray:
    """The accepted pool for one auction date, as a firma_key-sorted object array."""
    pool = candidate_pool_as_of(index, pd.Timestamp(date), exclude=excluded)
    return np.sort(pool["firma_key"].to_numpy(dtype=object))


def _check_reachable(keys: np.ndarray, reach: np.ndarray, pool_keys: np.ndarray, where: str) -> None:
    if (np.isin(keys, pool_keys) != reach).any():
        _fail(f"reconciled.reachable disagrees with the pool {where}; pass the same index "
              f"and exclude to every call")


# ----------------------------------------------------------------------------
# Public interface.
# ----------------------------------------------------------------------------
def reconcile_positives(index: EventIndex,
                        queries: pd.DataFrame,
                        positives: pd.DataFrame,
                        *,
                        exclude: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Classify every observed positive (grade 1/2) against the pool of its auction date.
    Returns RECONCILED_DTYPES columns, one row per positive, sorted by (query_id,
    firma_key). Unreachable positives stay here for coverage accounting; they are never
    inserted into a pool or a key stream. Reachability comes from classify_pairs, which
    also rejects keys that are not normalized or not in the registry.
    """
    _check_index(index)
    q = _validate_queries(queries)
    pos = _validate_positives(positives, q)
    excluded = _check_exclude(exclude)
    classified = classify_pairs(index, pos[["query_id", "firma_key", "aa_tarihi", "grade"]],
                                exclude=excluded)
    out = pd.DataFrame({
        "query_id": pd.array(classified["query_id"].to_numpy(dtype=object), dtype="string"),
        "firma_key": pd.array(classified["firma_key"].to_numpy(dtype=object), dtype="string"),
        "aa_tarihi": classified["aa_tarihi"].to_numpy(dtype="datetime64[ns]"),
        "grade": classified["grade"].to_numpy(dtype=np.int8),
        "reachable": classified["reachable"].to_numpy(dtype=bool),
        "lane": pd.Categorical(classified["lane"].astype(object), dtype=LANE_DTYPE),
    })
    return out.sort_values(["query_id", "firma_key"], kind="mergesort").reset_index(drop=True)


def build_query_ledger(index: EventIndex,
                       queries: pd.DataFrame,
                       reconciled: pd.DataFrame,
                       participants: pd.DataFrame,
                       *,
                       exclude: Iterable[str] | None = None,
                       validation_start: pd.Timestamp = VALIDATION_START) -> pd.DataFrame:
    """
    One row per query (LEDGER_DTYPES, sorted by query_id). Quarantine comes only from the
    caller's audit flags; positive counts are kept for quarantined queries so their losses
    can be reported. n_phase2_zero_in_pool / _out_pool count the query's same-auction
    participants who are not its positive bidders, split by pool membership; the out-of-
    pool count is a diagnostic only. status, first match wins: quarantined, zero_positive,
    zero_reachable (including empty pools), reachable_positive. Each date's pool is built
    once and released before the next.
    """
    _check_index(index)
    q = _validate_queries(queries)
    rec = _check_reconciled(reconciled, q)
    parts = _validate_participants(participants)
    excluded = _check_exclude(exclude)
    start = _check_validation_start(validation_start)

    n = len(q)
    dates = q["aa_tarihi"].to_numpy()
    codes = q["aa_kodu"].to_numpy(dtype=object)
    counts = _positive_counts(rec, q["query_id"])

    # Positives that also attended their auction are not Phase-2 zeros.
    rec_at = pd.Index(q["query_id"]).get_indexer(rec["query_id"])
    rec_keys = rec["firma_key"].to_numpy(dtype=object)
    rec_reach = rec["reachable"].to_numpy(dtype=bool)
    rec_dates = rec["aa_tarihi"].to_numpy()
    part_idx = pd.MultiIndex.from_arrays([parts["aa_kodu"].to_numpy(dtype=object),
                                          parts["firma_key"].to_numpy(dtype=object)])
    rec_is_part = pd.MultiIndex.from_arrays([codes[rec_at], rec_keys]).isin(part_idx)

    part_codes = parts["aa_kodu"].to_numpy(dtype=object)
    part_keys = parts["firma_key"].to_numpy(dtype=object)
    pool_size = np.zeros(n, np.int64)
    zero_in = np.zeros(n, np.int64)
    zero_out = np.zeros(n, np.int64)
    for date in np.unique(dates):
        rows = np.flatnonzero(dates == date)
        pool_keys = _pool_keys(index, date, excluded)
        pool_size[rows] = len(pool_keys)
        on_day = rec_dates == date
        _check_reachable(rec_keys[on_day], rec_reach[on_day], pool_keys,
                         f"on {pd.Timestamp(date).date()}")
        day_codes = np.unique(codes[rows])
        m = np.isin(part_codes, day_codes)
        in_pool = pd.Series(np.isin(part_keys[m], pool_keys), dtype=bool).groupby(part_codes[m])
        n_in = in_pool.sum().reindex(codes[rows]).fillna(0).to_numpy(dtype=np.int64)
        n_all = in_pool.size().reindex(codes[rows]).fillna(0).to_numpy(dtype=np.int64)
        zero_in[rows] = n_in
        zero_out[rows] = n_all - n_in
    zero_in -= np.bincount(rec_at[rec_is_part & rec_reach], minlength=n)
    zero_out -= np.bincount(rec_at[rec_is_part & ~rec_reach], minlength=n)

    dup = q["is_duplicate_listing"].to_numpy(dtype=bool)
    orph = q["is_orphan_query"].to_numpy(dtype=bool)
    reason, status = _derive_reason_status(dup, orph, counts["n_pos"], counts["n_reach_pos"])
    period = np.where(dates < start.to_datetime64(), "fit",
                      np.where(dates < SPLIT_BOUNDARY.to_datetime64(), "validation", "test"))

    ledger = pd.DataFrame({
        "query_id": q["query_id"], "plaka": q["plaka"], "aa_kodu": q["aa_kodu"],
        "aa_tarihi": dates, "split": q["split"],
        "period": pd.Categorical(period, dtype=PERIOD_DTYPE),
        "is_duplicate_listing": dup, "is_orphan_query": orph,
        "quarantine_reason": pd.array(reason, dtype="string"),
        "pool_size": pool_size, **counts,
        "n_phase2_zero_in_pool": zero_in, "n_phase2_zero_out_pool": zero_out,
        "status": pd.Categorical(status, dtype=STATUS_DTYPE),
    })
    ledger = ledger.astype({c: "int32" for c in COUNT_COLUMNS})[list(LEDGER_DTYPES)]
    return ledger.sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _prepare_stream(index: EventIndex, ledger: pd.DataFrame, reconciled: pd.DataFrame,
                    participants: pd.DataFrame, exclude) -> tuple:
    """Shared eager validation: ledger, reconciled against it, participants, exclude."""
    _check_index(index)
    led = _validate_ledger(ledger)
    rec = _check_reconciled(reconciled, led)
    counts = _positive_counts(rec, led["query_id"])
    if any((counts[c] != led[c].to_numpy()).any() for c in counts):
        _fail("reconciled positive counts disagree with the ledger; build both from the same inputs")
    return led, rec, _validate_participants(participants), _check_exclude(exclude)


def iter_training_keys(index: EventIndex,
                       ledger: pd.DataFrame,
                       reconciled: pd.DataFrame,
                       participants: pd.DataFrame,
                       *,
                       negatives_per_query: int,
                       seed: str,
                       max_rows: int,
                       periods: tuple[str, ...] = ("fit",),
                       exclude: Iterable[str] | None = None) -> Iterator[pd.DataFrame]:
    """
    Training KEYS for reachable_positive queries in `periods`: ('fit',), or ('fit',
    'validation') for the pre-2026 refit. Test is never allowed. Per query: every reachable
    positive plus min(negatives_per_query, pool_size - n_reach_pos) negatives drawn without
    replacement from the rest of the pool. There is no default budget.

    A negative's rank is the first 8 bytes (big-endian unsigned) of SHA-256 over UTF-8
    json.dumps([seed, query_id, firma_key], ensure_ascii=False, separators=(',', ':')),
    ties broken by firma_key; the lowest ranks are kept. No RNG and no Python hash(), so
    selection does not depend on input order or chunk size. Rows are emitted in
    (aa_tarihi, query_id, firma_key) order, not hash order. Arguments are validated before
    the iterator is returned.
    """
    budget = _int_arg(negatives_per_query, "negatives_per_query", 0)
    rows = _int_arg(max_rows, "max_rows", 1)
    if not isinstance(seed, str) or seed == "":
        _fail("seed must be a non-empty string")
    if not isinstance(periods, tuple) or periods not in TRAINING_PERIODS:
        _fail(f"training periods must be one of {list(TRAINING_PERIODS)}, got {periods!r}")
    led, rec, parts, excluded = _prepare_stream(index, ledger, reconciled, participants, exclude)
    chosen = led.loc[led["status"].eq("reachable_positive") & led["period"].isin(periods)]
    return _generate_keys(index, chosen, rec, parts, excluded, rows, budget=budget, seed=seed)


def iter_evaluation_keys(index: EventIndex,
                         ledger: pd.DataFrame,
                         reconciled: pd.DataFrame,
                         participants: pd.DataFrame,
                         *,
                         periods: tuple[str, ...],
                         max_rows: int,
                         allow_test: bool = False,
                         exclude: Iterable[str] | None = None) -> Iterator[pd.DataFrame]:
    """
    Full-pool evaluation KEYS, no sampling, for every non-quarantined query in `periods`,
    including zero_positive and zero_reachable queries. An empty pool emits no rows, and
    its ledger row remains. Periods containing 'test' require allow_test=True. Arguments
    are validated before the iterator is returned.
    """
    rows = _int_arg(max_rows, "max_rows", 1)
    if not isinstance(allow_test, (bool, np.bool_)):
        raise TypeError("[query_universe] allow_test must be a bool")
    if (not isinstance(periods, tuple) or not periods or len(set(periods)) != len(periods)
            or any(p not in PERIODS for p in periods)):
        _fail(f"evaluation periods must be a non-empty tuple of distinct {list(PERIODS)}, got {periods!r}")
    if "test" in periods and not allow_test:
        _fail("the test period is held out; pass allow_test=True for the final evaluation")
    led, rec, parts, excluded = _prepare_stream(index, ledger, reconciled, participants, exclude)
    chosen = led.loc[led["status"].ne("quarantined") & led["period"].isin(periods)]
    return _generate_keys(index, chosen, rec, parts, excluded, rows, budget=None, seed=None)


def coverage_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Counts per observed (period, status) group, COVERAGE_DTYPES. No metric is scored."""
    led = _validate_ledger(ledger)
    grouped = led.groupby(["period", "status"], observed=True, sort=True)
    out = grouped.agg(
        n_queries=("query_id", "size"), n_pos=("n_pos", "sum"),
        n_reach_pos=("n_reach_pos", "sum"), n_win=("n_win", "sum"),
        n_reach_win=("n_reach_win", "sum"), n_pool_pairs=("pool_size", "sum"),
        n_phase2_zero_in_pool=("n_phase2_zero_in_pool", "sum"),
        n_phase2_zero_out_pool=("n_phase2_zero_out_pool", "sum"),
    ).reset_index()
    if out.empty:
        out = pd.DataFrame({c: pd.Series([], dtype=d) for c, d in COVERAGE_DTYPES.items()})
    return out.astype(COVERAGE_DTYPES)[list(COVERAGE_DTYPES)].reset_index(drop=True)


# ----------------------------------------------------------------------------
# The key stream — one date pool and one chunk in memory at a time.
# ----------------------------------------------------------------------------
def _negative_rank(seed: str, query_id: str, firma_key: str) -> int:
    text = json.dumps([seed, query_id, firma_key], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _sample_negatives(negatives: np.ndarray, seed: str, query_id: str, budget: int) -> np.ndarray:
    """The `budget` lowest-ranked negatives; `negatives` is firma_key-sorted."""
    if budget >= len(negatives):
        return negatives
    if budget == 0:
        return negatives[:0]
    ranks = np.fromiter((_negative_rank(seed, query_id, k) for k in negatives),
                        dtype=np.uint64, count=len(negatives))
    order = np.argsort(ranks, kind="stable")           # stable: ties keep firma_key order
    return negatives[order[:budget]]


def _key_frame(cols: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": pd.array(cols["query_id"], dtype="string"),
        "aa_tarihi": cols["aa_tarihi"].astype("datetime64[ns]"),
        "firma_key": pd.array(cols["firma_key"], dtype="string"),
        "grade": cols["grade"].astype(np.int8),
        "label_provenance": pd.Categorical.from_codes(cols["provenance"], dtype=PROVENANCE_DTYPE),
        "split": pd.array(cols["split"], dtype="string"),
        "period": pd.Categorical.from_codes(cols["period"], dtype=PERIOD_DTYPE),
    })


def _generate_keys(index: EventIndex, chosen: pd.DataFrame, rec: pd.DataFrame,
                   parts: pd.DataFrame, excluded: frozenset[str], max_rows: int,
                   *, budget: int | None, seed: str | None) -> Iterator[pd.DataFrame]:
    """budget None = the full pool (evaluation); otherwise sampled negatives (training)."""
    rec = rec.loc[rec["query_id"].isin(chosen["query_id"])]
    positives = {qid: (g["firma_key"].to_numpy(dtype=object), g["grade"].to_numpy(dtype=np.int8),
                       g["reachable"].to_numpy(dtype=bool))
                 for qid, g in rec.groupby("query_id", sort=False)}
    parts = parts.loc[parts["aa_kodu"].isin(chosen["aa_kodu"])]
    attended = {code: frozenset(g.to_numpy(dtype=object))
                for code, g in parts.groupby("aa_kodu", sort=False)["firma_key"]}
    no_positive = (np.array([], dtype=object), np.array([], dtype=np.int8), np.array([], dtype=bool))

    pending: list[dict[str, np.ndarray]] = []
    n_pending = 0
    order = chosen.sort_values(["aa_tarihi", "query_id"], kind="mergesort")
    for date, day in order.groupby("aa_tarihi", sort=True):
        pool_keys = _pool_keys(index, date, excluded)
        if (day["pool_size"].to_numpy() != len(pool_keys)).any():
            _fail(f"pool size on {pd.Timestamp(date).date()} is {len(pool_keys)}, not the "
                  f"ledger's; pass the same index and exclude to every call")
        stamp = pd.Timestamp(date).to_datetime64()
        for row in day.itertuples(index=False):
            keys, grades, reach = positives.get(row.query_id, no_positive)
            _check_reachable(keys, reach, pool_keys, f"for query {row.query_id}")
            pos_keys, pos_grades = keys[reach], grades[reach]
            negatives = pool_keys[~np.isin(pool_keys, pos_keys)]
            if budget is not None:
                negatives = _sample_negatives(negatives, seed, row.query_id, budget)
            selected = np.sort(np.concatenate([pos_keys, negatives]))
            m = len(selected)
            if m == 0:
                continue
            # Labels are attached only now, after the keys are fixed.
            at = pd.Index(pos_keys).get_indexer(selected)
            grade = np.zeros(m, dtype=np.int8)
            grade[at >= 0] = pos_grades[at[at >= 0]]
            went = attended.get(row.aa_kodu, frozenset())
            is_part = np.fromiter((k in went for k in selected), dtype=bool, count=m)
            provenance = np.select([grade == 2, grade == 1, is_part], [0, 1, 2], default=3)
            pending.append({
                "query_id": np.full(m, row.query_id, dtype=object),
                "aa_tarihi": np.full(m, stamp),
                "firma_key": selected,
                "grade": grade,
                "provenance": provenance.astype(np.int8),
                "split": np.full(m, row.split, dtype=object),
                "period": np.full(m, PERIODS.index(row.period), dtype=np.int8),
            })
            n_pending += m
            if n_pending >= max_rows:
                merged = {c: np.concatenate([p[c] for p in pending]) for c in pending[0]}
                start = 0
                while n_pending - start >= max_rows:
                    yield _key_frame({c: v[start:start + max_rows] for c, v in merged.items()})
                    start += max_rows
                rest = {c: v[start:] for c, v in merged.items()}
                n_pending -= start
                pending = [rest] if n_pending else []
    if n_pending:
        yield _key_frame({c: np.concatenate([p[c] for p in pending]) for c in pending[0]})

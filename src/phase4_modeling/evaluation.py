"""
evaluation.py  —  Phase-4 ranking metrics and the history-only baselines.

It implements the METRIC CONTRACT documented in query_universe.py, unchanged, and four
fixed baseline scorers that any trained ranker must beat. It trains no model, chooses no
budget, draws nothing and performs NO file I/O.

FLOW   iter_evaluation_keys -> features.attach_features -> score columns (score_baselines
       or a model) -> evaluate_stream -> per-query rows -> summarize_metrics
Several scorers are evaluated in ONE pass over the stream, so features are computed once.

BASELINES (score_baselines; higher score = call first; NaN = no evidence, ranked last)
  score_hash_random     SHA-256 rank of (seed, query_id, firma_key) — the floor
  score_recent_bid      -firm_days_since_last_bid
  score_bid_volume      firm_n_cars_bid_prior
  score_marka_affinity  aff_marka_n_prior

STREAM CONTRACT (evaluate_stream)
  * Rows arrive strictly ordered by (aa_tarihi, query_id, firma_key) — the key-stream order.
    Chunks are transport units: a query is complete only when query_id changes or the
    stream ends. A query that reappears, or any order violation, raises.
  * COMPLETENESS. Every non-quarantined ledger query of `periods` with pool_size > 0 must
    arrive exactly once with pool_size rows, and its grade >= 1 rows must equal its
    reachable positives in `reconciled` (keys and grades). Anything missing, extra or
    partial raises. Empty-pool queries have no rows and score 0 hits.
  * RANKING. T = the first min(K, pool_size) rows by score descending, NaN last, ties by
    firma_key ascending. Scores must be numeric and finite or NaN; +-inf raises.
  * Test is held out: periods containing 'test' require allow_test=True, as in
    iter_evaluation_keys.

METRICS (summarize_metrics; per period and scorer; the query_universe definitions)
  P = observed positives, R = P in the pool, W = the grade-2 firm, gains 0/1/3 by grade.
  Q_reach = non-quarantined queries with |R| > 0; Q_pos = non-quarantined with |P| > 0.
    cond_recall_macro@K   mean over Q_reach of |T n R| / |R|
    cond_recall_micro@K   sum |T n R| / sum |R|              over Q_reach
    ndcg@K                mean over Q_reach; discount 1/log2(rank+1); ideal from full pool
    e2e_recall@K          sum |T n P| / sum |P|              over Q_pos
    winner_hit@K          sum |T n W| / sum |W|              over Q_pos
    e2e_recall_all@K, winner_hit_all@K
                          conservative: quarantined positives join the denominator, 0 hits
    query_coverage, pair_coverage, winner_coverage      K-independent, over Q_pos
    n_queries, n_zero_positive, n_zero_reachable, n_empty_pool, n_quarantined   counts
  An empty denominator gives value NaN with n_queries 0. PRIMARY_K = 20; 10 and 50 are
  reported beside it.
"""
from __future__ import annotations

import re
import numbers
from typing import Iterable

import numpy as np
import pandas as pd

from phase4_modeling.query_universe import (GAIN_BY_GRADE, PERIOD_DTYPE, PERIODS,
                                            STATUS_DTYPE, _check_reconciled, _negative_rank,
                                            _positive_counts, _validate_ledger)

KS = (10, 20, 50)
PRIMARY_K = 20
RANDOM_SEED = "phase4-hash-random-baseline"
BASELINES = ("score_hash_random", "score_recent_bid", "score_bid_volume",
             "score_marka_affinity")
BASELINE_INPUTS = ["query_id", "firma_key", "firm_days_since_last_bid",
                   "firm_n_cars_bid_prior", "aff_marka_n_prior"]
STREAM_COLUMNS = ["query_id", "aa_tarihi", "firma_key", "grade"]
LEDGER_COUNTS = ["pool_size", "n_pos", "n_reach_pos", "n_win", "n_reach_win"]

QUERY_BASE_DTYPES = {
    "scorer": "string", "query_id": "string", "period": PERIOD_DTYPE, "status": STATUS_DTYPE,
    **{c: "int32" for c in LEDGER_COUNTS},
}
SUMMARY_DTYPES = {
    "period": PERIOD_DTYPE, "scorer": "string", "metric": "string", "k": "Int32",
    "value": "float64", "numerator": "float64", "denominator": "float64", "n_queries": "int64",
}

_GAIN = np.array([GAIN_BY_GRADE[g] for g in (0, 1, 2)], dtype=np.float64)
_HITS = re.compile(r"^hits_at_(\d+)$")


def _fail(message: str) -> None:
    raise ValueError(f"[evaluation] {message}")


def _require_frame(df: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[evaluation] {name} must be a pandas DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(f"{name} is missing required columns: {missing}")


def metric_columns(ks: tuple[int, ...]) -> list[str]:
    """Per-query metric column names, K ascending."""
    return [f"{m}_at_{k}" for k in ks for m in ("hits", "win_hits", "ndcg")]


# ----------------------------------------------------------------------------
# Baselines.
# ----------------------------------------------------------------------------
def score_baselines(features: pd.DataFrame, *, seed: str = RANDOM_SEED) -> pd.DataFrame:
    """The four BASELINES as float64 columns on the index of `features`."""
    _require_frame(features, BASELINE_INPUTS, "features")
    if not isinstance(seed, str) or seed == "":
        _fail("seed must be a non-empty string")
    qid = features["query_id"].to_numpy(dtype=object)
    key = features["firma_key"].to_numpy(dtype=object)
    rank = np.fromiter((_negative_rank(seed, q, k) for q, k in zip(qid, key)),
                       dtype=np.uint64, count=len(features))

    def num(col: str) -> np.ndarray:
        return features[col].astype("float64").to_numpy()
    return pd.DataFrame({
        # a low hash rank is "drawn first", as for training negatives
        "score_hash_random": -(rank.astype(np.float64)),
        "score_recent_bid": -num("firm_days_since_last_bid"),
        "score_bid_volume": num("firm_n_cars_bid_prior"),
        "score_marka_affinity": num("aff_marka_n_prior"),
    }, index=features.index)


# ----------------------------------------------------------------------------
# Argument checks.
# ----------------------------------------------------------------------------
def _check_ks(ks) -> tuple[int, ...]:
    if not isinstance(ks, tuple) or not ks:
        _fail(f"ks must be a non-empty tuple of ints, got {ks!r}")
    for k in ks:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral) or k < 1:
            _fail(f"ks must hold ints >= 1, got {k!r}")
    if len(set(ks)) != len(ks):
        _fail("ks repeats a value")
    return tuple(sorted(int(k) for k in ks))


def _check_periods(periods, allow_test) -> tuple[str, ...]:
    if not isinstance(allow_test, (bool, np.bool_)):
        raise TypeError("[evaluation] allow_test must be a bool")
    if (not isinstance(periods, tuple) or not periods or len(set(periods)) != len(periods)
            or any(p not in PERIODS for p in periods)):
        _fail(f"periods must be a non-empty tuple of distinct {list(PERIODS)}, got {periods!r}")
    if "test" in periods and not allow_test:
        _fail("the test period is held out; pass allow_test=True for the final evaluation")
    return periods


def _check_score_columns(score_columns) -> tuple[str, ...]:
    if (not isinstance(score_columns, tuple) or not score_columns
            or not all(isinstance(c, str) and c for c in score_columns)
            or len(set(score_columns)) != len(score_columns)):
        _fail(f"score_columns must be a non-empty tuple of distinct names, got {score_columns!r}")
    clash = set(score_columns) & set(STREAM_COLUMNS)
    if clash:
        _fail(f"score_columns reuse key columns: {sorted(clash)}")
    return score_columns


def _scores(s: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_numeric_dtype(s.dtype):
        _fail(f"{name} must be numeric, got {s.dtype}")
    out = s.astype("float64").to_numpy()
    if np.isinf(out).any():
        _fail(f"{name} holds infinite scores")
    return out


# ----------------------------------------------------------------------------
# Per-query metrics.
# ----------------------------------------------------------------------------
def _rank_metrics(grade: np.ndarray, score: np.ndarray, ks: tuple[int, ...]) -> list[float]:
    """grade/score of one complete query, rows in firma_key order. Returns metric_columns values."""
    n = len(grade)
    nan = np.isnan(score)
    # lexsort: last key is primary. Row position breaks ties, i.e. firma_key ascending.
    order = np.lexsort((np.arange(n), -np.where(nan, 0.0, score), nan))
    top = grade[order]
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    ideal = np.sort(_GAIN[grade])[::-1]
    dcg = np.cumsum(_GAIN[top] * disc)
    idcg = np.cumsum(ideal * disc)
    hits = np.cumsum(top >= 1)
    wins = np.cumsum(top == 2)
    out = []
    for k in ks:
        m = min(k, n) - 1
        out += [int(hits[m]), int(wins[m]), dcg[m] / idcg[m] if idcg[m] > 0 else np.nan]
    return out


def _empty_metrics(ks: tuple[int, ...]) -> list[float]:
    out = []
    for _ in ks:
        out += [0, 0, np.nan]
    return out


def evaluate_stream(scored_chunks: Iterable[pd.DataFrame],
                    ledger: pd.DataFrame,
                    reconciled: pd.DataFrame,
                    *,
                    score_columns: tuple[str, ...],
                    periods: tuple[str, ...],
                    ks: tuple[int, ...] = KS,
                    allow_test: bool = False) -> pd.DataFrame:
    """
    Consume a scored full-pool key stream and return one row per (scorer, query) for every
    non-quarantined ledger query of `periods`: QUERY_BASE_DTYPES columns followed by
    metric_columns(ks). Sorted by scorer (in score_columns order), then query_id.
    Enforces the STREAM CONTRACT; see the module docstring.
    """
    score_columns = _check_score_columns(score_columns)
    periods = _check_periods(periods, allow_test)
    ks = _check_ks(ks)
    led = _validate_ledger(ledger)
    rec = _check_reconciled(reconciled, led)
    counts = _positive_counts(rec, led["query_id"])
    if any((counts[c] != led[c].to_numpy()).any() for c in counts):
        _fail("reconciled positive counts disagree with the ledger; build both from the same inputs")

    chosen = led.loc[led["status"].ne("quarantined") & led["period"].isin(periods)]
    info = chosen.set_index("query_id")
    reach = rec.loc[rec["reachable"] & rec["query_id"].isin(chosen["query_id"])]
    expected_pos = {q: dict(zip(g["firma_key"], g["grade"].astype(int)))
                    for q, g in reach.groupby("query_id", sort=False)}
    dates = dict(zip(info.index, info["aa_tarihi"].to_numpy().view("int64")))
    pool_size = dict(zip(info.index, info["pool_size"].to_numpy()))

    results: dict[str, list[list[float]]] = {}

    def finish(qid: str, day: int, key: np.ndarray, grade: np.ndarray,
               scores: list[np.ndarray]) -> None:
        if qid not in pool_size:
            _fail(f"query {qid} is not a non-quarantined {list(periods)} ledger query")
        if qid in results:
            _fail(f"query {qid} arrives twice; rows must be ordered by (aa_tarihi, query_id, firma_key)")
        if day != dates[qid]:
            _fail(f"query {qid} arrives with a date other than its ledger date")
        if len(key) != pool_size[qid]:
            _fail(f"query {qid} has {len(key)} rows, but its pool has {pool_size[qid]}; "
                  f"a partial or padded query group is never scored")
        positive = grade >= 1
        got = dict(zip(key[positive].tolist(), grade[positive].tolist()))
        if got != expected_pos.get(qid, {}):
            _fail(f"query {qid} positive rows disagree with its reachable positives")
        results[qid] = [_rank_metrics(grade, s, ks) for s in scores]

    carry = None                                     # the unfinished last group
    prev = None                                      # (day, qid, key) of the last row seen
    for chunk in scored_chunks:
        _require_frame(chunk, STREAM_COLUMNS + list(score_columns), "chunk")
        if len(chunk) == 0:
            continue
        a = chunk["aa_tarihi"]
        if not pd.api.types.is_datetime64_dtype(a.dtype) or isinstance(a.dtype, pd.DatetimeTZDtype) \
                or a.isna().any():
            _fail("chunk.aa_tarihi must be a non-null naive datetime64 column")
        g = chunk["grade"]
        if pd.api.types.is_bool_dtype(g.dtype) or not pd.api.types.is_integer_dtype(g.dtype) \
                or g.isna().any() or not g.isin([0, 1, 2]).all():
            _fail("chunk.grade must be integers in {0, 1, 2}")
        cols = {
            "day": a.astype("datetime64[ns]").to_numpy().view("int64"),
            "qid": chunk["query_id"].astype(object).to_numpy(),
            "key": chunk["firma_key"].astype(object).to_numpy(),
            "grade": g.to_numpy(dtype=np.int64),
            **{c: _scores(chunk[c], f"chunk.{c}") for c in score_columns},
        }
        if any(pd.api.types.infer_dtype(cols[c], skipna=False) != "string" for c in ("qid", "key")):
            _fail("chunk.query_id and chunk.firma_key must be non-null strings")

        day, qid, key = cols["day"], cols["qid"], cols["key"]
        if prev is not None:
            day = np.concatenate([[prev[0]], day])
            qid = np.concatenate([np.array([prev[1]], dtype=object), qid])
            key = np.concatenate([np.array([prev[2]], dtype=object), key])
        same_q = qid[1:] == qid[:-1]
        ok = (day[1:] > day[:-1]) | ((day[1:] == day[:-1])
                                     & ((qid[1:] > qid[:-1]) | (same_q & (key[1:] > key[:-1]))))
        if not ok.all():
            _fail("rows are not strictly ordered by (aa_tarihi, query_id, firma_key)")
        prev = (cols["day"][-1], cols["qid"][-1], cols["key"][-1])

        if carry is not None:
            cols = {c: np.concatenate([carry[c], cols[c]]) for c in cols}
        starts = np.flatnonzero(np.r_[True, cols["qid"][1:] != cols["qid"][:-1]])
        bounds = list(starts) + [len(cols["qid"])]
        for lo, hi in zip(bounds[:-2], bounds[1:-1]):     # all groups but the last are complete
            finish(cols["qid"][lo], int(cols["day"][lo]), cols["key"][lo:hi],
                   cols["grade"][lo:hi], [cols[c][lo:hi] for c in score_columns])
        carry = {c: v[bounds[-2]:] for c, v in cols.items()}
    if carry is not None:
        finish(carry["qid"][0], int(carry["day"][0]), carry["key"], carry["grade"],
               [carry[c] for c in score_columns])

    missing = [q for q, n in pool_size.items() if n > 0 and q not in results]
    if missing:
        _fail(f"{len(missing)} queries with a non-empty pool never arrived, e.g. {missing[:3]}; "
              f"the stream is incomplete")

    names = metric_columns(ks)
    frames = []
    for s, scorer in enumerate(score_columns):
        values = [results[q][s] if q in results else _empty_metrics(ks) for q in info.index]
        base = pd.DataFrame({
            "scorer": scorer, "query_id": info.index.to_numpy(dtype=object),
            "period": info["period"].to_numpy(), "status": info["status"].to_numpy(),
            **{c: info[c].to_numpy() for c in LEDGER_COUNTS},
        })
        metrics = pd.DataFrame(values, columns=names)
        frames.append(pd.concat([base, metrics], axis=1))
    out = pd.concat(frames, ignore_index=True)
    dtypes = {**QUERY_BASE_DTYPES,
              **{c: ("float64" if c.startswith("ndcg") else "int32") for c in names}}
    out = out.astype(dtypes)
    order = pd.Categorical(out["scorer"].astype(object), categories=list(score_columns))
    out = out.assign(_o=order.codes).sort_values(["_o", "query_id"], kind="mergesort")
    return out.drop(columns="_o").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------------
def _ratio(num: float, den: float, n: int) -> tuple[float, float, float, int]:
    return (num / den if den > 0 else np.nan), float(num), float(den), (n if den > 0 else 0)


def summarize_metrics(per_query: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """
    SUMMARY_DTYPES rows per (period, scorer, metric, k) for the periods present in
    per_query. Every scorer must cover every non-quarantined ledger query of those periods
    with the ledger's counts, or it raises. k is NA for K-independent rows.
    """
    _require_frame(per_query, list(QUERY_BASE_DTYPES), "per_query")
    ks = tuple(sorted(int(m.group(1)) for c in per_query.columns if (m := _HITS.match(c))))
    ks = _check_ks(ks)
    _require_frame(per_query, metric_columns(ks), "per_query")
    led = _validate_ledger(ledger)
    periods = [p for p in PERIODS if p in set(per_query["period"].astype(object))]
    scope = led.loc[led["period"].isin(periods)]
    quarantined = scope.loc[scope["status"].eq("quarantined")]
    evaluable = scope.loc[scope["status"].ne("quarantined")].set_index("query_id")

    rows = []
    for scorer, pq in per_query.groupby(per_query["scorer"].astype(object), sort=False):
        pq = pq.set_index(pq["query_id"].astype(object))
        if pq.index.duplicated().any() or set(pq.index) != set(evaluable.index):
            _fail(f"scorer {scorer} does not cover exactly the non-quarantined queries of {periods}")
        ref = evaluable.loc[pq.index]
        for c in LEDGER_COUNTS + ["period", "status"]:
            if (pq[c].astype(object).to_numpy() != ref[c].astype(object).to_numpy()).any():
                _fail(f"per_query.{c} disagrees with the ledger for scorer {scorer}")
        for period in periods:
            q = pq.loc[pq["period"].astype(object).eq(period)]
            quar = quarantined.loc[quarantined["period"].astype(object).eq(period)]
            reach = q.loc[q["n_reach_pos"] > 0]
            pos = q.loc[q["n_pos"] > 0]

            def add(metric, k, value):
                rows.append((period, scorer, metric, k, *value))
            add("n_queries", None, (float(len(q)), float(len(q)), np.nan, len(q)))
            add("n_zero_positive", None, (float((q["status"] == "zero_positive").sum()),) * 2 + (np.nan, len(q)))
            add("n_zero_reachable", None, (float((q["status"] == "zero_reachable").sum()),) * 2 + (np.nan, len(q)))
            add("n_empty_pool", None, (float((q["pool_size"] == 0).sum()),) * 2 + (np.nan, len(q)))
            add("n_quarantined", None, (float(len(quar)), float(len(quar)), np.nan, len(quar)))
            add("query_coverage", None, _ratio(len(reach), len(pos), len(pos)))
            add("pair_coverage", None, _ratio(pos["n_reach_pos"].sum(), pos["n_pos"].sum(), len(pos)))
            add("winner_coverage", None, _ratio(pos["n_reach_win"].sum(), pos["n_win"].sum(), len(pos)))
            for k in ks:
                hits, wins, ndcg = reach[f"hits_at_{k}"], reach[f"win_hits_at_{k}"], reach[f"ndcg_at_{k}"]
                add("cond_recall_macro", k,
                    _ratio((hits / reach["n_reach_pos"]).sum(), len(reach), len(reach)))
                add("cond_recall_micro", k, _ratio(hits.sum(), reach["n_reach_pos"].sum(), len(reach)))
                add("ndcg", k, _ratio(ndcg.sum(), len(reach), len(reach)))
                e_hits, e_wins = pos[f"hits_at_{k}"].sum(), pos[f"win_hits_at_{k}"].sum()
                add("e2e_recall", k, _ratio(e_hits, pos["n_pos"].sum(), len(pos)))
                add("winner_hit", k, _ratio(e_wins, pos["n_win"].sum(), len(pos)))
                add("e2e_recall_all", k, _ratio(e_hits, pos["n_pos"].sum() + quar["n_pos"].sum(),
                                                 len(pos) + len(quar)))
                add("winner_hit_all", k, _ratio(e_wins, pos["n_win"].sum() + quar["n_win"].sum(),
                                                len(pos) + len(quar)))
    out = pd.DataFrame(rows, columns=list(SUMMARY_DTYPES))
    out["k"] = pd.array([pd.NA if k is None else k for k in out["k"]], dtype="Int32")
    return out.astype(SUMMARY_DTYPES).reset_index(drop=True)

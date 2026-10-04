"""
ranker.py  —  the Phase-4 (Model A) LightGBM LambdaRank ranker: training, scoring and the
model artifact.

It turns a sampled training key stream with features into ONE ranker and scores any
feature frame with it. It chooses no budget, scores no metric, never sees the test period
and reads no data file. Its only file I/O is save_ranker / load_ranker on a caller path
(and reading the recorded source files for their SHA-256 provenance).

FLOW   iter_training_keys -> features.attach_features -> collect_training -> train_ranker
       iter_evaluation_keys -> features.attach_features -> score -> evaluation.evaluate_stream

MODEL INPUTS. Exactly features.FEATURE_COLUMNS (19), in that order; every other column of a
frame is ignored. veh_marka, veh_kasa_tipi and veh_lokasyon are categorical: a value is
coded by its position in the sorted VOCABULARY, and an unknown or NA value is missing
(NaN). The vocabulary is fitted on fit-period training rows only (fit_vocabulary) and is
REUSED UNCHANGED by the (fit, validation) refit, so validation-only categories are
missing in both selection and refit. Numeric inputs are float32; NA -> NaN, bool -> 0/1.

TRAINING COMPLETENESS (collect_training). Chunks are transport units: query groups are
reassembled across chunk boundaries. Rows must be strictly ordered by (aa_tarihi,
query_id, firma_key). Every reachable_positive ledger query of `periods` must arrive
exactly once with n_reach_pos + min(budget, pool_size - n_reach_pos) rows, its grade >= 1
rows must equal its reachable positives in `reconciled` (keys and grades), and its date
and period must match the ledger. Negative IDENTITIES are not re-derived — only their
count is checked; passing the same immutable EventIndex, exclude and seed to
iter_training_keys stays the caller's contract. Test rows never train.

OBJECTIVE. lambdarank with label_gain 0/1/3 (GAIN_BY_GRADE) and truncation level 20
(PRIMARY_K). No validation set, no early stopping, no bagging, no feature sampling;
deterministic=True, force_col_wise=True and a fixed thread count and seed. Groups are
bounded by MAX_GROUP_ROWS, our own guard, far above any budget + positives.

ARTIFACT (save_ranker / load_ranker). One canonical JSON file {payload, payload_sha256}.
The payload binds format, config, vocabulary, model inputs, categorical indices, label
gain, truncation level, provenance, model_string and model_sha256. load_ranker recomputes
both digests and re-checks every item against the model string and the Booster. Digests
detect accidental corruption; they are not security signatures.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from phase4_modeling import features as _features
from phase4_modeling import query_universe as _qu
from phase4_modeling.evaluation import PRIMARY_K
from phase4_modeling.features import FEATURE_COLUMNS
from phase4_modeling.query_universe import (GAIN_BY_GRADE, PERIODS, TRAINING_PERIODS,
                                            _check_reconciled, _int_arg, _positive_counts,
                                            _validate_ledger)

FORMAT = "phase4-ranker-v1"
MODEL_INPUTS = list(FEATURE_COLUMNS)
CATEGORICAL = ("veh_marka", "veh_kasa_tipi", "veh_lokasyon")
CATEGORICAL_INDEX = tuple(MODEL_INPUTS.index(c) for c in CATEGORICAL)
NUMERIC = tuple(c for c in MODEL_INPUTS if c not in CATEGORICAL)
LABEL_GAIN = tuple(GAIN_BY_GRADE[g] for g in sorted(GAIN_BY_GRADE))      # (0, 1, 3)
TRUNCATION_LEVEL = PRIMARY_K                                              # 20
MAX_GROUP_ROWS = 10_000
KEY_COLUMNS = ["query_id", "aa_tarihi", "firma_key", "grade", "period"]

if LABEL_GAIN != (0, 1, 3) or sorted(GAIN_BY_GRADE) != [0, 1, 2]:
    raise ValueError(f"[ranker] GAIN_BY_GRADE must be 0/1/3, got {GAIN_BY_GRADE}")


@dataclass(frozen=True)
class RankerConfig:
    """Everything that decides a model. The sampling fields must match the TrainingSet."""
    negatives_per_query: int
    sampling_seed: str
    num_boost_round: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    num_threads: int = 4
    random_state: int = 20260929


@dataclass(frozen=True)
class TrainingSet:
    """Complete query groups in stream order. Key strings are not kept."""
    numeric: np.ndarray                     # float32 (n_rows, len(NUMERIC))
    categorical: dict[str, pd.Categorical]  # raw strings, NA kept
    labels: np.ndarray                      # int8 grades 0/1/2
    group_sizes: np.ndarray                 # int32, one per query, stream order
    query_ids: tuple[str, ...]
    periods: tuple[str, ...]
    negatives_per_query: int
    sampling_seed: str
    keys_sha256: str                        # over every group's (query_id, firma_key, grade)

    @property
    def n_rows(self) -> int:
        return int(len(self.labels))

    @property
    def n_queries(self) -> int:
        return int(len(self.group_sizes))


@dataclass(frozen=True)
class Ranker:
    booster: lgb.Booster
    config: RankerConfig
    vocabulary: dict[str, tuple[str, ...]]
    provenance: dict
    model_string: str


def _fail(message: str) -> None:
    raise ValueError(f"[ranker] {message}")


def _require_frame(df: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[ranker] {name} must be a pandas DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(f"{name} is missing required columns: {missing}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


# ----------------------------------------------------------------------------
# Argument checks.
# ----------------------------------------------------------------------------
def _check_int(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"[ranker] {name} must be an int, got {value!r}")
    if value < minimum:
        _fail(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_seed(value, name: str) -> str:
    if not isinstance(value, str) or value == "":
        _fail(f"{name} must be a non-empty string")
    return value


def _check_config(config: RankerConfig) -> RankerConfig:
    if not isinstance(config, RankerConfig):
        raise TypeError("[ranker] config must be a RankerConfig")
    _check_int(config.negatives_per_query, "negatives_per_query", 0)
    _check_seed(config.sampling_seed, "sampling_seed")
    _check_int(config.num_boost_round, "num_boost_round", 1)
    _check_int(config.num_leaves, "num_leaves", 2)
    _check_int(config.min_data_in_leaf, "min_data_in_leaf", 1)
    _check_int(config.num_threads, "num_threads", 1)
    _check_int(config.random_state, "random_state", 0)
    lr = config.learning_rate
    if isinstance(lr, (bool, np.bool_)) or not isinstance(lr, float) or not (0 < lr < np.inf):
        _fail(f"learning_rate must be a positive finite float, got {lr!r}")
    return config


def _check_training_periods(periods) -> tuple[str, ...]:
    if not isinstance(periods, tuple) or periods not in TRAINING_PERIODS:
        _fail(f"training periods must be one of {list(TRAINING_PERIODS)}, got {periods!r}")
    return periods


def _check_vocabulary(vocabulary) -> dict[str, tuple[str, ...]]:
    """Exactly the CATEGORICAL keys; each a non-empty, sorted, distinct tuple of non-blank strings."""
    if not isinstance(vocabulary, dict) or set(vocabulary) != set(CATEGORICAL):
        _fail(f"vocabulary must be a dict with exactly the keys {list(CATEGORICAL)}")
    out = {}
    for col in CATEGORICAL:
        values = vocabulary[col]
        if not isinstance(values, (tuple, list)) or not values:
            _fail(f"vocabulary[{col}] must be a non-empty tuple of strings")
        if not all(isinstance(v, str) and v.strip() != "" for v in values):
            _fail(f"vocabulary[{col}] holds non-string or blank values")
        if list(values) != sorted(set(values)):
            _fail(f"vocabulary[{col}] must be sorted and distinct")
        out[col] = tuple(values)
    return out


# ----------------------------------------------------------------------------
# Input columns.
# ----------------------------------------------------------------------------
def _numeric_block(frame: pd.DataFrame) -> np.ndarray:
    """NUMERIC inputs as float32 (n, len(NUMERIC)); NA -> NaN, bool -> 0/1, +-inf raises."""
    out = np.empty((len(frame), len(NUMERIC)), dtype=np.float32)
    for j, col in enumerate(NUMERIC):
        s = frame[col]
        if pd.api.types.is_bool_dtype(s.dtype):
            if s.isna().any():
                _fail(f"{col} is boolean with nulls")
            out[:, j] = s.to_numpy(dtype=np.float32)
        elif pd.api.types.is_numeric_dtype(s.dtype):
            v = s.astype("float64").to_numpy(na_value=np.nan)
            if np.isinf(v).any():
                _fail(f"{col} holds infinite values")
            out[:, j] = v
        else:
            _fail(f"{col} must be numeric or boolean, got {s.dtype}")
    return out


def _category_values(frame: pd.DataFrame, col: str) -> np.ndarray:
    """String-or-NA object array; NA stays NA (it is missing, never a category)."""
    vals = frame[col].astype(object).to_numpy()
    present = ~pd.isna(vals)
    if not all(isinstance(v, str) for v in vals[present]):
        _fail(f"{col} holds non-string values")
    return np.where(present, vals, None)


def _codes(values: np.ndarray, vocab: tuple[str, ...]) -> np.ndarray:
    """Vocabulary position as float32; unknown or NA -> NaN."""
    at = pd.Index(vocab, dtype=object).get_indexer(values)
    return np.where(at >= 0, at, np.nan).astype(np.float32)


def _matrix(numeric: np.ndarray, cats: dict[str, np.ndarray],
            vocabulary: dict[str, tuple[str, ...]]) -> np.ndarray:
    n = len(numeric)
    x = np.empty((n, len(MODEL_INPUTS)), dtype=np.float32)
    for j, col in enumerate(MODEL_INPUTS):
        if col in CATEGORICAL:
            x[:, j] = _codes(cats[col], vocabulary[col])
        else:
            x[:, j] = numeric[:, NUMERIC.index(col)]
    return x


def encode(frame: pd.DataFrame, vocabulary: dict[str, tuple[str, ...]]) -> np.ndarray:
    """MODEL_INPUTS of `frame` as a float32 (n, 19) matrix; every other column is ignored."""
    _require_frame(frame, MODEL_INPUTS, "frame")
    vocab = _check_vocabulary(vocabulary)
    cats = {c: _category_values(frame, c) for c in CATEGORICAL}
    return _matrix(_numeric_block(frame), cats, vocab)


# ----------------------------------------------------------------------------
# Training set.
# ----------------------------------------------------------------------------
def collect_training(chunks: Iterable[pd.DataFrame],
                     ledger: pd.DataFrame,
                     reconciled: pd.DataFrame,
                     *,
                     negatives_per_query: int,
                     sampling_seed: str,
                     periods: tuple[str, ...] = ("fit",)) -> TrainingSet:
    """
    Reassemble a sampled training stream (iter_training_keys + attach_features, with the
    same budget, seed and periods) into complete query groups and check it against the
    ledger — see TRAINING COMPLETENESS. Raises on anything missing, repeated, partial,
    padded, unordered, mislabelled or from the test period.
    """
    budget = _check_int(negatives_per_query, "negatives_per_query", 0)
    seed = _check_seed(sampling_seed, "sampling_seed")
    periods = _check_training_periods(periods)
    led = _validate_ledger(ledger)
    rec = _check_reconciled(reconciled, led)
    counts = _positive_counts(rec, led["query_id"])
    if any((counts[c] != led[c].to_numpy()).any() for c in counts):
        _fail("reconciled positive counts disagree with the ledger; build both from the same inputs")

    chosen = led.loc[led["status"].eq("reachable_positive") & led["period"].isin(periods)]
    if chosen.empty:
        _fail(f"the ledger has no reachable_positive queries in {list(periods)}")
    info = chosen.set_index("query_id")
    n_reach = info["n_reach_pos"].to_numpy(dtype=np.int64)
    expected_rows = dict(zip(info.index, n_reach + np.minimum(
        budget, info["pool_size"].to_numpy(dtype=np.int64) - n_reach)))
    expected_day = dict(zip(info.index, info["aa_tarihi"].to_numpy().view("int64")))
    expected_period = dict(zip(info.index, info["period"].astype(object)))
    reach = rec.loc[rec["reachable"] & rec["query_id"].isin(chosen["query_id"])]
    expected_pos = {q: dict(zip(g["firma_key"], g["grade"].astype(int)))
                    for q, g in reach.groupby("query_id", sort=False)}

    hasher = hashlib.sha256()
    done: list[str] = []
    seen: set[str] = set()
    sizes: list[int] = []

    def finish(qid, day, period, key, grade) -> None:
        if qid not in expected_rows:
            _fail(f"query {qid} is not a reachable_positive {list(periods)} ledger query")
        if qid in seen:
            _fail(f"query {qid} arrives twice; rows must be ordered by (aa_tarihi, query_id, firma_key)")
        if (day != expected_day[qid]).any() or (period != expected_period[qid]).any():
            _fail(f"query {qid} arrives with a date or period other than its ledger row")
        if len(key) > MAX_GROUP_ROWS:
            _fail(f"query {qid} has {len(key)} rows, above MAX_GROUP_ROWS={MAX_GROUP_ROWS}")
        if len(key) != expected_rows[qid]:
            _fail(f"query {qid} has {len(key)} rows, but its budget gives {expected_rows[qid]}; "
                  f"a partial or padded query group never trains")
        positive = grade >= 1
        got = dict(zip(key[positive].tolist(), grade[positive].tolist()))
        if got != expected_pos.get(qid, {}):
            _fail(f"query {qid} positive rows disagree with its reachable positives")
        hasher.update(_canonical([qid, key.tolist(), grade.tolist()]))
        seen.add(qid)
        done.append(qid)
        sizes.append(len(key))

    numeric, labels = [], []
    cats: dict[str, list[pd.Categorical]] = {c: [] for c in CATEGORICAL}
    carry = None                                     # key columns of the unfinished group
    prev = None                                      # (day, qid, key) of the last row seen
    for chunk in chunks:
        _require_frame(chunk, KEY_COLUMNS + MODEL_INPUTS, "chunk")
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
        period = chunk["period"].astype(object).to_numpy()
        if pd.isna(period).any() or not np.isin(period, np.array(PERIODS, dtype=object)).all():
            _fail(f"chunk.period must only hold {list(PERIODS)}")
        if (period == "test").any():
            _fail("the test period never trains")
        cols = {
            "day": a.astype("datetime64[ns]").to_numpy().view("int64"),
            "qid": chunk["query_id"].astype(object).to_numpy(),
            "key": chunk["firma_key"].astype(object).to_numpy(),
            "grade": g.to_numpy(dtype=np.int64),
            "period": period,
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

        # inputs are converted per chunk; only the light key columns wait for the group end
        numeric.append(_numeric_block(chunk))
        labels.append(cols["grade"].astype(np.int8))
        for c in CATEGORICAL:
            v = _category_values(chunk, c)
            seen_values = pd.Index(sorted(set(v[pd.notna(v)].tolist())), dtype=object)
            cats[c].append(pd.Categorical(v, categories=seen_values))

        if carry is not None:
            cols = {c: np.concatenate([carry[c], cols[c]]) for c in cols}
        starts = np.flatnonzero(np.r_[True, cols["qid"][1:] != cols["qid"][:-1]])
        bounds = list(starts) + [len(cols["qid"])]
        for lo, hi in zip(bounds[:-2], bounds[1:-1]):     # all groups but the last are complete
            finish(cols["qid"][lo], cols["day"][lo:hi], cols["period"][lo:hi],
                   cols["key"][lo:hi], cols["grade"][lo:hi])
        carry = {c: v[bounds[-2]:] for c, v in cols.items()}
    if carry is not None:
        finish(carry["qid"][0], carry["day"], carry["period"], carry["key"], carry["grade"])

    missing = [q for q in expected_rows if q not in seen]
    if missing:
        _fail(f"{len(missing)} training queries never arrived, e.g. {missing[:3]}; "
              f"the stream is incomplete")
    return TrainingSet(
        numeric=np.concatenate(numeric),
        categorical={c: pd.api.types.union_categoricals(v, sort_categories=True)
                     for c, v in cats.items()},
        labels=np.concatenate(labels),
        group_sizes=np.array(sizes, dtype=np.int32),
        query_ids=tuple(done), periods=periods,
        negatives_per_query=budget, sampling_seed=seed,
        keys_sha256=hasher.hexdigest(),
    )


def fit_vocabulary(training: TrainingSet) -> dict[str, tuple[str, ...]]:
    """Sorted distinct non-NA values of each CATEGORICAL input; fit-only training sets only."""
    if not isinstance(training, TrainingSet):
        raise TypeError("[ranker] training must be a TrainingSet")
    if training.periods != ("fit",):
        _fail("the vocabulary is fitted on ('fit',) training only; the refit reuses the "
              "selected model's vocabulary")
    vocab = {}
    for col in CATEGORICAL:
        cat = training.categorical[col]
        used = np.unique(cat.codes[cat.codes >= 0])
        vocab[col] = tuple(sorted(cat.categories[used].tolist()))
    return _check_vocabulary(vocab)


# ----------------------------------------------------------------------------
# Training and scoring.
# ----------------------------------------------------------------------------
def _params(config: RankerConfig) -> dict:
    return {
        "objective": "lambdarank",
        "label_gain": list(LABEL_GAIN),
        "lambdarank_truncation_level": TRUNCATION_LEVEL,
        "metric": "None",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "feature_fraction": 1.0,
        "num_threads": config.num_threads,
        "seed": config.random_state,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _source_hashes() -> dict[str, str]:
    """SHA-256 of the recorded Phase-4 sources that decide what a model sees."""
    return {f"phase4_modeling/{Path(p).name}": _sha256(Path(p).read_bytes())
            for p in (__file__, _features.__file__, _qu.__file__)}


def train_ranker(training: TrainingSet,
                 config: RankerConfig,
                 *,
                 vocabulary: dict[str, tuple[str, ...]] | None = None) -> Ranker:
    """
    Train one lambdarank Booster for exactly config.num_boost_round rounds. vocabulary=None
    fits it (fit-only training); the (fit, validation) refit must pass the selected model's
    vocabulary, which is reused unchanged.
    """
    if not isinstance(training, TrainingSet):
        raise TypeError("[ranker] training must be a TrainingSet")
    config = _check_config(config)
    if (config.negatives_per_query != training.negatives_per_query
            or config.sampling_seed != training.sampling_seed):
        _fail("config negatives_per_query / sampling_seed disagree with the training set")
    if vocabulary is None:
        vocab = fit_vocabulary(training)
    else:
        vocab = _check_vocabulary(vocabulary)
    if training.group_sizes.max() > MAX_GROUP_ROWS:
        _fail(f"a training group exceeds MAX_GROUP_ROWS={MAX_GROUP_ROWS}")

    cats = {c: np.asarray(training.categorical[c].astype(object)) for c in CATEGORICAL}
    cats = {c: np.where(pd.isna(v), None, v) for c, v in cats.items()}
    x = _matrix(training.numeric, cats, vocab)
    data = lgb.Dataset(x, label=training.labels, group=training.group_sizes,
                       feature_name=MODEL_INPUTS, categorical_feature=list(CATEGORICAL_INDEX),
                       free_raw_data=True)
    booster = lgb.train(_params(config), data, num_boost_round=config.num_boost_round)
    del data, x
    if booster.num_trees() != config.num_boost_round:
        _fail(f"training stopped after {booster.num_trees()} of {config.num_boost_round} "
              f"rounds (no admissible split); this config cannot be used")
    provenance = {
        "periods": list(training.periods),
        "negatives_per_query": training.negatives_per_query,
        "sampling_seed": training.sampling_seed,
        "n_rows": training.n_rows,
        "n_queries": training.n_queries,
        "n_positive_rows": int((training.labels >= 1).sum()),
        "keys_sha256": training.keys_sha256,
        "vocabulary_source": "fitted" if vocabulary is None else "given",
        "source_sha256": _source_hashes(),
        "lightgbm_version": lgb.__version__,
        "num_trees": booster.num_trees(),
    }
    model_string = booster.model_to_string()
    ranker = Ranker(booster=booster, config=config, vocabulary=vocab,
                    provenance=provenance, model_string=model_string)
    _check_ranker(ranker)
    return ranker


def score(ranker: Ranker, frame: pd.DataFrame, *, num_iteration: int | None = None) -> np.ndarray:
    """
    float64 scores, one per row of `frame` in row order (higher = call first). Rows are
    scored independently, so results do not depend on chunking. num_iteration uses the
    first 1..num_trees trees; None uses all.
    """
    if not isinstance(ranker, Ranker):
        raise TypeError("[ranker] ranker must be a Ranker")
    n_trees = ranker.booster.num_trees()
    it = n_trees if num_iteration is None else _check_int(num_iteration, "num_iteration", 1)
    if it > n_trees:
        _fail(f"num_iteration {it} exceeds the model's {n_trees} trees")
    x = encode(frame, ranker.vocabulary)
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.asarray(ranker.booster.predict(x, num_iteration=it,
                                            num_threads=ranker.config.num_threads),
                     dtype=np.float64)
    if out.shape != (len(frame),) or not np.isfinite(out).all():
        _fail("the booster returned non-finite or misshaped scores")
    return out


# ----------------------------------------------------------------------------
# Artifact.
# ----------------------------------------------------------------------------
_PARAM_LINE = re.compile(r"^\[([A-Za-z0-9_]+): (.*)\]$")
_PAYLOAD_KEYS = {"format", "config", "vocabulary", "model_inputs", "categorical_index",
                 "label_gain", "truncation_level", "provenance", "model_string", "model_sha256"}
_PROVENANCE_KEYS = {"periods", "negatives_per_query", "sampling_seed", "n_rows", "n_queries",
                    "n_positive_rows", "keys_sha256", "vocabulary_source", "source_sha256",
                    "lightgbm_version", "num_trees"}


def _model_facts(model_string: str) -> tuple[dict[str, str], dict[str, str]]:
    """(header key=value lines, [name: value] parameter lines) of a LightGBM model string."""
    lines = model_string.splitlines()
    header = {}
    for line in lines:
        if line.startswith("Tree="):
            break
        if "=" in line:
            k, v = line.split("=", 1)
            header[k] = v
    params = {}
    if "parameters:" in lines:
        for line in lines[lines.index("parameters:") + 1:]:
            if line == "end of parameters":
                break
            m = _PARAM_LINE.match(line)
            if m:
                params[m.group(1)] = m.group(2)
    return header, params


def _check_ranker(ranker: Ranker) -> None:
    """Internal consistency of config, vocabulary, provenance, model string and Booster."""
    config = _check_config(ranker.config)
    vocab = _check_vocabulary(ranker.vocabulary)
    prov = ranker.provenance
    if not isinstance(prov, dict) or set(prov) != _PROVENANCE_KEYS:
        _fail(f"provenance must hold exactly {sorted(_PROVENANCE_KEYS)}")
    if tuple(prov["periods"]) not in TRAINING_PERIODS:
        _fail("provenance periods are not a training period set")
    if (prov["negatives_per_query"] != config.negatives_per_query
            or prov["sampling_seed"] != config.sampling_seed):
        _fail("provenance sampling disagrees with the config")
    if prov["lightgbm_version"] != lgb.__version__:
        _fail(f"model built with LightGBM {prov['lightgbm_version']}, running {lgb.__version__}")
    booster = ranker.booster
    if booster.feature_name() != MODEL_INPUTS:
        _fail("booster feature order differs from MODEL_INPUTS")
    n_trees = booster.num_trees()
    if not (n_trees == prov["num_trees"] == config.num_boost_round):
        _fail("booster tree count disagrees with provenance / config")

    header, params = _model_facts(ranker.model_string)
    if not header.get("objective", "").split(" ")[0] == "lambdarank":
        _fail(f"model objective is {header.get('objective')!r}, not lambdarank")
    try:
        gains = tuple(float(v) for v in params.get("label_gain", "").split(","))
        trunc = int(params.get("lambdarank_truncation_level", ""))
    except ValueError:
        _fail("model string lacks label_gain / lambdarank_truncation_level")
    if gains != tuple(float(g) for g in LABEL_GAIN) or trunc != TRUNCATION_LEVEL:
        _fail(f"model label_gain {gains} / truncation {trunc} differ from the contract")
    if header.get("feature_names", "").split(" ") != MODEL_INPUTS:
        _fail("model string feature_names differ from MODEL_INPUTS")
    infos = header.get("feature_infos", "").split(" ")
    if len(infos) != len(MODEL_INPUTS):
        _fail("model string feature_infos do not cover MODEL_INPUTS")
    for j, (col, info) in enumerate(zip(MODEL_INPUTS, infos)):
        if info == "none":
            continue
        if j in CATEGORICAL_INDEX:
            try:
                used = [int(v) for v in info.split(":")]
            except ValueError:
                _fail(f"model treats categorical input {col} as numeric")
            # LightGBM lists the category values seen in training; -1 is its missing sentinel
            if any(u < -1 or u >= len(vocab[col]) for u in used):
                _fail(f"model categories of {col} fall outside its vocabulary")
        elif not info.startswith("["):
            _fail(f"model treats numeric input {col} as categorical")


def _payload(ranker: Ranker) -> dict:
    return {
        "format": FORMAT,
        "config": dataclasses.asdict(ranker.config),
        "vocabulary": {c: list(v) for c, v in ranker.vocabulary.items()},
        "model_inputs": list(MODEL_INPUTS),
        "categorical_index": list(CATEGORICAL_INDEX),
        "label_gain": list(LABEL_GAIN),
        "truncation_level": TRUNCATION_LEVEL,
        "provenance": ranker.provenance,
        "model_string": ranker.model_string,
        "model_sha256": _sha256(ranker.model_string.encode("utf-8")),
    }


def save_ranker(ranker: Ranker, path, *, overwrite: bool = False) -> str:
    """Write the canonical artifact; returns its payload_sha256. Never overwrites by default."""
    if not isinstance(ranker, Ranker):
        raise TypeError("[ranker] ranker must be a Ranker")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError("[ranker] overwrite must be a bool")
    _check_ranker(ranker)
    payload = _payload(ranker)
    digest = _sha256(_canonical(payload))
    text = _canonical({"payload": payload, "payload_sha256": digest}).decode("utf-8") + "\n"
    path = Path(path)
    if path.exists() and not overwrite:
        _fail(f"{path} exists; pass overwrite=True to replace it")
    with open(path, "w" if overwrite else "x", encoding="utf-8") as fh:
        fh.write(text)
    return digest


def load_ranker(path) -> Ranker:
    """Read an artifact and re-check both digests and every contract item; raises on any mismatch."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"payload", "payload_sha256"}:
        _fail("artifact must hold exactly payload and payload_sha256")
    payload = doc["payload"]
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        _fail(f"artifact payload must hold exactly {sorted(_PAYLOAD_KEYS)}")
    if _sha256(_canonical(payload)) != doc["payload_sha256"]:
        _fail("artifact payload digest mismatch")
    if payload["format"] != FORMAT:
        _fail(f"artifact format {payload['format']!r} is not {FORMAT!r}")
    model_string = payload["model_string"]
    if not isinstance(model_string, str) or _sha256(model_string.encode("utf-8")) != payload["model_sha256"]:
        _fail("artifact model digest mismatch")
    if payload["model_inputs"] != MODEL_INPUTS:
        _fail("artifact model inputs differ from MODEL_INPUTS")
    if payload["categorical_index"] != list(CATEGORICAL_INDEX):
        _fail("artifact categorical indices differ from the contract")
    if payload["label_gain"] != list(LABEL_GAIN) or payload["truncation_level"] != TRUNCATION_LEVEL:
        _fail("artifact label gain / truncation level differ from the contract")
    cfg = payload["config"]
    if not isinstance(cfg, dict) or set(cfg) != {f.name for f in dataclasses.fields(RankerConfig)}:
        _fail("artifact config fields differ from RankerConfig")
    vocab = payload["vocabulary"]
    if not isinstance(vocab, dict):
        _fail("artifact vocabulary must be a mapping")
    ranker = Ranker(booster=lgb.Booster(model_str=model_string), config=RankerConfig(**cfg),
                    vocabulary=_check_vocabulary(vocab), provenance=payload["provenance"],
                    model_string=model_string)
    _check_ranker(ranker)
    return ranker

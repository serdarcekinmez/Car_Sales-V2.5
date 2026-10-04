"""
candidate_pool.py  —  the point-in-time known-firm pool for one auction date.

For one auction date it answers: "which firms could we already have known about
before this auction opened, and what kind of evidence did we have for each?"

One function serves three callers — serving (which firms to rank), training (where
negatives are drawn from) and evaluation (which firms count as reachable) — so the
three can never silently disagree about the candidate universe.

CUTOFF. `as_of` is the auction day's midnight. Evidence is admissible only if its
effective time is STRICTLY BEFORE it. Nothing from the auction day itself is used — in
particular not that auction's own participation list, which is unknown when calls are
planned.

ADMISSIBLE EVIDENCE:
  * bid           — clean_reel_db, effective time e = max(aa_tarihi, bid_timestamp)
  * participation — attendance at an earlier auction (cleaned AA Katılım), aa_tarihi
  * directory     — firma_olusturma_tarihi from the firm registry

BID TIMING. aa_tarihi and bid_timestamp are NOT equivalent. The 2026-09-22 audit of
clean_reel_db found two bids with aa_tarihi 2025-03-13 but bid_timestamp 2025-04-02
10:38 — dating by aa_tarihi alone admitted them 20 days early. A bid therefore counts
only from e = max(aa_tarihi, bid_timestamp): for a midnight cutoff c, e < c exactly
when aa_tarihi < c AND bid_timestamp < c, so neither a late bid on an earlier auction
nor an early-stamped bid on a later auction leaks. The same e drives bid counts,
first/last bid and the active window. This is availability under the supplied dates
only; no ingestion-time feed exists and none is assumed.

PRECISION AND RECENCY. Evidence keeps full nanosecond precision; nothing is floored for
ordering. Participation is a recorded auction date (midnight), not an inferred
attendance time. The days_since_* outputs are CALENDAR-day differences,
floor(as_of / day) - floor(evidence / day): evidence from 23:59 yesterday is 1 day old.

DATE INPUTS are strict — a wrong date must fail, not shift the pool quietly:
  * bid aa_tarihi and bid_timestamp, and participation aa_tarihi, must be non-null for
    every nonblank firm key. aa_tarihi is a day field and must be a midnight. There is
    no fallback from bid_timestamp to aa_tarihi.
  * A null directory date is simply no directory evidence; a malformed one raises.
  * Numbers and booleans are never read as epoch offsets. Timezone-aware or mixed
    timezone values are rejected, never converted. Unparseable or out-of-nanosecond-
    range values raise. Nothing is normalized.
  * Rows with a blank firm key are dropped before their dates are checked.

LANES — descriptive metadata, mutually exclusive, first match wins. A lane NEVER
removes a known firm; inactivity is not closure.
  A  at least one bid with e in [as_of - ACTIVE_WINDOW_DAYS, as_of)  (fixed 365 days)
  B  not A, but some bid or participation before as_of
  C  not A or B, but a directory date before as_of
Pool MEMBERSHIP is monotone in as_of (once known, always known) for a fixed registry
and exclusion set. Lane membership is NOT: a firm can move C -> A -> B -> A.

known_since is the EARLIEST admissible evidence time from any source, at full
precision. known_via names the source of that earliest time; the priority
bid > participation > directory only breaks EXACT timestamp ties.

ASSUMPTION — directory dates. Lane C treats firma_olusturma_tarihi as the date the
firm's record was created. That is unverified; it could be a migration or backfill
stamp. Null dates are never admitted (fail closed). More generally the registry is a
PRESENT-DAY snapshot: its identity folding, conflict collapse and is_internal flag
describe the export, not any past date. Filtering event dates does NOT make that
snapshot historically valid.

LIMITATIONS
  * No purchase evidence. clean_sales carries no sale date. A firm known only through
    a completed purchase gets NO knownness claim before its directory date — and for
    13 of the 30 purchase-only firms in the current export that date is later than
    their first purchase, or missing.
  * No administrative eligibility exists in any source. A future feed would arrive
    through `exclude` (it can only remove firms) — never as a knownness lane.
  * Target-side positives are never inserted. A firm that bid on a car but has no
    earlier evidence is simply absent: a coverage miss, not a pool member.

Internal firms (UÇAR / CARINA family) never appear. Every event firm must exist in the
registry; an absent firm's internal status is unknown, so we raise instead of guessing.

This module performs no file I/O and WRITES NOTHING. Callers load the inputs with the
existing helpers and pass DataFrames in.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from common.normalize import ascii_key, ascii_key_series

ACTIVE_WINDOW_DAYS = 365

BID_COLUMNS = ["aa_tarihi", "bid_timestamp", "firma_adi"]      # clean_reel_db
PARTICIPATION_COLUMNS = ["aa_tarihi", "firma_key"]             # clean_participation
REGISTRY_COLUMNS = ["firma_key", "firma_olusturma_tarihi", "is_internal"]
PAIR_COLUMNS = ["firma_key", "aa_tarihi"]

LANE_DTYPE = pd.CategoricalDtype(["A", "B", "C"])
KNOWN_VIA_DTYPE = pd.CategoricalDtype(["bid", "participation", "directory"])

# is_active_365d mirrors lane A: a bid within the fixed ACTIVE_WINDOW_DAYS window.
# days_since_* are calendar-day differences; known_since keeps full precision.
POOL_DTYPES = {
    "firma_key": "string",
    "lane": LANE_DTYPE,
    "known_since": "datetime64[ns]",
    "days_since_known": "Int32",
    "known_via": KNOWN_VIA_DTYPE,
    "is_active_365d": "bool",
    "days_since_last_bid": "Int32",                 # NA = no prior bid
    "days_since_last_participation": "Int32",       # NA = no prior participation
    "total_bids_prior": "Int32",
}
POOL_COLUMNS = list(POOL_DTYPES)

_DAY_NS = 86_400 * 1_000_000_000
_NAT = np.iinfo(np.int64).min           # the int64 view of NaT
_FAR = np.iinfo(np.int64).max           # stands in for "no date" when taking a minimum


@dataclass(frozen=True)
class EventIndex:
    """
    Per-firm sorted evidence times (int64 ns), built once and reused for every as_of.
    bid_dates holds one effective time e = max(aa_tarihi, bid_timestamp) per BID (not
    per auction), so a prefix count is also the number of prior bids. All arrays are
    read-only.
    """
    bid_dates: dict[str, np.ndarray]
    part_dates: dict[str, np.ndarray]
    dir_date: dict[str, int]               # non-null directory dates only
    registry_keys: frozenset[str]
    internal: frozenset[str]


# ----------------------------------------------------------------------------
# Input checks — fail loudly here rather than produce a quietly wrong pool.
# ----------------------------------------------------------------------------
def _fail(message: str) -> None:
    raise ValueError(f"[candidate_pool] {message}")


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[candidate_pool] {name} must be a pandas DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(f"{name} is missing required columns: {missing}")


def _to_ns(values: pd.Series, name: str, *, allow_null: bool,
           require_midnight: bool = False) -> np.ndarray:
    """
    Strict date column -> int64 nanoseconds (null -> _NAT). Empty columns of any dtype
    pass. Nulls, including an all-null numeric or boolean column, pass only when
    allow_null. A non-null value that parses to no date (e.g. '' or 'NaT') is invalid,
    never a null. Numbers and booleans are never read as epoch offsets, timezones are
    rejected rather than converted, and unparseable or out-of-nanosecond-range values
    raise. Nothing is normalized.
    """
    dtype = values.dtype
    null = values.isna().to_numpy(dtype=bool)
    if isinstance(dtype, pd.DatetimeTZDtype):
        _fail(f"{name} must be timezone-naive, got {dtype}")
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        if not null.all():
            _fail(f"{name} holds {dtype} values; numbers are never read as dates")
        if not allow_null and null.any():
            _fail(f"{name} has {int(null.sum())} null dates; evidence must be dated")
        return np.full(len(values), _NAT, dtype=np.int64)
    if pd.api.types.is_timedelta64_dtype(dtype):
        _fail(f"{name} holds durations, not dates")
    if not pd.api.types.is_datetime64_dtype(dtype):
        present = values.to_numpy(dtype=object)[~null]
        if any(isinstance(v, (bool, np.bool_, numbers.Number)) for v in present):
            _fail(f"{name} holds numeric or boolean values; numbers are never read as dates")
        if any(getattr(v, "tzinfo", None) is not None for v in present):
            _fail(f"{name} holds timezone-aware values; they are rejected, never converted")
    try:
        s = pd.to_datetime(values, errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"[candidate_pool] {name} holds invalid dates: {exc}") from exc
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        _fail(f"{name} must be timezone-naive, got {s.dtype}")
    if not pd.api.types.is_datetime64_dtype(s.dtype):
        _fail(f"{name} did not parse to naive dates (mixed timezones?), got {s.dtype}")
    try:
        s = s.astype("datetime64[ns]")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"[candidate_pool] {name} holds dates outside the nanosecond "
                         f"range: {exc}") from exc
    ns = np.asarray(s.to_numpy(dtype="datetime64[ns]")).view("int64")
    missing = ns == _NAT
    coerced = missing & ~null
    if coerced.any():
        _fail(f"{name} holds {int(coerced.sum())} non-null values that are not dates "
              f"(e.g. '' or 'NaT')")
    if not allow_null and missing.any():
        _fail(f"{name} has {int(missing.sum())} null dates; evidence must be dated")
    if require_midnight:
        bad = ~missing & (ns % _DAY_NS != 0)
        if bad.any():
            _fail(f"{name} holds {int(bad.sum())} non-midnight values; it is a day field")
    return ns


def _check_normalized(keys: pd.Series, name: str) -> None:
    """Keys must already be in the common ascii_key form."""
    bad = keys.ne(ascii_key_series(keys)).fillna(False)
    if bad.any():
        _fail(f"{name} holds {int(bad.sum())} keys not in ascii_key form, "
              f"e.g. {keys[bad].head(3).tolist()}")


def _check_in_registry(keys: Iterable[str], registry_keys: frozenset[str], name: str) -> None:
    missing = sorted(set(keys) - registry_keys)
    if missing:
        _fail(f"{len(missing)} nonblank {name} keys are absent from the registry, "
              f"e.g. {missing[:5]}. Their internal status is unknown and is never assumed; "
              f"build the registry from the same inputs.")


def _check_as_of(as_of) -> int:
    if as_of is None or isinstance(as_of, (bool, np.bool_, numbers.Number)):
        _fail(f"as_of must be a date, got {as_of!r}")
    try:
        ts = pd.Timestamp(as_of)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"[candidate_pool] as_of is not a valid date: {as_of!r}") from exc
    if ts is pd.NaT:
        _fail("as_of is NaT")
    if ts.tz is not None:
        _fail("as_of must be timezone-naive")
    if ts != ts.normalize():
        _fail(f"as_of must be an auction-day midnight, got {ts}")
    try:
        return int(ts.value)
    except OverflowError as exc:
        raise ValueError(f"[candidate_pool] as_of is outside the nanosecond range: {ts}") from exc


def _check_exclude(exclude: Iterable[str] | None) -> frozenset[str]:
    """Caller-supplied exclusions. None means none — never a default list."""
    if exclude is None:
        return frozenset()
    if isinstance(exclude, str):
        raise TypeError("[candidate_pool] exclude must be a collection of firma_key strings, "
                        "not a single string")
    keys = frozenset(exclude)
    bad = sorted(str(k) for k in keys if not isinstance(k, str) or k == "" or ascii_key(k) != k)
    if bad:
        _fail(f"exclude holds keys not in ascii_key form (they would silently match nothing): {bad[:5]}")
    return keys


# ----------------------------------------------------------------------------
# Index construction.
# ----------------------------------------------------------------------------
def _group_sorted(keys: np.ndarray, ns: np.ndarray) -> dict[str, np.ndarray]:
    """Parallel key/date arrays -> {key: sorted read-only int64 array}."""
    if len(keys) == 0:
        return {}
    frame = pd.DataFrame({"k": keys, "t": ns}).sort_values(["k", "t"], kind="mergesort")
    out: dict[str, np.ndarray] = {}
    for key, grp in frame.groupby("k", sort=False)["t"]:
        arr = grp.to_numpy(dtype=np.int64, copy=True)
        arr.setflags(write=False)
        out[str(key)] = arr
    return out


def build_event_index(bids: pd.DataFrame,
                      participation: pd.DataFrame,
                      registry: pd.DataFrame) -> EventIndex:
    """
    Build the reusable evidence index.

    bids          — clean_reel_db rows: aa_tarihi, bid_timestamp, firma_adi (folded here
                    to firma_key, because Phase 1 does not emit firma_key for bids).
                    Each bid is indexed at e = max(aa_tarihi, bid_timestamp).
    participation — clean_participation rows: aa_tarihi, firma_key
    registry      — firm_registry rows: firma_key, firma_olusturma_tarihi, is_internal

    An empty source contributes no evidence; it never erases another source's.
    Blank firm keys in bids/participation are dropped (they name no firm) before their
    dates are checked; every remaining date follows the strict rules in _to_ns.
    """
    _require_columns(bids, BID_COLUMNS, "bids")
    _require_columns(participation, PARTICIPATION_COLUMNS, "participation")
    _require_columns(registry, REGISTRY_COLUMNS, "registry")

    # --- registry: the identity spine ---
    reg_keys = registry["firma_key"].astype("string")
    if reg_keys.isna().any() or reg_keys.eq("").any():
        _fail("registry has null or blank firma_key")
    _check_normalized(reg_keys, "registry.firma_key")
    if reg_keys.duplicated().any():
        _fail(f"registry firma_key is not unique, e.g. {reg_keys[reg_keys.duplicated()].head(3).tolist()}")
    internal_col = registry["is_internal"]
    if not pd.api.types.is_bool_dtype(internal_col) or internal_col.isna().any():
        _fail("registry.is_internal must be a non-null boolean column")
    registry_keys = frozenset(reg_keys.tolist())
    internal = frozenset(reg_keys[internal_col.astype(bool).to_numpy()].tolist())

    dir_ns = _to_ns(registry["firma_olusturma_tarihi"], "registry.firma_olusturma_tarihi",
                    allow_null=True)
    has_dir = dir_ns != _NAT
    dir_date = {str(k): int(d) for k, d in zip(reg_keys.to_numpy()[has_dir], dir_ns[has_dir])}

    # --- bids: fold the raw name with the common normalization ---
    bid_keys = ascii_key_series(bids["firma_adi"])
    keep = bid_keys.ne("").to_numpy()
    bid_keys = bid_keys[keep]
    auction_ns = _to_ns(bids["aa_tarihi"][keep], "bids.aa_tarihi", allow_null=False,
                        require_midnight=True)
    stamp_ns = _to_ns(bids["bid_timestamp"][keep], "bids.bid_timestamp", allow_null=False)
    bid_ns = np.maximum(auction_ns, stamp_ns)          # not known before BOTH have passed
    _check_in_registry(bid_keys.unique(), registry_keys, "bid")

    # --- participation: keys arrive already folded; verify, never re-fold silently ---
    part_keys = participation["firma_key"].astype("string").fillna("")
    _check_normalized(part_keys, "participation.firma_key")
    keep = part_keys.ne("").to_numpy()
    part_keys = part_keys[keep]
    part_ns = _to_ns(participation["aa_tarihi"][keep], "participation.aa_tarihi", allow_null=False,
                     require_midnight=True)
    _check_in_registry(part_keys.unique(), registry_keys, "participation")

    return EventIndex(
        bid_dates=_group_sorted(bid_keys.to_numpy(dtype=object), bid_ns),
        part_dates=_group_sorted(part_keys.to_numpy(dtype=object), part_ns),
        dir_date=dir_date,
        registry_keys=registry_keys,
        internal=internal,
    )


# ----------------------------------------------------------------------------
# Evidence and lanes — one code path shared by the pool and the pair audit.
# ----------------------------------------------------------------------------
def _evidence(index: EventIndex, keys: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """
    For each (key, cutoff) position: prior counts and first/last times, all < cutoff,
    at full precision. n_win counts bids in the fixed [cutoff - 365 days, cutoff).
    """
    n = len(keys)
    ev = {
        "n_bid": np.zeros(n, np.int64), "n_win": np.zeros(n, np.int64),
        "n_part": np.zeros(n, np.int64),
        "first_bid": np.full(n, _NAT), "last_bid": np.full(n, _NAT),
        "first_part": np.full(n, _NAT), "last_part": np.full(n, _NAT),
        "dir": np.full(n, _NAT),
    }
    if n == 0:
        return ev
    window_ns = ACTIVE_WINDOW_DAYS * _DAY_NS
    groups = pd.Series(np.arange(n)).groupby(keys, sort=False).indices
    for key, pos in groups.items():
        tt = t[pos]
        a = index.bid_dates.get(key)
        if a is not None:
            start = np.maximum(tt, _NAT + window_ns) - window_ns    # no int64 wrap-around
            hi = np.searchsorted(a, tt, side="left")               # bids strictly before t
            lo = np.searchsorted(a, start, side="left")            # window is left-inclusive
            ev["n_bid"][pos] = hi
            ev["n_win"][pos] = hi - lo
            ev["first_bid"][pos] = np.where(hi > 0, a[0], _NAT)
            ev["last_bid"][pos] = np.where(hi > 0, a[np.maximum(hi - 1, 0)], _NAT)
        p = index.part_dates.get(key)
        if p is not None:
            hi = np.searchsorted(p, tt, side="left")
            ev["n_part"][pos] = hi
            ev["first_part"][pos] = np.where(hi > 0, p[0], _NAT)
            ev["last_part"][pos] = np.where(hi > 0, p[np.maximum(hi - 1, 0)], _NAT)
        d = index.dir_date.get(key)
        if d is not None:
            ev["dir"][pos] = np.where(d < tt, d, _NAT)
    return ev


def _lanes(ev: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mutually exclusive lane per position (object array, None = unknown) and known mask."""
    has_bid, has_part, has_dir = ev["n_bid"] > 0, ev["n_part"] > 0, ev["dir"] != _NAT
    lane_a = ev["n_win"] > 0
    lane_b = ~lane_a & (has_bid | has_part)
    lane_c = ~lane_a & ~lane_b & has_dir
    lane = np.full(len(lane_a), None, dtype=object)
    lane[lane_a], lane[lane_b], lane[lane_c] = "A", "B", "C"
    return lane, lane_a | lane_b | lane_c


def _days_since(t: np.ndarray, x: np.ndarray) -> pd.arrays.IntegerArray:
    """Calendar-day difference floor(t/day) - floor(x/day); independent of intraday order."""
    missing = x == _NAT
    days = np.zeros(len(x), np.int64)
    days[~missing] = t[~missing] // _DAY_NS - x[~missing] // _DAY_NS
    return pd.arrays.IntegerArray(days.astype("int32"), missing)


# ----------------------------------------------------------------------------
# Public interface.
# ----------------------------------------------------------------------------
def candidate_pool_as_of(index: EventIndex,
                         as_of,
                         *,
                         exclude: Iterable[str] | None = None) -> pd.DataFrame:
    """
    The known-firm pool for one auction date: every non-internal registry firm with
    admissible evidence strictly before `as_of`, minus caller exclusions.

    known_since is the earliest admissible evidence time at full precision; known_via is
    its source, with bid > participation > directory breaking only exact ties.
    `exclude` is applied after lane assignment and only removes rows. It is never
    defaulted — historical evaluation must not silently apply a present-day list.
    Returns POOL_COLUMNS with POOL_DTYPES, sorted by firma_key.
    """
    t0 = _check_as_of(as_of)
    excluded = _check_exclude(exclude)

    keys = np.array(sorted(index.registry_keys - index.internal), dtype=object)
    t = np.full(len(keys), t0, dtype=np.int64)
    ev = _evidence(index, keys, t)
    lane, known = _lanes(ev)
    keep = known & ~np.isin(keys, np.array(sorted(excluded), dtype=object))

    ev = {k: v[keep] for k, v in ev.items()}
    keys, t, lane = keys[keep], t[keep], lane[keep]

    starts = np.stack([ev["first_bid"], ev["first_part"], ev["dir"]])
    known_since = np.where(starts == _NAT, _FAR, starts).min(axis=0)
    known_via = np.full(len(keys), None, dtype=object)  # source of the minimum; later wins ties
    known_via[ev["dir"] == known_since] = "directory"
    known_via[ev["first_part"] == known_since] = "participation"
    known_via[ev["first_bid"] == known_since] = "bid"

    pool = pd.DataFrame({
        "firma_key": pd.array(keys, dtype="string"),
        "lane": pd.Categorical(lane, dtype=LANE_DTYPE),
        "known_since": known_since.view("datetime64[ns]"),
        "days_since_known": _days_since(t, known_since),
        "known_via": pd.Categorical(known_via, dtype=KNOWN_VIA_DTYPE),
        "is_active_365d": ev["n_win"] > 0,
        "days_since_last_bid": _days_since(t, ev["last_bid"]),
        "days_since_last_participation": _days_since(t, ev["last_part"]),
        "total_bids_prior": pd.array(ev["n_bid"].astype("int32"), dtype="Int32"),
    })
    return pool[POOL_COLUMNS].reset_index(drop=True)


def classify_pairs(index: EventIndex,
                   pairs: pd.DataFrame,
                   *,
                   exclude: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Audit / coverage helper. For each (firma_key, aa_tarihi) pair — typically the
    observed bidder pairs of some cars — would that firm have been in the pool built
    for that auction date? Returns a copy of `pairs` plus:
      lane      — A/B/C, or missing when unreachable
      reachable — True when the firm is in that date's pool
    Uses the same evidence and lane code as candidate_pool_as_of.
    """
    _require_columns(pairs, PAIR_COLUMNS, "pairs")
    excluded = _check_exclude(exclude)

    keys_s = pairs["firma_key"].astype("string")
    if keys_s.isna().any() or keys_s.eq("").any():
        _fail("pairs has null or blank firma_key")
    _check_normalized(keys_s, "pairs.firma_key")
    _check_in_registry(keys_s.unique(), index.registry_keys, "pairs")
    t = _to_ns(pairs["aa_tarihi"], "pairs.aa_tarihi", allow_null=False, require_midnight=True)

    keys = keys_s.to_numpy(dtype=object)
    lane, known = _lanes(_evidence(index, keys, t))
    blocked = np.isin(keys, np.array(sorted(index.internal | excluded), dtype=object))
    reachable = known & ~blocked
    lane[~reachable] = None

    out = pairs.copy()
    out["lane"] = pd.Categorical(lane, dtype=LANE_DTYPE)
    out["reachable"] = reachable
    return out

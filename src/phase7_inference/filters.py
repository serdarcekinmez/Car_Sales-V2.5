"""
filters.py  —  Phase 7: versioned customer eligibility filters over a ranked firm pool.

Filters are business rules, kept outside the model: they never change a score or the
order of the model's ranking. They only REMOVE firms (hard exclusions); the call list is
then the first top_n of the remaining firms in model order, so an excluded firm is
backfilled from further down the pool. No reranking or preference weighting exists in
this version. There is no expression language: only the fields below, each validated.

RULES (FILTER_VERSION fixes their meaning; rules_sha256 fixes their values)
  top_n               int 1..MAX_TOP_N, call-list length (default DEFAULT_TOP_N)
  exclude_firm_keys   firma_key values (ascii_key form) the customer does not want
  firm_cities         keep only firms whose CURRENT member-directory city is in this list
                      (matched after ascii_key folding). The directory is a present-day
                      snapshot: it is not proof of the firm's city on a past date, and a
                      vehicle's location is never used as a firm's region. A firm with no
                      directory city is excluded with reason city_unknown.
  active_365d_only    keep only firms with a bid in the 365 days before the prediction date
                      (point-in-time pool flag). When the history ends long before the
                      prediction date this mostly reflects missing data, not inactivity;
                      the caller must say so (predict.py does).

PUBLIC API
  parse_filters(spec)                -> FilterSpec (None / {} = defaults)
  describe_filters(spec)             -> JSON-safe {"version", "rules", "rules_sha256"}
  apply_filters(ranked, spec)        -> (selected, excluded_ahead, summary)
"""
from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.normalize import ascii_key

FILTER_VERSION = "phase7-filters-v1"
DEFAULT_TOP_N = 20
MAX_TOP_N = 200
MAX_LIST = 500
RULE_FIELDS = ("top_n", "exclude_firm_keys", "firm_cities", "active_365d_only")
REASONS = ("excluded_by_customer", "city_unknown", "city_not_selected", "no_bid_365d")
RANKED_COLUMNS = ["firma_key", "rank_original", "sehir", "firm_is_active_365d"]


def _fail(msg: str) -> None:
    raise ValueError(f"[filters] {msg}")


@dataclass(frozen=True)
class FilterSpec:
    top_n: int = DEFAULT_TOP_N
    exclude_firm_keys: tuple[str, ...] = ()
    firm_cities: tuple[str, ...] = ()
    active_365d_only: bool = False


def _strings(value, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{name} must be a list of strings")
    if len(value) > MAX_LIST:
        _fail(f"{name} holds more than {MAX_LIST} values")
    if not all(isinstance(v, str) and v.strip() for v in value):
        _fail(f"{name} must hold non-blank strings")
    return tuple(sorted(set(v.strip() for v in value)))


def parse_filters(spec: Mapping | FilterSpec | None) -> FilterSpec:
    """A validated FilterSpec; unknown fields and wrong types raise."""
    if isinstance(spec, FilterSpec):
        spec = {f: getattr(spec, f) for f in RULE_FIELDS}
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping):
        raise TypeError("[filters] filters must be a mapping or None")
    unknown = sorted(str(k) for k in spec if k not in RULE_FIELDS)
    if unknown:
        _fail(f"unknown filter fields {unknown}; allowed: {list(RULE_FIELDS)}")
    top_n = spec.get("top_n", DEFAULT_TOP_N)
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, numbers.Integral) \
            or not 1 <= top_n <= MAX_TOP_N:
        _fail(f"top_n must be an int in 1..{MAX_TOP_N}")
    keys = _strings(spec.get("exclude_firm_keys", ()), "exclude_firm_keys")
    bad = [k for k in keys if ascii_key(k) != k]
    if bad:
        _fail(f"exclude_firm_keys must be firma_key values (ascii_key form), got {bad[:5]}")
    cities = _strings(spec.get("firm_cities", ()), "firm_cities")
    folded = tuple(sorted({ascii_key(c) for c in cities}))
    if "" in folded:
        _fail("firm_cities holds a value with no letters or digits")
    active = spec.get("active_365d_only", False)
    if not isinstance(active, (bool, np.bool_)):
        _fail("active_365d_only must be a bool")
    return FilterSpec(top_n=int(top_n), exclude_firm_keys=keys, firm_cities=folded,
                      active_365d_only=bool(active))


def describe_filters(spec: FilterSpec) -> dict:
    rules = {"top_n": spec.top_n, "exclude_firm_keys": list(spec.exclude_firm_keys),
             "firm_cities": list(spec.firm_cities), "active_365d_only": spec.active_365d_only}
    text = json.dumps({"version": FILTER_VERSION, "rules": rules}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return {"version": FILTER_VERSION, "rules": rules,
            "rules_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def apply_filters(ranked: pd.DataFrame, spec: FilterSpec
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    ranked: the FULL scored pool in model order (rank_original 1..n). Returns
      selected        the first top_n eligible rows, model order kept, plus rank (1..top_n)
      excluded_ahead  excluded rows ranked above the last selected row (or all excluded
                      rows when the list is short), with their reasons
      summary         JSON-safe counts per reason over the whole pool, pool size, list
                      length and exclusion keys that matched no pool firm
    """
    if not isinstance(spec, FilterSpec):
        raise TypeError("[filters] spec must be a FilterSpec (use parse_filters)")
    missing = [c for c in RANKED_COLUMNS if c not in ranked.columns]
    if missing:
        _fail(f"ranked lacks {missing}")
    order = ranked["rank_original"].to_numpy()
    if len(order) and not np.array_equal(order, np.arange(1, len(order) + 1)):
        _fail("ranked must be in model order with rank_original 1..n")

    n = len(ranked)
    keys = ranked["firma_key"].astype(str).to_numpy()
    city = ranked["sehir"].astype("string").fillna("").map(ascii_key).to_numpy(dtype=object)
    reasons = {
        "excluded_by_customer": np.isin(keys, np.array(spec.exclude_firm_keys, dtype=object)),
        "city_unknown": np.zeros(n, bool), "city_not_selected": np.zeros(n, bool),
        "no_bid_365d": np.zeros(n, bool),
    }
    if spec.firm_cities:
        reasons["city_unknown"] = city == ""
        reasons["city_not_selected"] = (city != "") & ~np.isin(
            city, np.array(spec.firm_cities, dtype=object))
    if spec.active_365d_only:
        reasons["no_bid_365d"] = ~ranked["firm_is_active_365d"].to_numpy(dtype=bool)
    excluded = np.zeros(n, bool)
    for r in REASONS:
        excluded |= reasons[r]

    eligible = np.flatnonzero(~excluded)
    take = eligible[:spec.top_n]
    selected = ranked.iloc[take].copy()
    selected.insert(0, "rank", np.arange(1, len(take) + 1))
    cut = int(take[-1]) if len(take) == spec.top_n else n - 1
    ahead = np.flatnonzero(excluded[:cut + 1])
    excluded_ahead = ranked.iloc[ahead].copy()
    excluded_ahead["reasons"] = [[r for r in REASONS if reasons[r][i]] for i in ahead]
    summary = {
        "pool_size": n,
        "n_eligible": int(len(eligible)),
        "n_selected": int(len(take)),
        "excluded_by_reason": {r: int(reasons[r].sum()) for r in REASONS},
        "exclusion_keys_not_in_pool": sorted(set(spec.exclude_firm_keys) - set(keys)),
        "order": "model order kept; hard exclusions only; excluded firms backfilled from "
                 "the remaining pool; no reranking",
    }
    return selected.reset_index(drop=True), excluded_ahead.reset_index(drop=True), summary

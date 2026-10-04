"""
anomaly_flags.py  —  Step 4: spot the bad rows.

Two kinds of work:
  * test auction codes (TST*/TAA*) — the ONLY rows Phase 1 is allowed to DROP.
    Real codes look like CAA####. Matching is case-insensitive (the data holds a
    lowercase 'tst2002', which a case-sensitive match would silently miss).
  * extreme / impossible values — these we only FLAG (km<=0, km>1e6, eks out of
    range, >500 bids on a single lot). Nothing here is dropped; flags ride along
    in the parquet so they can be filtered or inspected later.
"""
from __future__ import annotations

import pandas as pd

# Test codes to drop; real codes to keep. Both matched case-insensitively.
# Non-capturing group so str.contains doesn't warn about match groups.
TEST_CODE_RE = r"^\s*(?:TST|TAA)"
REAL_CODE_RE = r"^\s*CAA\d{4}\s*$"


def is_test_code(code_series: pd.Series) -> pd.Series:
    """Boolean mask: rows whose auction code is a TST*/TAA* test code."""
    s = code_series.astype("string")
    return s.str.contains(TEST_CODE_RE, case=False, regex=True, na=False)


def drop_test_codes(df: pd.DataFrame, code_col: str, table: str,
                    log: list[dict]) -> pd.DataFrame:
    """Drop TST/TAA test-code rows from one table and log how many fell."""
    from common.normalize import add_log
    mask = is_test_code(df[code_col])
    n = int(mask.sum())
    add_log(log, "step4_test_codes", "drop",
            f"{table}: removed {n} TST/TAA test-code rows (case-insensitive)",
            rows_in=len(df), rows_out=len(df) - n)
    return df.loc[~mask].copy()


def km_flags(km_clean: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(km_nonpositive_flag, km_extreme_flag) — km<=0 and km>1,000,000."""
    nonpos = (km_clean <= 0).fillna(False).astype("int8")
    extreme = (km_clean > 1_000_000).fillna(False).astype("int8")
    return nonpos, extreme


def eks_puani_flag(eks_clean: pd.Series, lo: float = 0.0, hi: float = 10.0) -> pd.Series:
    """eks_puani_out_of_range_flag — expertise score outside [0, 10]."""
    out = ((eks_clean < lo) | (eks_clean > hi)).fillna(False).astype("int8")
    return out


def bids_per_query_extreme_flag(query_id: pd.Series, threshold: int = 500) -> pd.Series:
    """
    Flag (do NOT drop) bids that belong to a lot with > threshold bids.
    Some lots genuinely attract thousands of bids (e.g. CAA0050); we keep them
    but mark them so their weight can be checked downstream.
    """
    counts = query_id.map(query_id.value_counts())
    return (counts > threshold).fillna(False).astype("int8")

"""
participation.py  —  the auction "who showed up" list.

Phase 2 needs the pool of firms that PARTICIPATED in each auction code (from the
AA Katılım workbook) so every car in that auction can be scored against the firms
that were actually in the room. We do NOT invent a new firm identity here: each
firm is reduced to the very same Phase-1 ASCII firm key (Turkish-folded, upper,
alphanumerics only) that the cleaning phase used.

IMPORTANT — what "participation" does and does not mean:
  Participation is attendance at an AUCTION, not interest in a CAR. This file
  contains no plate and no lot reference. A firm registered for an auction may
  never have looked at most of its lots, so crossing participants with every car
  (done later, in labels.py) produces a convenient TRAINING negative, not proof
  that the firm considered and rejected that car. Historical participation is
  also not the pre-auction callable firm universe.

Exclusion order (each count is measured on what SURVIVED the previous step):
  1. test auctions (TST*/TAA*) — the shared Phase-1 rule, so Phase 2 sees the
     same auction universe Phase 1 does
  2. internal / related-party firms (UÇAR / CARINA family)
  3. blank firm keys
  4. collapse repeat registrations to one row per (aa_kodu, firma_key)

This module is READ-ONLY on Datasets/raw/ and writes NOTHING: it returns the
participation table and its statistics in memory.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Reuse the Phase-1 primitives — do NOT re-implement firm-key, internal or
# test-code logic. Shared rules live in src/common and stay phase-agnostic.
from common.normalize import ascii_key_series
from common.entity_resolution import flag_internal
from common.anomaly_flags import is_test_code
from common.raw_files import RawFileError, find_single_raw

SHEET = "Rapor"  # every raw workbook keeps its data on a sheet called "Rapor"

# The AA Katılım file names its columns differently from the rest of the data.
AA_KODU_RAW = "Açık Artırma Kodu"
AA_TARIHI_RAW = "Açık Artırma Tarihi"
FIRMA_RAW = "Firma Kısa Unvanı"
# Optional: a source-system firm id. Used ONLY to audit identity ambiguity; it is
# never carried onto a deduplicated row and never replaces firma_key.
FIRMA_ID_RAW = "FirmaID"

# The frozen output contract: one row per firm-in-auction, in this exact order.
PARTICIPATION_COLUMNS = ["aa_kodu", "aa_tarihi", "firma_adi", "firma_key"]


def find_aa_katilim(raw_dir: Path) -> Path:
    """The single AA Katılım participation workbook in raw_dir (by filename keyword)."""
    try:
        return find_single_raw(raw_dir, "aa_katilim")
    except RawFileError as exc:
        raise SystemExit(f"[phase2] {exc}") from None


def load_aa_katilim(path: Path) -> pd.DataFrame:
    """Read the 'Rapor' sheet of the AA Katılım participation workbook."""
    df = pd.read_excel(path, sheet_name=SHEET)
    missing = [c for c in (AA_KODU_RAW, AA_TARIHI_RAW, FIRMA_RAW) if c not in df.columns]
    if missing:
        raise SystemExit(f"[phase2] AA Katılım is missing columns: {missing}")
    return df


def _identity_audit(raw_rows: pd.DataFrame, firma_key: pd.Series) -> dict:
    """Measure where the name-derived firm key disagrees with the source FirmaID."""
    if FIRMA_ID_RAW not in raw_rows.columns:
        # Say so explicitly: "no data" must never be reported as "no collisions".
        return {"available": False,
                "reason": f"column {FIRMA_ID_RAW!r} is not present in this export"}

    pair = pd.DataFrame({
        "firma_key": firma_key.values,
        "firma_id": raw_rows[FIRMA_ID_RAW].values,
        "firma_adi": raw_rows[FIRMA_RAW].astype("string").str.strip().values,
    })
    # A blank id tells us nothing, so it cannot prove or disprove a collision.
    id_txt = pair["firma_id"].astype("string").str.strip()
    pair = pair.loc[pair["firma_id"].notna() & id_txt.ne("") & id_txt.notna()]

    ids_per_key = pair.groupby("firma_key")["firma_id"].nunique()
    keys_per_id = pair.groupby("firma_id")["firma_key"].nunique()
    multi_id_keys = sorted(ids_per_key[ids_per_key > 1].index.tolist())

    collisions = []
    for key in multi_id_keys:
        grp = pair.loc[pair["firma_key"] == key]
        collisions.append({
            "firma_key": key,
            "firma_ids": sorted(grp["firma_id"].unique().tolist()),
            "firma_adi_variants": sorted(set(grp["firma_adi"].dropna().tolist())),
        })

    return {
        "available": True,
        "rows_with_firma_id": int(len(pair)),
        "n_distinct_firma_id": int(pair["firma_id"].nunique()),
        "n_distinct_firma_key": int(pair["firma_key"].nunique()),
        # One key seen under several ids => the fold MERGED distinct source records.
        "n_keys_with_multiple_ids": int(len(multi_id_keys)),
        # One id seen under several keys => the fold SPLIT one source record.
        "n_ids_with_multiple_keys": int((keys_per_id > 1).sum()),
        "collisions": collisions,
    }


def clean_participation(aa_raw: pd.DataFrame,
                        internal_lookup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tidy AA Katılım into one (aa_kodu, firma_key) row per firm-in-auction."""
    df = pd.DataFrame(index=aa_raw.index)
    df["aa_kodu"] = aa_raw[AA_KODU_RAW].astype("string").str.strip()
    df["aa_tarihi"] = pd.to_datetime(aa_raw[AA_TARIHI_RAW], errors="coerce")
    df["firma_adi"] = (aa_raw[FIRMA_RAW].astype("string").str.strip()
                       .str.replace(r"\s+", " ", regex=True))
    df["firma_key"] = ascii_key_series(aa_raw[FIRMA_RAW])

    rows_in = len(df)

    # ---- 1. test auctions: the same shared rule Phase 1 applies ----
    test_mask = is_test_code(df["aa_kodu"])
    n_test = int(test_mask.sum())
    df = df.loc[~test_mask]

    # ---- 2. internal firms, counted on what survived step 1 ----
    is_internal, _ = flag_internal(aa_raw.loc[df.index, FIRMA_RAW], internal_lookup)
    n_internal = int((is_internal == 1).sum())
    df = df.loc[is_internal == 0]

    # ---- 3. blank firm keys: null-safe, evaluated on the CURRENT frame ----
    blank_mask = df["firma_key"].fillna("").str.len().eq(0)
    n_blank = int(blank_mask.sum())
    df = df.loc[~blank_mask]

    # Audit identity BEFORE collapsing, while every registration is still visible.
    identity = _identity_audit(aa_raw.loc[df.index], df["firma_key"])

    # ---- 4. a firm can register many times for one auction; keep it once ----
    rows_before_dedup = len(df)
    df = (df.drop_duplicates(["aa_kodu", "firma_key"])
          .reset_index(drop=True))
    n_duplicates = rows_before_dedup - len(df)

    df = df[PARTICIPATION_COLUMNS]
    stats = {
        "rows_in": rows_in,
        "n_test_code_dropped": n_test,
        "n_internal_dropped": n_internal,
        "n_blank_key_dropped": n_blank,
        "n_duplicate_registrations_collapsed": n_duplicates,
        "rows_out": len(df),
        "n_auctions": int(df["aa_kodu"].nunique()),
        "n_firms": int(df["firma_key"].nunique()),
        "identity_audit": identity,
    }
    return df, stats


def participants_by_auction(participation: pd.DataFrame) -> dict[str, list[str]]:
    """Group the clean participation table into {aa_kodu: [firma_key, ...]}."""
    return {k: v.tolist()
            for k, v in participation.groupby("aa_kodu")["firma_key"]}

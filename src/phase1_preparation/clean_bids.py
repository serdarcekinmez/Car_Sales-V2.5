"""
clean_bids.py  —  build Datasets/interim/clean_reel_db.parquet (frozen schema).

One row per bid. Two things make this module non-trivial:

  * INTERNAL bids (UÇAR/CARINA) are REMOVED here and parked in
    removed_internal_bids.parquet, so no related-party bid survives into
    clean_reel_db.

  * is_winning_bid_reeldb is derived FROM THE SALES JOIN, not from Teklif.Durumu
    (crash-point #3: Durumu is 93.4% empty and lot-level, not bid-level). A bid
    wins iff its firm == the lot's actual buyer AND the lot is_sold.

bid_timestamp uses the ready-made 'Teklif Tarihi Tam' column (crash-point #4 of
the old code, which stitched date+time and mis-parsed ~63% of rows).

V2 SCHEMA DEVIATION FROM V1 (approved by Serdar, 14 -> 13 columns):
  * REMOVED internal_firm_flag — internal bids are deleted a few lines below, so
    on every surviving row the column was structurally constant 0 and carried no
    information. The independent residual_internal_matches() check remains the
    real safety guard. bid_timestamp_invalid_flag is deliberately KEPT: it is 0
    only because the current timestamps all parse, not because it can never fire.
"""
from __future__ import annotations

import pandas as pd

from common.normalize import ascii_key_series, standardize_plaka, add_flag, add_log
from common.entity_resolution import flag_internal
from common.anomaly_flags import bids_per_query_extreme_flag
# single canonical definition of the train/test cut — owned by clean_sales.py
from phase1_preparation.clean_sales import SPLIT_BOUNDARY

BID_TS_FORMAT = "%d.%m.%Y %H:%M:%S"

FROZEN_BIDS_COLUMNS = [
    # key
    "plaka", "aa_kodu", "aa_tarihi",
    # core
    "firma_adi", "teklif", "bid_timestamp", "query_id", "is_winning_bid_reeldb",
    # ledger
    "split", "dataset_version_id",
    # anomaly flags
    "bids_per_query_extreme_flag", "plaka_format_valid", "bid_timestamp_invalid_flag",
]


def build_sales_lookup(clean_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal (plaka, aa_kodu) -> (buyer_key, is_sold) lookup for the winner flag.
    buyer_key is the ASCII key of the actual buyer so the bid firm can be matched
    on the same folded form. Deduped to one row per lot (sold row wins).
    """
    look = pd.DataFrame({
        "plaka": clean_sales["plaka"],
        "aa_kodu": clean_sales["aa_kodu"],
        "buyer_key": ascii_key_series(clean_sales["alici_firma_kisa_unvani"]),
        "is_sold": clean_sales["is_sold"].astype("int8"),
    })
    look = (look.sort_values("is_sold")            # sold (1) sorts last -> kept
            .drop_duplicates(["plaka", "aa_kodu"], keep="last")
            .reset_index(drop=True))
    return look


def build_clean_bids(teklif: pd.DataFrame, sales_lookup: pd.DataFrame,
                     internal_lookup: pd.DataFrame, version_id: str,
                     flags: list[dict], log: list[dict]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build clean_reel_db (internal bids removed) plus removed_internal_bids.
    `teklif` must already have test-code rows dropped. Returns (kept, removed, stats).
    """
    df = pd.DataFrame(index=teklif.index)

    plaka, plaka_valid = standardize_plaka(teklif["Plaka"])
    df["plaka"] = plaka
    df["aa_kodu"] = teklif["AA Kodu"].astype("string").str.strip()
    df["aa_tarihi"] = pd.to_datetime(teklif["AA Tarihi"], errors="coerce")
    df["firma_adi"] = (teklif["Firma Adı"].astype("string").str.strip()
                       .str.replace(r"\s+", " ", regex=True))
    df["teklif"] = pd.to_numeric(teklif["Teklif"], errors="coerce")
    df["bid_timestamp"] = pd.to_datetime(teklif["Teklif Tarihi Tam"],
                                         format=BID_TS_FORMAT, errors="coerce")
    df["query_id"] = df["plaka"] + "::" + df["aa_kodu"]
    df["plaka_format_valid"] = plaka_valid
    df["bid_timestamp_invalid_flag"] = df["bid_timestamp"].isna().astype("int8")

    # ---- remove internal (sister-company) bids ----
    internal_flag, internal_group = flag_internal(teklif["Firma Adı"], internal_lookup)
    n_internal = int((internal_flag == 1).sum())
    removed = df.loc[internal_flag == 1].copy()
    removed["internal_group"] = internal_group.loc[removed.index].astype("string")
    removed["dataset_version_id"] = version_id
    add_log(log, "step5_internal", "remove",
            f"clean_reel_db: removed {n_internal} internal (UÇAR/CARINA) bid rows",
            rows_in=len(df), rows_out=len(df) - n_internal)

    df = df.loc[internal_flag == 0].copy()

    # ---- winner flag from the sales join (NOT from Teklif.Durumu) ----
    merged = df.merge(sales_lookup, on=["plaka", "aa_kodu"], how="left")
    merged.index = df.index
    firm_key = ascii_key_series(df["firma_adi"])
    df["is_winning_bid_reeldb"] = (
        (firm_key.values == merged["buyer_key"].values)
        & (merged["is_sold"].fillna(0).values == 1)
        & (merged["buyer_key"].fillna("").values != "")
    )
    df["is_winning_bid_reeldb"] = df["is_winning_bid_reeldb"].astype("int8")

    # ---- ledger + remaining flags ----
    df["split"] = pd.Series(
        ["train" if (pd.notna(d) and d < SPLIT_BOUNDARY) else "test"
         for d in df["aa_tarihi"]], index=df.index)
    df["dataset_version_id"] = version_id
    df["bids_per_query_extreme_flag"] = bids_per_query_extreme_flag(df["query_id"])

    add_flag(flags, "clean_reel_db", "is_winning_bid_reeldb",
             df["is_winning_bid_reeldb"] == 1, "bid firm == lot buyer AND lot is_sold")
    add_flag(flags, "clean_reel_db", "bids_per_query_extreme_flag",
             df["bids_per_query_extreme_flag"] == 1, "lot has > 500 bids (kept, flagged)")
    add_flag(flags, "clean_reel_db", "bid_timestamp_invalid_flag",
             df["bid_timestamp_invalid_flag"] == 1, "Teklif Tarihi Tam unparseable")
    add_flag(flags, "clean_reel_db", "plaka_format_invalid",
             df["plaka_format_valid"] == 0, "plate fails standard format")

    kept = _finalize(df)
    win = kept[kept["is_winning_bid_reeldb"] == 1]
    stats = {
        "n_rows": len(kept),
        "n_removed_internal": n_internal,
        "n_winning": int(len(win)),
        "n_winning_lots": int(win.groupby(["plaka", "aa_kodu"]).ngroups),
        "n_extreme_bid_rows": int((kept["bids_per_query_extreme_flag"] == 1).sum()),
        "n_extreme_lots": int(kept.loc[kept["bids_per_query_extreme_flag"] == 1]
                              .groupby(["plaka", "aa_kodu"]).ngroups),
        "n_bad_timestamp": int((kept["bid_timestamp_invalid_flag"] == 1).sum()),
        "n_train": int((kept["split"] == "train").sum()),
        "n_test": int((kept["split"] == "test").sum()),
    }
    return kept, removed, stats


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Select frozen columns in order and make object columns parquet-safe."""
    missing = [c for c in FROZEN_BIDS_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"[clean_bids] missing frozen columns: {missing}")
    out = df[FROZEN_BIDS_COLUMNS].copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out

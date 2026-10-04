"""
labels.py  —  the graded "who to call" training table.

For every car that went through an auction (a "query") we list every firm that
could plausibly have bought it, and give that (car, firm) pair a graded label:

  * grade 2 — the firm bid on this car AND is its supported auction winner
  * grade 1 — the firm bid on this car without qualifying auction-win evidence
  * grade 0 — the firm merely participated in that auction and never bid on this car

Candidates for a car = participants of the SAME auction  ∪  bidders on that car.
A firm appears at most once per car, carrying its strongest grade.

HOW SALE EVIDENCE IS ESTABLISHED (the important design point)
    A car can be listed twice in one auction. We never pick a "winning" sales row
    to represent it, because the duplicate listings can disagree about the vehicle
    itself (different stock records, different transmission) and choosing one would
    silently invent an answer. Instead we AGGREGATE: we look only at the COMPLETED
    sale records (is_sold, which Phase 1 derives from Satış Tutarı > 0) and ask
    whether they AGREE. Agreement gives a state; disagreement is reported and stops
    the run. Nothing here re-defines what "sold" means, and bid activity is never
    used to decide which sale or buyer is authoritative.

    auction_sale       every completed sale says "not off-auction"  -> grade 2 possible
    off_auction_sale   every completed sale says "off-auction"      -> grade 2 excluded
    no_completed_sale  the car has sales records but none completed -> grade 2 excluded
    missing_sales      no sales record at all (bid-only orphan car) -> grade 2 excluded

    Off-auction sales remain COMPLETED SALES. They are not "unsold"; their upstream
    winner flag is simply not accepted as a Model-A auction win.

This module labels only: no features, no eligibility, no sampling, no model. It
returns everything in memory and WRITES NOTHING — the orchestrator persists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

# Shared primitives — never re-implement the firm-key fold or the split boundary.
from common.normalize import ascii_key_series
from phase1_preparation.clean_sales import SPLIT_BOUNDARY

# The frozen one-row-per-(query, firm) output contract, in order.
OUTPUT_COLUMNS = [
    "query_id", "plaka", "aa_kodu", "aa_tarihi", "firma_key",
    "grade", "is_bid", "is_win", "candidate_source", "split", "has_positive",
]

# One row per car-in-auction. No vehicle attribute and no stock id is carried.
QUERY_COLUMNS = ["query_id", "plaka", "aa_kodu", "aa_tarihi", "sale_type_state"]

# One row per car-in-auction, recording what the sale evidence looked like.
QUERY_AUDIT_COLUMNS = [
    "query_id", "n_sales_rows", "is_duplicate_listing", "is_orphan_query",
    "n_completed_sales", "sale_type_state", "off_auction_values_observed",
    "n_distinct_completed_buyers", "ambiguity_reason",
]

SALE_TYPE_STATES = ("auction_sale", "off_auction_sale", "no_completed_sale", "missing_sales")


def _fail(message: str, offenders, evidence: pd.DataFrame | None = None) -> None:
    """Stop with an informative error naming the offending queries and the evidence."""
    ids = [str(q) for q in list(offenders)[:5]]
    text = (f"[phase2_labels] {message}\n"
            f"  offending query_id(s): {len(list(offenders))} total, first {len(ids)}: {ids}")
    if evidence is not None and len(evidence):
        text += f"\n  evidence:\n{evidence.head(10).to_string()}"
    raise ValueError(text)


def firma_key_of(names: pd.Series) -> pd.Series:
    """Reduce a firm-name column to the canonical Phase-1 ASCII firm key."""
    return ascii_key_series(names)


def arrow_schema() -> pa.Schema:
    """Describe the exact column types of the labeled_candidates output table."""
    return pa.schema([
        ("query_id", pa.string()), ("plaka", pa.string()), ("aa_kodu", pa.string()),
        ("aa_tarihi", pa.timestamp("ns")), ("firma_key", pa.string()),
        ("grade", pa.int8()), ("is_bid", pa.bool_()), ("is_win", pa.bool_()),
        ("candidate_source", pa.string()), ("split", pa.string()),
        ("has_positive", pa.bool_()),
    ])


def _check_query_dates(frame: pd.DataFrame, source: str) -> None:
    """Every query must carry exactly one usable auction date in this source."""
    if frame["aa_tarihi"].isna().any():
        bad = frame.loc[frame["aa_tarihi"].isna(), "query_id"].unique()
        _fail(f"{source}: missing auction date (aa_tarihi is NaT)", bad)
    n_dates = frame.groupby("query_id")["aa_tarihi"].nunique()
    bad = n_dates[n_dates > 1].index
    if len(bad):
        _fail(f"{source}: a query carries conflicting auction dates", bad,
              frame.loc[frame["query_id"].isin(bad[:3]), ["query_id", "aa_tarihi"]])


def _aggregate_sale_evidence(sales: pd.DataFrame) -> pd.DataFrame:
    """Derive each car's sale-type state by AGREEMENT over its completed sales only."""
    grouped = sales.groupby("query_id", sort=True)
    out = pd.DataFrame({"n_sales_rows": grouped.size()})
    completed = sales.loc[sales["is_sold"] == 1]
    out["n_completed_sales"] = (completed.groupby("query_id").size()
                                .reindex(out.index).fillna(0).astype("int16"))

    # The off-auction evidence must be present and unambiguous on completed sales.
    if len(completed) and completed["off_auction_sale_flag"].isna().any():
        bad = completed.loc[completed["off_auction_sale_flag"].isna(), "query_id"].unique()
        _fail("completed sale carries missing off-auction evidence", bad)

    def _observed(values: pd.Series) -> str:
        return "|".join(sorted({str(int(v)) for v in values}))   # deterministic order
    obs = completed.groupby("query_id")["off_auction_sale_flag"].apply(_observed)
    out["off_auction_values_observed"] = obs.reindex(out.index).fillna("")

    # Buyer identity is compared on the canonical key, never on display text.
    named = completed.loc[completed["buyer_key"].fillna("").str.len() > 0]
    out["n_distinct_completed_buyers"] = (named.groupby("query_id")["buyer_key"].nunique()
                                          .reindex(out.index).fillna(0).astype("int16"))

    # Conflicting completed-sale types are never silently resolved.
    conflict = out.index[out["off_auction_values_observed"] == "0|1"]
    if len(conflict):
        _fail("completed sales disagree on off-auction type", conflict,
              completed.loc[completed["query_id"].isin(conflict[:3]),
                            ["query_id", "off_auction_sale_flag", "satis_tutari_present"]]
              if "satis_tutari_present" in completed.columns else None)

    state = pd.Series("no_completed_sale", index=out.index, dtype="object")
    state = state.mask(out["off_auction_values_observed"] == "0", "auction_sale")
    state = state.mask(out["off_auction_values_observed"] == "1", "off_auction_sale")
    out["sale_type_state"] = state

    # A normal auction sale with two different buyers cannot be attributed. This
    # guard is independent of whether any bid happened to match one of them.
    ambiguous = out.index[(out["sale_type_state"] == "auction_sale")
                          & (out["n_distinct_completed_buyers"] > 1)]
    if len(ambiguous):
        _fail("auction_sale query has multiple distinct completed buyers", ambiguous,
              named.loc[named["query_id"].isin(ambiguous[:3]), ["query_id", "buyer_key"]])
    return out


def build_query_table(clean_sales: pd.DataFrame,
                      clean_bids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the one-row-per-car query table plus its per-query sale-evidence audit."""
    sales = pd.DataFrame({
        "plaka": clean_sales["plaka"].astype("string").str.strip(),
        "aa_kodu": clean_sales["aa_kodu"].astype("string").str.strip(),
        "aa_tarihi": pd.to_datetime(clean_sales["aa_tarihi"]),
        "is_sold": clean_sales["is_sold"].astype("int8"),
        "off_auction_sale_flag": clean_sales["off_auction_sale_flag"],
        "buyer_key": ascii_key_series(clean_sales["alici_firma_kisa_unvani"]),
    })
    blank = (sales["plaka"].fillna("").str.len() == 0) | (sales["aa_kodu"].fillna("").str.len() == 0)
    if blank.any():
        _fail("clean_sales row has a blank plaka or aa_kodu", ["<blank key>"] * int(blank.sum()))
    sales["query_id"] = sales["plaka"] + "::" + sales["aa_kodu"]
    _check_query_dates(sales, "clean_sales")

    evidence = _aggregate_sale_evidence(sales)

    keys = (sales.drop_duplicates("query_id")
            .set_index("query_id")[["plaka", "aa_kodu", "aa_tarihi"]])
    sales_q = keys.join(evidence["sale_type_state"]).reset_index()

    # ---- bid side: validate BEFORE any deduplication hides a conflict ----
    bids = pd.DataFrame({
        "query_id": clean_bids["query_id"].astype("string").str.strip(),
        "plaka": clean_bids["plaka"].astype("string").str.strip(),
        "aa_kodu": clean_bids["aa_kodu"].astype("string").str.strip(),
        "aa_tarihi": pd.to_datetime(clean_bids["aa_tarihi"]),
    })
    rebuilt = bids["plaka"] + "::" + bids["aa_kodu"]
    mismatch = bids["query_id"].ne(rebuilt)
    if mismatch.any():
        _fail("clean_reel_db query_id disagrees with plaka::aa_kodu",
              bids.loc[mismatch, "query_id"].unique(),
              bids.loc[mismatch, ["query_id", "plaka", "aa_kodu"]])
    _check_query_dates(bids, "clean_reel_db")

    shared = bids.merge(sales_q[["query_id", "aa_tarihi"]], on="query_id",
                        how="inner", suffixes=("_bid", "_sales"))
    disagree = shared["aa_tarihi_bid"].ne(shared["aa_tarihi_sales"])
    if disagree.any():
        _fail("sales and bid auction dates disagree for a shared query",
              shared.loc[disagree, "query_id"].unique(),
              shared.loc[disagree, ["query_id", "aa_tarihi_bid", "aa_tarihi_sales"]])

    orphan = (bids.loc[~bids["query_id"].isin(sales_q["query_id"])]
              .drop_duplicates("query_id").copy())
    orphan["sale_type_state"] = "missing_sales"

    query_table = pd.concat([sales_q, orphan[QUERY_COLUMNS]], ignore_index=True)
    query_table = query_table[QUERY_COLUMNS].reset_index(drop=True)

    # ---- the audit: what the evidence looked like, ambiguity included ----
    aud = evidence.reset_index()
    aud["is_duplicate_listing"] = aud["n_sales_rows"] > 1
    aud["is_orphan_query"] = False
    orph_aud = pd.DataFrame({
        "query_id": orphan["query_id"], "n_sales_rows": 0,
        "n_completed_sales": 0, "sale_type_state": "missing_sales",
        "off_auction_values_observed": "", "n_distinct_completed_buyers": 0,
        "is_duplicate_listing": False, "is_orphan_query": True,
    })
    audit = pd.concat([aud, orph_aud], ignore_index=True)

    reasons = []
    for dup, orp, nbuy, state in zip(audit["is_duplicate_listing"], audit["is_orphan_query"],
                                     audit["n_distinct_completed_buyers"], audit["sale_type_state"]):
        r = []
        if dup:
            r.append("duplicate_listing")
        if nbuy > 1:
            r.append("multiple_completed_sales_distinct_buyers")
        if orp or state == "missing_sales":
            r.append("missing_sales")
        reasons.append("|".join(sorted(r)))          # deterministic order
    audit["ambiguity_reason"] = reasons

    audit["n_sales_rows"] = audit["n_sales_rows"].astype("int16")
    audit["n_completed_sales"] = audit["n_completed_sales"].astype("int16")
    audit["n_distinct_completed_buyers"] = audit["n_distinct_completed_buyers"].astype("int16")
    audit = audit[QUERY_AUDIT_COLUMNS].sort_values("query_id").reset_index(drop=True)
    return query_table, audit


def build_bidder_pairs(clean_bids: pd.DataFrame,
                       query_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse bids to one (query, firm) row and gate the win flag on sale evidence."""
    df = pd.DataFrame({
        "query_id": clean_bids["query_id"].astype("string").str.strip(),
        "firma_key": firma_key_of(clean_bids["firma_adi"]),
        "win": clean_bids["is_winning_bid_reeldb"].astype("int8"),
    })
    blank = df["firma_key"].fillna("").str.len() == 0
    if blank.any():
        _fail("bid row has a blank firma_key", df.loc[blank, "query_id"].unique())

    # one row per (car, firm): did this firm ever win the lot it bid on?
    pairs = df.groupby(["query_id", "firma_key"], as_index=False)["win"].max()

    ref = query_table[["query_id", "aa_kodu", "sale_type_state"]]
    pairs = pairs.merge(ref, on="query_id", how="left", indicator=True)
    lost = pairs["_merge"].ne("both")
    if lost.any():
        _fail("bidder pair has no matching query (failed join)",
              pairs.loc[lost, "query_id"].unique())
    pairs = pairs.drop(columns="_merge")

    # A winner flag is only meaningful where a completed sale exists at all.
    unsupported = pairs.loc[(pairs["win"] == 1)
                            & pairs["sale_type_state"].isin(["no_completed_sale", "missing_sales"])]
    if len(unsupported):
        _fail("upstream winner flag on a query with no completed sale",
              unsupported["query_id"].unique(),
              unsupported[["query_id", "firma_key", "sale_type_state"]])

    # grade-2 evidence: won the lot AND every completed sale was a normal auction sale.
    pairs["is_win"] = (pairs["win"] == 1) & (pairs["sale_type_state"] == "auction_sale")

    winners = pairs.loc[pairs["is_win"]].groupby("query_id")["firma_key"].nunique()
    many = winners[winners > 1].index
    if len(many):
        _fail("more than one final winning firm for a query", many,
              pairs.loc[pairs["query_id"].isin(many[:3]) & pairs["is_win"],
                        ["query_id", "firma_key"]])
    return pairs[["query_id", "aa_kodu", "firma_key", "is_win"]]


def assign_grade(is_bid: pd.Series, is_win: pd.Series) -> pd.Series:
    """Turn the bid/win booleans into the graded relevance label 0, 1 or 2."""
    bid = is_bid.to_numpy(dtype=bool)
    win = is_win.to_numpy(dtype=bool)
    return pd.Series(np.where(win, 2, np.where(bid, 1, 0)).astype("int8"),
                     index=is_bid.index, dtype="int8")


def assign_source(is_participant: pd.Series, is_bid: pd.Series) -> pd.Series:
    """Label where each candidate came from: participation, bid, or both."""
    src = pd.Series("participation", index=is_participant.index, dtype="object")
    src = src.mask(~is_participant & is_bid, "bid")
    src = src.mask(is_participant & is_bid, "both")
    return src


def add_split(aa_tarihi: pd.Series) -> pd.Series:
    """Mark each row train (auction before the boundary) or test; reject undated rows."""
    d = pd.to_datetime(aa_tarihi)
    if d.isna().any():
        raise ValueError("[phase2_labels] add_split: aa_tarihi contains NaT; "
                         f"{int(d.isna().sum())} undated row(s) cannot be split")
    return pd.Series(np.where(d < SPLIT_BOUNDARY, "train", "test"),
                     index=aa_tarihi.index, dtype="object")


def mark_has_positive(query_id: pd.Series, grade: pd.Series) -> pd.Series:
    """Flag every row whose car has at least one bidder (any grade >= 1)."""
    return (grade >= 1).groupby(query_id).transform("max").astype(bool)


def candidate_block_for_auction(queries_aa: pd.DataFrame,
                                participants_aa: list[str],
                                bids_aa: pd.DataFrame) -> pd.DataFrame:
    """Build every graded (car, firm) candidate row for ONE auction code."""
    parts = pd.DataFrame({"firma_key": pd.unique(pd.Series(participants_aa, dtype="string"))})
    q = queries_aa[["query_id", "plaka", "aa_kodu", "aa_tarihi"]]

    # grade-0 universe: every participant of THIS auction crossed with every car in it.
    if len(parts):
        part = q.merge(parts, how="cross")
    else:
        part = q.iloc[0:0].assign(firma_key=pd.Series([], dtype="string"))
    part["is_participant"] = True

    bids = bids_aa[["query_id", "firma_key", "is_win"]]
    part = part.merge(bids, on=["query_id", "firma_key"], how="left", indicator=True)
    part["is_bid"] = part["_merge"].eq("both")
    part["is_win"] = part["is_win"].eq(True)      # unmatched (NaN) -> False
    part = part.drop(columns="_merge")

    # bidders who did NOT participate in this auction: add them as 'bid' candidates.
    bidonly = bids.loc[~bids["firma_key"].isin(set(participants_aa))].copy()
    if len(bidonly):
        bidonly = bidonly.merge(q, on="query_id", how="left")
        bidonly["is_participant"] = False
        bidonly["is_bid"] = True
        bidonly["is_win"] = bidonly["is_win"].eq(True)

    block = pd.concat([part, bidonly], ignore_index=True)
    block["is_bid"] = block["is_bid"].eq(True)
    block["is_win"] = block["is_win"].eq(True)

    block["grade"] = assign_grade(block["is_bid"], block["is_win"])
    block["candidate_source"] = assign_source(block["is_participant"], block["is_bid"])
    block["split"] = add_split(block["aa_tarihi"])
    block["has_positive"] = mark_has_positive(block["query_id"], block["grade"])
    return block[OUTPUT_COLUMNS]


def block_assertion_counters(block: pd.DataFrame) -> dict:
    """Count any rule violations in one auction's block (all must stay zero)."""
    return {
        "dup_pairs": int(block.duplicated(["query_id", "firma_key"]).sum()),
        "bad_grade": int((~block["grade"].isin([0, 1, 2])).sum()),
        "residual_internal": int(block["firma_key"]
                                 .str.contains(r"UCAR|CARINA|KARINA", regex=True, na=False).sum()),
        "positive_not_bid": int(((block["grade"] >= 1) & (~block["is_bid"])).sum()),
        "win_not_grade2": int(((block["is_win"]) & (block["grade"] != 2)).sum()),
        "bad_source": int((~block["candidate_source"].isin(
            ["participation", "bid", "both"])).sum()),
    }

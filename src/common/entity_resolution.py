"""
entity_resolution.py  —  who is who.

Two jobs:
  1. INTERNAL (sister-company) firms — UÇAR / CARINA family. Their bids must be
     removed from the bid table (they are not arms-length buyers) and their
     purchases flagged on the sales side. Matching is lookup-driven and runs on
     the Turkish-folded ASCII key, with a hard residual check afterwards.
  2. satici_tipi — classify each SELLER as rent-car/fleet ('rac'), the big
     institutional fleets TEB ARVAL / VARLIK ('tebarvar'), or everything else
     ('diger'). Uses the Üye-Kurumsal "Detay Tip" ground-truth when available,
     otherwise a regex fallback on the firm name.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.normalize import ascii_key_series, ascii_key

# ----------------------------------------------------------------------------
# Internal-firm lookup.
#
# The CSV is the human-readable record of which raw names belong to which
# internal group. Matching itself uses each group's `match_pattern` (a regex on
# the ASCII key) — this is what guarantees the residual check can reach 0 even
# for spellings we have not seen yet. exclude_from_model=True means: drop their
# bids / flag their purchases so the model never learns from a related-party deal.
# ----------------------------------------------------------------------------
INTERNAL_LOOKUP_DEFAULT = [
    # group,  raw_value,                                  match_pattern, exclude_from_model
    ("UCAR",   "UÇAR OTOMOTİV ÜMRANİYE",                  "UCAR",        True),
    ("UCAR",   "UÇAR OTOMOTİV",                           "UCAR",        True),
    ("UCAR",   "UÇARLAR AUTO",                            "UCAR",        True),
    ("UCAR",   "MELİH EREN SÖNMEZ (UÇAR OTOMOTİV)",       "UCAR",        True),
    ("CARINA", "CARINA",                                  "CARINA|KARINA", True),
    ("CARINA", "KARİNA",                                  "CARINA|KARINA", True),
    ("CARINA", "KARINA",                                  "CARINA|KARINA", True),
]
INTERNAL_LOOKUP_COLUMNS = ["group", "raw_value", "lookup_key",
                           "match_pattern", "exclude_from_model"]


def ensure_internal_lookup(path: Path) -> pd.DataFrame:
    """
    Load Datasets/lookups/internal_firm_lookup.csv, creating it from the
    mandatory defaults if it does not exist. Returns the lookup DataFrame
    (with the ASCII lookup_key filled in).
    """
    if path.exists():
        df = pd.read_csv(path)
    else:
        rows = [{"group": g, "raw_value": rv, "match_pattern": mp,
                 "exclude_from_model": ex}
                for (g, rv, mp, ex) in INTERNAL_LOOKUP_DEFAULT]
        df = pd.DataFrame(rows)
        df["lookup_key"] = ascii_key_series(df["raw_value"])
        df = df[INTERNAL_LOOKUP_COLUMNS]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    if "lookup_key" not in df.columns or df["lookup_key"].isna().any():
        df["lookup_key"] = ascii_key_series(df["raw_value"])
    return df


def internal_pattern(lookup: pd.DataFrame) -> str:
    """Combine every group's match_pattern into one regex for the ASCII key."""
    pats = sorted(set(str(p) for p in lookup["match_pattern"].dropna()))
    return "|".join(f"(?:{p})" for p in pats)


def flag_internal(name_series: pd.Series, lookup: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return (is_internal flag 0/1, group label) for a firm-name column.
    A name is internal if its ASCII key matches any group's pattern.
    """
    keys = ascii_key_series(name_series)
    group = pd.Series(pd.NA, index=name_series.index, dtype="string")
    is_internal = pd.Series(0, index=name_series.index, dtype="int8")
    for grp, sub in lookup.groupby("group"):
        pat = "|".join(f"(?:{p})" for p in sorted(set(sub["match_pattern"].astype(str))))
        hit = keys.str.contains(pat, regex=True, na=False)
        is_internal = (is_internal | hit.astype("int8")).astype("int8")
        group = group.mask(hit, str(grp))
    return is_internal, group


def residual_internal_matches(name_series: pd.Series) -> pd.Series:
    """
    Hard residual check: ASCII keys that still look like UÇAR/CARINA/KARINA.
    Must be empty after internal bids are removed.
    """
    keys = ascii_key_series(name_series)
    mask = keys.str.contains(r"UCAR|CARINA|KARINA", regex=True, na=False)
    return name_series[mask]


# ----------------------------------------------------------------------------
# Seller-type classification (satici_tipi).
# ----------------------------------------------------------------------------
# Üye-Kurumsal "Detay Tip" -> our seller class. Filo Kiralama is the only
# rent-car/fleet category; the rest are ordinary dealers/consumers -> 'diger'.
# Keys are the *ASCII folded* form (no spaces), to match ascii_key_series output.
DETAY_TIP_MAP = {
    "FILOKIRALAMA": "rac",
    "GALERI": "diger",
    "YETKILISATICI": "diger",
    "OZELSIRKET": "diger",
    "TUKETICI": "diger",
}


def infer_satici_tipi(name_series: pd.Series) -> pd.Series:
    """
    Regex fallback when there is no Üye-Kurumsal ground truth.
    TEB ARVAL / VARLIK -> 'tebarvar' (big institutional fleets, kept separate),
    RAC/RENT/FİLO/LEASE/KİRALAMA -> 'rac', everything else -> 'diger'.
    Runs on the ASCII key so Turkish spellings (FİLO) are caught.
    """
    keys = ascii_key_series(name_series)
    out = pd.Series("diger", index=name_series.index, dtype="object")
    rac = keys.str.contains(r"RAC|RENT|FILO|LEASE|KIRALAMA", regex=True, na=False)
    out = out.mask(rac, "rac")
    tebarvar = keys.str.contains(r"TEBAR|VARLIK", regex=True, na=False)
    out = out.mask(tebarvar, "tebarvar")   # checked last so it wins over 'rac'
    return out


def build_satici_tipi(seller_series: pd.Series,
                      uye: pd.DataFrame | None) -> tuple[pd.Series, float]:
    """
    Classify each seller. Returns (satici_tipi series, ground_truth_match_rate).
    Ground truth from Üye-Kurumsal wins where the seller is found there; the
    regex fallback fills in everyone else.
    """
    seller_key = ascii_key_series(seller_series)
    result = infer_satici_tipi(seller_series)          # regex baseline for all
    match_rate = 0.0
    if uye is not None and "Firma Ünvanı" in uye.columns and "Detay Tip" in uye.columns:
        uye_key = ascii_key_series(uye["Firma Ünvanı"])
        uye_class = (ascii_key_series(uye["Detay Tip"]).map(DETAY_TIP_MAP)
                     .fillna("diger"))
        # last wins on duplicate keys; that is fine for a directory lookup
        gt = dict(zip(uye_key, uye_class))
        matched = seller_key.map(gt)
        hit = matched.notna()
        result = result.mask(hit, matched)             # ground truth wins (rac/diger)
        match_rate = float(hit.mean()) * 100.0
    # Preserve the explicitly-requested institutional class even over ground truth:
    # TEB ARVAL / VARLIK are isolated as 'tebarvar' (a refinement of fleet, not a
    # contradiction — they register as Filo Kiralama). Documented in the report.
    tebarvar = seller_key.str.contains(r"TEBAR|VARLIK", regex=True, na=False)
    result = result.mask(tebarvar, "tebarvar")
    return result.astype("object"), match_rate

"""
clean_sales.py  —  build Datasets/interim/clean_sales.parquet (frozen schema).

One row per (plaka, aa_kodu, aa_tarihi) sale listing. Everything here is
derive-and-flag: we add cleaned columns, engineered position/age features and
anomaly flags, but we never drop a sale (test-code rows are dropped upstream in
run_phase1). Column names and order are the frozen Section-5 contract.

V2 SCHEMA DEVIATIONS FROM V1 (approved by Serdar, 3 changes, 48 -> 49 columns):
  * REMOVED has_airbag_issue — no raw airbag signal exists, V1 value was the
    constant 0 on all 60,899 rows, and nothing downstream consumed it.
  * ADDED model_uzantisi (raw "Model Uzantısı") — pre-auction vehicle attribute,
    97.4% populated, far more granular than model (3,098 vs 410 values).
  * ADDED kasa_tipi (raw "Kasa Tipi") — pre-auction vehicle attribute, 97.6%
    populated, only 16 categories, likely firm-specialization signal.
  Deliberately NOT added: "Şehir" (it is the BUYER's city = outcome-side leakage)
  and "KK Uygunluğu" (business meaning not yet established — deferred).

V2 BUSINESS-RULE DEVIATION FROM V1 (approved, schema unchanged at 49 columns):
  * is_sold is now derived from the PRICE: (satis_tutari > 0), the original
    May-2026 Auction Priority rule agreed with company managers — not from the
    raw "Satış Durumu" text. Raw "Satış Durumu" is now only an OPTIONAL audit
    signal (recorded as the sale_status_price_mismatch flag count) and the
    pipeline no longer requires the column to be present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.normalize import (ascii_key_series, parse_numeric_value,
                              standardize_plaka, add_flag)
from common.entity_resolution import flag_internal, build_satici_tipi
from common.anomaly_flags import km_flags, eks_puani_flag

# Valid auction-date window (from the data profile). Anything outside is flagged.
# AA_DATE_MAX is the DEFAULT end (first export, 2026-06-04); run_phase1 passes the
# end of the export actually being processed (date_window), so later exports keep
# their newer auctions instead of flagging them invalid (Serdar, 2026-10-03).
AA_DATE_MIN = pd.Timestamp("2025-01-03")
AA_DATE_MAX = pd.Timestamp("2026-06-04")
SPLIT_BOUNDARY = pd.Timestamp("2026-01-01")

# Damage severity weights for hasar_compound (Phase-1 heuristic, documented in
# the report): local-paint < paint < replaced < chassis.
HASAR_WEIGHTS = {"boyali": 1.0, "lokal_boyali": 0.5, "degisen": 2.0, "sase": 3.0}

FROZEN_SALES_COLUMNS = [
    # key
    "plaka", "aa_kodu", "aa_tarihi",
    # vehicle
    "marka", "model", "model_uzantisi", "model_yili", "car_age", "km_clean",
    "km_per_year", "eks_puani_clean", "hasar_compound", "tramer_adedi",
    "tramer_tutari", "tramer_per_age", "min_fiyat_clean", "min_fiyat_was_zero",
    "tramer_to_min_fiyat_ratio", "satici_tipi", "yakit_tipi", "vites_tipi",
    "kasa_tipi", "renk", "lokasyon",
    # position
    "sira_no", "relative_position", "hat_size", "is_early_in_line",
    "aa_girisi_sira", "gunler_onceki_denemeden", "aa_day_of_week", "aa_month",
    # model B target / eligibility
    "satis_tutari", "net_satis", "tamamlama", "is_sold", "model_b_eligible",
    # join / label support
    "alici_firma_kisa_unvani", "alici_id", "internal_buyer_flag",
    "off_auction_sale_flag",
    # ledger
    "split", "dataset_version_id",
    # anomaly flags
    "km_nonpositive_flag", "km_extreme_flag", "sold_price_anomaly_flag",
    "eks_puani_out_of_range_flag", "plaka_format_valid", "aa_tarihi_invalid_flag",
]


def parse_ekspertiz(series: pd.Series) -> pd.DataFrame:
    """
    Parse the structured expertise string 'B:n LB:n D:n S:n' into its four
    damage counts plus a single weighted hasar_compound score.
    B = Boyalı (painted), LB = Lokal Boyalı, D = Değişmiş (replaced), S = Şase (chassis).
    """
    s = series.astype("string").fillna("")
    out = pd.DataFrame(index=series.index)
    out["boyali"] = pd.to_numeric(s.str.extract(r"B:(\d+)")[0], errors="coerce").fillna(0)
    out["lokal_boyali"] = pd.to_numeric(s.str.extract(r"LB:(\d+)")[0], errors="coerce").fillna(0)
    out["degisen"] = pd.to_numeric(s.str.extract(r"D:(\d+)")[0], errors="coerce").fillna(0)
    out["sase"] = pd.to_numeric(s.str.extract(r"S:(\d+)")[0], errors="coerce").fillna(0)
    out["hasar_compound"] = (
        out["boyali"] * HASAR_WEIGHTS["boyali"]
        + out["lokal_boyali"] * HASAR_WEIGHTS["lokal_boyali"]
        + out["degisen"] * HASAR_WEIGHTS["degisen"]
        + out["sase"] * HASAR_WEIGHTS["sase"]
    )
    return out


def _clean_text(series: pd.Series) -> pd.Series:
    """Trim and collapse internal whitespace; keep Turkish letters for display."""
    return (series.astype("string").str.strip()
            .str.replace(r"\s+", " ", regex=True))


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise divide; result is NaN wherever the denominator is <= 0/NaN."""
    den = den.where(den > 0)
    return num / den


def build_clean_sales(satis: pd.DataFrame, uye: pd.DataFrame | None,
                      internal_lookup: pd.DataFrame, version_id: str,
                      flags: list[dict], log: list[dict], *,
                      date_window: tuple[pd.Timestamp, pd.Timestamp] = (AA_DATE_MIN, AA_DATE_MAX),
                      ) -> tuple[pd.DataFrame, dict]:
    """
    Build the frozen clean_sales table from the (already test-code-dropped) raw
    Detay Satış frame. Returns (clean_sales, stats). date_window = (first, last) valid
    auction date; rows outside get aa_tarihi_invalid_flag = 1.
    """
    date_min, date_max = (pd.Timestamp(d) for d in date_window)
    if not date_min <= date_max:
        raise ValueError("[clean_sales] date_window must be (first, last) with first <= last")
    df = pd.DataFrame(index=satis.index)

    # ---- key ----
    plaka, plaka_valid = standardize_plaka(satis["Plaka"])
    df["plaka"] = plaka
    df["aa_kodu"] = satis["AA Kodu"].astype("string").str.strip()
    df["aa_tarihi"] = pd.to_datetime(satis["AA Tarihi"], errors="coerce")

    # ---- vehicle: identity & condition ----
    df["marka"] = _clean_text(satis["Marka"])
    df["model"] = _clean_text(satis["Model"])
    # V2 addition: trim/engine variant, materially finer-grained than model.
    df["model_uzantisi"] = _clean_text(satis["Model Uzantısı"])
    df["model_yili"] = pd.to_numeric(satis["Model Yılı"], errors="coerce")
    year = df["aa_tarihi"].dt.year
    df["car_age"] = (year - df["model_yili"]).clip(lower=0)
    df["car_age"] = df["car_age"].where(df["model_yili"] > 0)   # no model year -> NaN age

    df["km_clean"] = parse_numeric_value(satis["KM"])
    df["km_per_year"] = _safe_div(df["km_clean"], df["car_age"])

    df["eks_puani_clean"] = parse_numeric_value(satis["Eks.Puanı"])
    eks = parse_ekspertiz(satis["Ekspertiz"])
    df["hasar_compound"] = eks["hasar_compound"]

    df["tramer_adedi"] = pd.to_numeric(satis["Tramer Adedi"], errors="coerce")
    df["tramer_tutari"] = pd.to_numeric(satis["Tramer Tutarı"], errors="coerce")
    df["tramer_per_age"] = _safe_div(df["tramer_tutari"], df["car_age"])

    df["min_fiyat_clean"] = parse_numeric_value(satis["Min.Fiyat"])
    df["min_fiyat_was_zero"] = (df["min_fiyat_clean"] <= 0).fillna(True).astype("int8")
    df["tramer_to_min_fiyat_ratio"] = _safe_div(df["tramer_tutari"], df["min_fiyat_clean"])

    df["satici_tipi"], gt_rate = build_satici_tipi(satis["Satıcı Kısa Ünvan"], uye)
    df["yakit_tipi"] = _clean_text(satis["Yakıt Tipi"])
    df["vites_tipi"] = _clean_text(satis["Vites Tipi"])
    # V2 addition: body type (Sedan / Hatchback / SUV / Panelvan / ...), 16 values.
    df["kasa_tipi"] = _clean_text(satis["Kasa Tipi"])
    df["renk"] = _clean_text(satis["Renk"])
    df["lokasyon"] = _clean_text(satis["Lokasyon"])

    # ---- position in the auction line ----
    df["sira_no"] = pd.to_numeric(satis["Sıra"], errors="coerce")
    df["hat_size"] = df.groupby("aa_kodu")["aa_kodu"].transform("size")
    rank = df.groupby("aa_kodu")["sira_no"].rank(method="first")
    df["relative_position"] = (rank / df["hat_size"]).clip(upper=1.0)
    df["is_early_in_line"] = (df["relative_position"] <= 0.25).astype("int8")
    df["aa_girisi_sira"] = pd.to_numeric(satis["AA Girişi"], errors="coerce")

    # days since this plate's previous auction attempt (first attempt -> NaN)
    order = df.sort_values(["plaka", "aa_tarihi"])
    prev = order.groupby("plaka")["aa_tarihi"].shift(1)
    gunler = (order["aa_tarihi"] - prev).dt.days
    df["gunler_onceki_denemeden"] = gunler.reindex(df.index)

    df["aa_day_of_week"] = df["aa_tarihi"].dt.dayofweek
    df["aa_month"] = df["aa_tarihi"].dt.month

    # ---- outcome / model-B target ----
    df["satis_tutari"] = pd.to_numeric(satis["Satış Tutarı"], errors="coerce")
    tamamlama_raw = satis["AA Esnası Tamamlama"] if "AA Esnası Tamamlama" in satis.columns else 0
    df["tamamlama"] = pd.to_numeric(pd.Series(tamamlama_raw, index=satis.index),
                                    errors="coerce").fillna(0)
    df["net_satis"] = df["satis_tutari"] - df["tamamlama"]
    # V2 BUSINESS RULE (Serdar, original May-2026 Auction Priority definition,
    # agreed with company managers): a sale is real iff it carries a valid
    # positive sale price. On the current data this agrees with the raw
    # "Satış Durumu" text on 100.0000% of 60,899 rows, so no current value moves.
    df["is_sold"] = (df["satis_tutari"] > 0).fillna(False).astype("int8")

    # Optional consistency audit: the status text is no longer the source of
    # truth, but when the column is present we record how often it disagrees
    # with the price rule so any future divergence becomes visible. The audit
    # NEVER changes is_sold, and the column is not required to exist.
    if "Satış Durumu" in satis.columns:
        status_sold = ascii_key_series(satis["Satış Durumu"]).eq("SATILDI")
        add_flag(flags, "clean_sales", "sale_status_price_mismatch",
                 status_sold.ne(df["is_sold"] == 1),
                 "raw Satış Durumu disagrees with is_sold (satis_tutari > 0)")

    # ---- buyer / label support ----
    df["alici_firma_kisa_unvani"] = _clean_text(satis["Alıcı Firma Kısa Ünvanı"])
    df["alici_id"] = pd.to_numeric(satis["Alıcı Id"], errors="coerce")
    internal_buyer, _ = flag_internal(satis["Alıcı Firma Kısa Ünvanı"], internal_lookup)
    df["internal_buyer_flag"] = internal_buyer
    df["off_auction_sale_flag"] = ascii_key_series(satis["Satış Tipi"]).eq("EVET").astype("int8")

    # model_b_eligible: a real, arms-length, completed sale. Internal buyers are
    # excluded now (decided). The off-auction/rent-car question (KARAR 2) is
    # DEFERRED to Phase 2 — we only flag it here, we do not exclude on it.
    df["model_b_eligible"] = ((df["is_sold"] == 1) &
                              (df["internal_buyer_flag"] == 0)).astype("int8")

    # ---- ledger ----
    df["split"] = np.where(df["aa_tarihi"] < SPLIT_BOUNDARY, "train", "test")
    df["dataset_version_id"] = version_id

    # ---- anomaly flags ----
    df["km_nonpositive_flag"], df["km_extreme_flag"] = km_flags(df["km_clean"])
    df["sold_price_anomaly_flag"] = (
        (df["is_sold"] == 1) &
        ((df["satis_tutari"] <= 0) | (df["satis_tutari"] > 50_000_000))
    ).fillna(False).astype("int8")
    df["eks_puani_out_of_range_flag"] = eks_puani_flag(df["eks_puani_clean"])
    df["plaka_format_valid"] = plaka_valid
    df["aa_tarihi_invalid_flag"] = (
        df["aa_tarihi"].isna() |
        (df["aa_tarihi"] < date_min) | (df["aa_tarihi"] > date_max)
    ).astype("int8")

    # ---- record flags for cleaning_flags.csv ----
    for fl, desc in [
        ("km_nonpositive_flag", "KM <= 0 (sentinel / missing odometer)"),
        ("km_extreme_flag", "KM > 1,000,000 (implausible odometer)"),
        ("min_fiyat_was_zero", "Min.Fiyat <= 0 (reserve price not set)"),
        ("sold_price_anomaly_flag", "sold with non-positive / >50M sale price"),
        ("eks_puani_out_of_range_flag", "expertise score outside [0,10]"),
        ("aa_tarihi_invalid_flag", "auction date missing / outside data window"),
        ("internal_buyer_flag", "buyer is an internal sister company (UÇAR/CARINA)"),
        ("off_auction_sale_flag", "Satış Tipi = Evet (off-auction / direct deal)"),
    ]:
        add_flag(flags, "clean_sales", fl, df[fl] == 1, desc)
    add_flag(flags, "clean_sales", "plaka_format_invalid",
             df["plaka_format_valid"] == 0, "plate fails ^\\d{2}[A-Z]{1,3}\\d{2,4}$")

    # ---- freeze schema + dtypes ----
    out = _finalize(df)
    stats = {
        "n_rows": len(out),
        "n_sold": int((out["is_sold"] == 1).sum()),
        "n_model_b_eligible": int((out["model_b_eligible"] == 1).sum()),
        "n_internal_buyer": int((out["internal_buyer_flag"] == 1).sum()),
        "n_off_auction": int((out["off_auction_sale_flag"] == 1).sum()),
        "n_train": int((out["split"] == "train").sum()),
        "n_test": int((out["split"] == "test").sum()),
        "satici_tipi_counts": out["satici_tipi"].value_counts().to_dict(),
        "gt_match_rate": gt_rate,
    }
    return out, stats


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Select frozen columns in order and make object columns parquet-safe."""
    missing = [c for c in FROZEN_SALES_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"[clean_sales] missing frozen columns: {missing}")
    out = df[FROZEN_SALES_COLUMNS].copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out

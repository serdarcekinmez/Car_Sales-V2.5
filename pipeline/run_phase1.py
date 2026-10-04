"""
run_phase1.py  —  Phase 1 orchestrator (cleaning + entity resolution, TRY-only).

Pipeline (V2 order — assertions run BEFORE anything final is written):
  load 3 raw files  ->  drop TST/TAA test codes  ->  resolve internal firms +
  satici_tipi  ->  build clean_sales  ->  build clean_reel_db (winner flag from
  the sales join, internal bids removed)  ->  HARD ASSERTIONS  ->  only if they
  all pass: write the parquets, the reference lookups and the audit CSVs  ->
  stdout summary.

Scope is ONLY Phase 1: no USD, no Model-A label, no train/test files, no
candidate pool. Run from the repo root:  python3 pipeline/run_phase1.py

V2 DEVIATIONS FROM V1 (all approved by Serdar):
  * NO reports/phase1_report.md and NO dependency on src/reporting.py — the
    human-readable Phase-1 documentation is maintained separately. The report-only
    stats keys (crash_points, script_outputs, rowcount_rows, internal_names,
    flags) and the flag_internal re-computation that fed them are gone.
  * The `internal_firm_flag sum == 0` assertion is REMOVED (the column no longer
    exists in the 13-column bid schema). The independent, stronger
    residual_internal_matches() == 0 check remains.
  * No unconditional access to raw "Durumu" anywhere (it is an optional column).
  * The hard-coded `60909 - test_codes` row assertion is replaced by the real
    invariant len(clean_sales) == len(satis): clean_sales must not drop a row.
  * The hard-coded train/test size ranges are replaced by the split CONTRACT
    (values, totals, and both date directions), reusing clean_sales.SPLIT_BOUNDARY
    so there is only one split rule in the codebase.
  * ASSERT-BEFORE-WRITE: a file named clean_sales.parquet / clean_reel_db.parquet
    always represents an assertion-passing run. A failed run still leaves its
    audit trail (pipeline_log.csv / cleaning_flags.csv) but no "clean" parquet.

DYNAMIC EXPORTS (Serdar, 2026-10-03; Phase-6 contract §5.1 H1, §5.2):
  * The valid auction-date window ends at the export's last day, read from the
    Detay Satış filename prefix "YYMM-YYMMDD" (e.g. "2501-260604" -> 2026-06-04), or
    given explicitly with --date-max YYYY-MM-DD. The start stays AA_DATE_MIN.
  * Exactly one raw workbook per role must be present; several candidates stop the run.
  * Before new outputs replace existing ones, the existing Phase-1 outputs are copied
    to Datasets/snapshots/<their dataset_version_id>/ (never overwritten), so every
    model keeps the exact data version it was trained on.

Run:  python3 pipeline/run_phase1.py [--date-max YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1_preparation import io_load            # noqa: E402
from common.normalize import ascii_key_series, add_log  # noqa: E402
from common.anomaly_flags import is_test_code, drop_test_codes  # noqa: E402
from common.entity_resolution import (ensure_internal_lookup,  # noqa: E402
                                      residual_internal_matches)
from phase1_preparation.clean_sales import (build_clean_sales,  # noqa: E402
                                            FROZEN_SALES_COLUMNS, AA_DATE_MIN,
                                            AA_DATE_MAX, SPLIT_BOUNDARY)
from phase1_preparation.clean_bids import (build_clean_bids,  # noqa: E402
                                           build_sales_lookup, FROZEN_BIDS_COLUMNS)

RAW_DIR = ROOT / "Datasets" / "raw"
INTERIM = ROOT / "Datasets" / "interim" / "phase1_preparation"
LOOKUPS = ROOT / "Datasets" / "lookups"
SNAPSHOTS = ROOT / "Datasets" / "snapshots"
_EXPORT_RANGE = re.compile(r"^(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})\b")


def export_date_max(satis_path: Path, override: str | None = None) -> pd.Timestamp:
    """Last valid auction date: --date-max, else the 'YYMM-YYMMDD' filename prefix."""
    if override is not None:
        return pd.Timestamp(override).normalize()
    m = _EXPORT_RANGE.match(satis_path.name)
    if not m:
        raise SystemExit(f"[run_phase1] cannot read the export end date from "
                         f"'{satis_path.name}'; pass --date-max YYYY-MM-DD")
    yy, mm, dd = m.group(3), m.group(4), m.group(5)
    return pd.Timestamp(f"20{yy}-{mm}-{dd}")


def check_single_raw_files(raw_dir: Path) -> None:
    """Exactly one workbook per required role; ambiguity stops the run."""
    for role, keys in io_load.FILE_ROLES.items():
        hits = [p.name for p in sorted(raw_dir.glob("*.xlsx"))
                if any(k in p.name.lower() for k in keys)]
        if len(hits) > 1:
            raise SystemExit(f"[run_phase1] several '{role}' workbooks in {raw_dir}: {hits}; "
                             f"keep exactly one (archive the others outside the raw folder)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_existing_outputs(interim: Path, snapshots: Path, new_version: str) -> str | None:
    """
    Copy the current Phase-1 outputs to snapshots/<their version>/ before they are
    replaced. An existing snapshot is reused only if its files are byte-identical;
    otherwise the run stops. Returns the archived version id (None if nothing to keep).
    """
    current = interim / "clean_sales.parquet"
    if not current.exists():
        return None
    old_version = str(pd.read_parquet(current, columns=["dataset_version_id"])
                      ["dataset_version_id"].iloc[0])
    if old_version == new_version:
        return None
    dst = snapshots / old_version
    files = [f for f in sorted(interim.iterdir()) if f.is_file()]
    if dst.exists():
        for f in files:
            if (dst / f.name).exists() and _sha256(dst / f.name) != _sha256(f):
                raise SystemExit(f"[run_phase1] snapshot {dst} differs from the current "
                                 f"{f.name}; resolve by hand, nothing was overwritten")
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
    manifest = dst / "snapshot_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "dataset_version_id": old_version, "archived_by": "run_phase1",
            "files": {f.name: _sha256(dst / f.name) for f in files}}, indent=2) + "\n")
    return old_version


def make_version_id(paths: dict) -> str:
    """Stable id: build date + short hash of raw filenames & sizes."""
    h = hashlib.sha1()
    for role in ("satis", "teklif", "uye"):
        p = paths.get(role)
        if p is not None:
            h.update(p.name.encode("utf-8"))
            h.update(str(p.stat().st_size).encode("utf-8"))
    return f"phase1_{date.today().isoformat()}_{h.hexdigest()[:8]}"


def write_name_lookup(raw: pd.Series, path: Path) -> int:
    """Reference lookup: distinct raw brand/model -> cleaned value + ASCII key."""
    clean = raw.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df = pd.DataFrame({"raw_value": raw.astype("string"), "clean_value": clean})
    df = df.dropna(subset=["clean_value"])
    df["ascii_key"] = ascii_key_series(df["clean_value"])
    agg = (df.groupby(["clean_value", "ascii_key"], dropna=False)
           .size().reset_index(name="n_rows")
           .sort_values("n_rows", ascending=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(path, index=False)
    return len(agg)


def write_audit_files(flags: list[dict], log: list[dict]) -> None:
    """Write pipeline_log.csv + cleaning_flags.csv (also called when an assertion fails)."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(INTERIM / "pipeline_log.csv", index=False)
    if flags:
        pd.DataFrame(flags).to_csv(INTERIM / "cleaning_flags.csv", index=False)


def run(date_max: str | None = None) -> None:
    """Run Phase 1 end to end: load, clean, assert, and only then write the outputs."""
    for d in (INTERIM, LOOKUPS):
        d.mkdir(parents=True, exist_ok=True)
    check_single_raw_files(RAW_DIR)

    flags: list[dict] = []
    log: list[dict] = []

    # ---- load ----
    data = io_load.load_phase1_inputs(RAW_DIR)
    satis_raw, teklif_raw, uye = data["satis"], data["teklif"], data["uye"]
    paths = data["_paths"]
    version_id = make_version_id(paths)
    window = (AA_DATE_MIN, export_date_max(paths["satis"], date_max))
    add_log(log, "step0_load", "load",
            f"satis={len(satis_raw)}, teklif={len(teklif_raw)}, "
            f"uye={'absent' if uye is None else len(uye)}")

    # ---- Step 4: drop test codes (the only deletion made by THIS file) ----
    n_satis_test = int(is_test_code(satis_raw["AA Kodu"]).sum())
    n_teklif_test = int(is_test_code(teklif_raw["AA Kodu"]).sum())
    satis = drop_test_codes(satis_raw, "AA Kodu", "Detay Satış", log)
    teklif = drop_test_codes(teklif_raw, "AA Kodu", "Detay Teklif", log)

    # ---- Step 5 setup: internal-firm lookup (created from the mandatory
    # defaults on the first real run if the CSV does not exist yet) ----
    internal_lookup = ensure_internal_lookup(LOOKUPS / "internal_firm_lookup.csv")

    # ---- Step 7a: clean_sales ----
    clean_sales, sales_stats = build_clean_sales(
        satis, uye, internal_lookup, version_id, flags, log, date_window=window)
    add_log(log, "step7_sales", "build",
            f"clean_sales rows={len(clean_sales)} cols={clean_sales.shape[1]}",
            rows_in=len(satis), rows_out=len(clean_sales))

    # ---- Step 7b: clean_reel_db (uses sales lookup for the winner flag) ----
    sales_lookup = build_sales_lookup(clean_sales)
    clean_bids, removed_internal, bids_stats = build_clean_bids(
        teklif, sales_lookup, internal_lookup, version_id, flags, log)
    add_log(log, "step7_bids", "build",
            f"clean_reel_db rows={len(clean_bids)} cols={clean_bids.shape[1]}",
            rows_in=len(teklif), rows_out=len(clean_bids))

    # ---- Section 6 assertions — BEFORE any final output is written ----
    residual_series = residual_internal_matches(clean_bids["firma_adi"])
    residual_internal = int(len(residual_series))
    assertions = run_assertions(clean_sales, clean_bids, satis,
                                residual_internal, flags, log, window=window)
    archived = archive_existing_outputs(INTERIM, SNAPSHOTS, version_id)
    add_log(log, "archive", "snapshot",
            f"previous outputs kept in snapshots/{archived}" if archived else "nothing to archive")

    # ---- everything below runs ONLY because every assertion passed ----
    clean_sales.to_parquet(INTERIM / "clean_sales.parquet", index=False)
    clean_bids.to_parquet(INTERIM / "clean_reel_db.parquet", index=False)
    removed_internal.to_parquet(INTERIM / "removed_internal_bids.parquet", index=False)
    add_log(log, "write", "parquet",
            "clean_sales.parquet, clean_reel_db.parquet, removed_internal_bids.parquet")

    n_brand = write_name_lookup(satis["Marka"], LOOKUPS / "brand_lookup.csv")
    n_model = write_name_lookup(satis["Model"], LOOKUPS / "model_lookup.csv")
    add_log(log, "write", "lookups",
            f"brand_lookup.csv ({n_brand}), model_lookup.csv ({n_model}), "
            "internal_firm_lookup.csv")

    write_audit_files(flags, log)

    stats = {
        "version_id": version_id,
        "date_window": [str(window[0].date()), str(window[1].date())],
        "dropped": {"satis_test": n_satis_test, "teklif_test": n_teklif_test},
        "sales": {**sales_stats},
        "bids": {**bids_stats},
        "residual_internal": residual_internal,
        "assertions": assertions,
    }
    print_summary(stats, clean_sales, clean_bids)


def run_assertions(clean_sales, clean_bids, satis, residual_internal, flags, log, *,
                   window=(AA_DATE_MIN, AA_DATE_MAX)):
    """Section-6 hard checks. Collect (name, passed); raise on the first failure."""
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        add_log(log, "assert", "PASS" if cond else "FAIL", name)
        if not cond:
            write_audit_files(flags, log)   # keep the trail of the failed run
            raise SystemExit(f"[assert] FAILED: {name}")

    # 1. frozen columns present
    check("clean_sales has all frozen columns",
          all(c in clean_sales.columns for c in FROZEN_SALES_COLUMNS))
    check("clean_reel_db has all frozen columns",
          all(c in clean_bids.columns for c in FROZEN_BIDS_COLUMNS))
    # 2. internal firms fully removed (independent re-check on the NAMES)
    check("residual UÇAR/CARINA matches == 0", residual_internal == 0)
    # 3. no test codes in either table
    check("no TST/TAA codes in clean_sales",
          int(is_test_code(clean_sales["aa_kodu"]).sum()) == 0)
    check("no TST/TAA codes in clean_reel_db",
          int(is_test_code(clean_bids["aa_kodu"]).sum()) == 0)
    # 4. query_id == plaka::aa_kodu, no blank/NaN
    qid = clean_bids["plaka"].astype("string") + "::" + clean_bids["aa_kodu"].astype("string")
    check("query_id == plaka::aa_kodu", clean_bids["query_id"].astype("string").equals(qid))
    check("no blank/NaN query_id",
          int(clean_bids["query_id"].isna().sum()) == 0 and
          int(clean_bids["query_id"].astype("string").str.strip().eq("").sum()) == 0)
    # 5. aa_tarihi within window
    s_dates = clean_sales["aa_tarihi"].dropna()
    check(f"clean_sales aa_tarihi within [{window[0].date()}, {window[1].date()}]",
          s_dates.min() >= window[0] and s_dates.max() <= window[1])
    # 6. clean_sales must not silently delete a row (derive-and-flag contract)
    check("clean_sales rows == rows entering build_clean_sales (no row loss)",
          len(clean_sales) == len(satis))
    # 7. the split CONTRACT (data-independent; reuses clean_sales.SPLIT_BOUNDARY)
    check("split ∈ {train, test} (sales)",
          set(clean_sales["split"].unique()) <= {"train", "test"})
    n_train = int((clean_sales["split"] == "train").sum())
    n_test = int((clean_sales["split"] == "test").sum())
    check("n_train + n_test == len(clean_sales)",
          n_train + n_test == len(clean_sales))
    train_dates = clean_sales.loc[clean_sales["split"] == "train", "aa_tarihi"].dropna()
    test_dates = clean_sales.loc[clean_sales["split"] == "test", "aa_tarihi"].dropna()
    check("every dated train row is before the split boundary",
          bool((train_dates < SPLIT_BOUNDARY).all()))
    check("every dated test row is on/after the split boundary",
          bool((test_dates >= SPLIT_BOUNDARY).all()))
    return checks


def print_summary(stats, clean_sales, clean_bids):
    """Print the Phase-1 console summary (counts are information, not assertions)."""
    line = "=" * 66
    out = [
        line, "PHASE 1 COMPLETE — cleaning + entity resolution (TRY-only)", line,
        f"version_id        : {stats['version_id']}",
        f"clean_sales       : {len(clean_sales):,} rows × {clean_sales.shape[1]} cols  "
        f"(train {stats['sales']['n_train']:,} / test {stats['sales']['n_test']:,})",
        f"clean_reel_db     : {len(clean_bids):,} rows × {clean_bids.shape[1]} cols  "
        f"(train {stats['bids']['n_train']:,} / test {stats['bids']['n_test']:,})",
        f"test codes dropped: satış {stats['dropped']['satis_test']} / "
        f"teklif {stats['dropped']['teklif_test']}",
        f"internal bids out : {stats['bids']['n_removed_internal']:,}  "
        f"(residual UÇAR/CARINA = {stats['residual_internal']})",
        f"internal buyers   : {stats['sales']['n_internal_buyer']:,} flagged",
        f"satici_tipi       : {stats['sales']['satici_tipi_counts']}  "
        f"(Üye-Kurumsal match {stats['sales']['gt_match_rate']:.1f}%)",
        f"off_auction sales : {stats['sales']['n_off_auction']:,} flagged (KARAR 2 deferred)",
        f"winning bids      : {stats['bids']['n_winning']:,}  (from sales join)",
        f"model_b_eligible  : {stats['sales']['n_model_b_eligible']:,}",
        f"assertions        : {sum(1 for _, ok in stats['assertions'] if ok)}/"
        f"{len(stats['assertions'])} passed  (all ran BEFORE the parquets were written)",
        "outputs           : Datasets/interim/phase1_preparation/{clean_sales,"
        "clean_reel_db,removed_internal_bids}.parquet",
        "                    Datasets/interim/phase1_preparation/{pipeline_log,"
        "cleaning_flags}.csv  +  Datasets/lookups/*.csv",
        line,
    ]
    print("\n".join(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: clean the raw exports")
    parser.add_argument("--date-max", help="last valid auction date (default: from the "
                                           "Detay Satış filename prefix YYMM-YYMMDD)")
    run(parser.parse_args().date_max)

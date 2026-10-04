"""
run_phase3_firm_registry.py  —  run and publish the firm-registry step.

This runner does ONE thing: it builds the source-observed firm registry with the
existing phase3_firm_universe.firm_registry functions, checks the result, and — only
if every check passes — publishes two artifacts:

  Datasets/interim/phase3_firm_universe/firm_registry.parquet
  reports/phase3_firm_universe/firm_registry_run_summary.json

It is NOT a general Phase-3 orchestrator. Later Phase-3 steps get their own runners.

WHAT THE REGISTRY IS: a list of firm keys OBSERVED in our sources. It is not a list of
every firm in the business, it does not establish permission to contact anyone, and it
describes no point in time. Those limitations are written into the summary itself so a
reader of the artifact cannot miss them.

PUBLICATION IS NOT A TRANSACTION. Both artifacts are prepared in full under temporary
names first, then replaced one at a time. Each replacement is atomic for its own file;
two replacements are two events. A downstream reader must therefore check that the
summary says SUCCESS and that its run_id matches the parquet's run_id metadata.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.normalize import ascii_key_series                  # noqa: E402
from phase3_firm_universe import firm_registry as FR            # noqa: E402
import phase2_labels.participation as P                         # noqa: E402

RAW_DIR = ROOT / "Datasets" / "raw"
PHASE1_DIR = ROOT / "Datasets" / "interim" / "phase1_preparation"
LOOKUPS = ROOT / "Datasets" / "lookups"
DEFAULT_INTERIM = ROOT / "Datasets" / "interim" / "phase3_firm_universe"
DEFAULT_REPORTS = ROOT / "reports" / "phase3_firm_universe"

# Column types the registry contract guarantees (checked, not assumed).
EXPECTED_DTYPES = {
    "firma_key": "string", "firma_adi_display": "string",
    "in_directory": "bool", "observed_as_bidder": "bool",
    "observed_as_participant": "bool", "observed_as_buyer": "bool",
    "is_internal": "bool",
    "directory_firm_ids": "string", "n_directory_firm_ids": "int16",
    "participation_firm_ids": "string", "n_participation_ids": "int16",
    "has_id_collision": "bool",
    "detay_tip": "string", "tip": "string", "profil": "string",
    "sehir": "string", "ilce": "string",
    "firma_olusturma_tarihi": "datetime64[ns]",
}

LIMITATIONS = [
    "The member-directory snapshot label is a source label; it is NOT verified historical "
    "attribute validity. The directory carries no attribute history.",
    "firma_olusturma_tarihi is a source RECORD creation date, not the date the company was founded.",
    "observed_as_bidder / observed_as_participant / observed_as_buyer are WHOLE-HISTORY flags "
    "computed within each source AFTER that source's own filters. False means 'not observed in "
    "that input', never 'this firm never did this'. They are not pre-auction features.",
    "Registry membership records that a firm was observed in a source. It does NOT establish "
    "eligibility, administrative permission to participate, or permission to be contacted.",
    "Multiple source ids under one firm key record IDENTITY AMBIGUITY within a single source. "
    "They do not resolve legal identity and do not prove separate real firms were merged.",
    "Approved source coverage is the member directory, bids, participation and completed-sale "
    "buyers. A seller-only source is deliberately excluded, so the registry does not claim to "
    "cover every firm in the business.",
]


def _new_run_id() -> str:
    """Make a unique id identifying this attempt in temp names, metadata and reports."""
    return f"{datetime.now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def _require_file(path: Path, what: str) -> Path:
    """Fail clearly and early when a required input is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"[run_phase3_firm_registry] required {what} not found: {path}")
    return path


def _jsonable(obj):
    """Make numpy/pandas scalars safe for json.dump."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT or obj is pd.NA:
        return None
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _records(frame: pd.DataFrame) -> list[dict]:
    """Turn an audit table into JSON records: every row, missing values as null."""
    # pandas' own JSON writer emits standards-compliant null (never NaN) and ISO dates.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON document, refusing to emit non-standard NaN/Infinity values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False, default=_jsonable), encoding="utf-8")


def load_inputs(raw_dir: Path, phase1_dir: Path, lookups_dir: Path) -> dict:
    """Read every registry input and record what was actually consumed."""
    bids_path = _require_file(phase1_dir / "clean_reel_db.parquet", "Phase-1 clean_reel_db")
    sales_path = _require_file(phase1_dir / "clean_sales.parquet", "Phase-1 clean_sales")
    # Phase 3 READS the shared lookup; it must never create or modify it.
    lookup_path = _require_file(lookups_dir / "internal_firm_lookup.csv", "internal-firm lookup")

    clean_bids = pd.read_parquet(bids_path)
    clean_sales = pd.read_parquet(sales_path)
    internal_lookup = pd.read_csv(lookup_path)

    directory = FR.load_member_directory(raw_dir)
    aa_path = P.find_aa_katilim(raw_dir)
    aa_raw = P.load_aa_katilim(aa_path)
    clean_participation, part_stats = P.clean_participation(aa_raw, internal_lookup)

    dir_path = next(p for p in sorted(Path(raw_dir).glob("*.xlsx"))
                    if "üye" in p.name.lower() or "uye" in p.name.lower()
                    or "kurumsal" in p.name.lower())

    records = [
        {"role": "member_directory", "path": str(dir_path), "rows": len(directory),
         "required_columns": FR.DIRECTORY_REQUIRED},
        # load_aa_katilim asserts the auction DATE column too, even though the registry
        # itself never reads the date — record the real dependency, not a convenient one.
        {"role": "aa_katilim", "path": str(aa_path), "rows": len(aa_raw),
         "required_columns": [P.AA_KODU_RAW, P.AA_TARIHI_RAW, P.FIRMA_RAW],
         "optional_columns": [FR.AA_FIRMA_ID]},
        {"role": "clean_reel_db", "path": str(bids_path), "rows": len(clean_bids),
         "required_columns": ["firma_adi"]},
        {"role": "clean_sales", "path": str(sales_path), "rows": len(clean_sales),
         "required_columns": ["alici_firma_kisa_unvani", "is_sold"]},
        {"role": "internal_firm_lookup", "path": str(lookup_path), "rows": len(internal_lookup),
         "required_columns": ["group", "match_pattern"]},
    ]
    return {"directory": directory, "aa_raw": aa_raw,
            "clean_participation": clean_participation, "participation_stats": part_stats,
            "clean_bids": clean_bids, "clean_sales": clean_sales,
            "internal_lookup": internal_lookup, "input_records": records}


def build(inputs: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Build the registry with the existing functions and gather independent source evidence."""
    dir_attrs, dir_conflicts = FR.collapse_directory(inputs["directory"])
    key_ids, id_evidence_available = FR.participation_key_ids(
        inputs["aa_raw"], inputs["clean_participation"])
    observed = FR.observed_firm_names(inputs["clean_bids"], inputs["clean_sales"],
                                      inputs["clean_participation"])
    registry, audit = FR.build_firm_registry(dir_attrs, dir_conflicts, key_ids,
                                             id_evidence_available, observed,
                                             inputs["internal_lookup"])

    # Evidence computed straight from the INPUTS, so the checks do not merely re-read
    # numbers the registry itself produced.
    directory = inputs["directory"]
    dir_keys = {k for k in directory["firma_key"] if isinstance(k, str) and k}
    bidder_keys = {k for k in ascii_key_series(inputs["clean_bids"]["firma_adi"]) if k}
    sold = inputs["clean_sales"].loc[inputs["clean_sales"]["is_sold"] == 1]
    buyer_keys = {k for k in ascii_key_series(sold["alici_firma_kisa_unvani"]) if k}
    participant_keys = set(inputs["clean_participation"]["firma_key"])

    dir_ids = {k: {int(v) for v in g[FR.DIR_FIRMA_ID].dropna()}
               for k, g in directory.loc[directory["firma_key"].isin(dir_keys)]
               .groupby("firma_key")}
    part_ids: dict[str, set] = {}
    if id_evidence_available:
        keyed = pd.DataFrame({
            "aa_kodu": inputs["aa_raw"][P.AA_KODU_RAW].astype("string").str.strip(),
            "firma_key": ascii_key_series(inputs["aa_raw"][P.FIRMA_RAW]),
            "firma_id": inputs["aa_raw"][FR.AA_FIRMA_ID].values})
        survived = set(map(tuple, inputs["clean_participation"][["aa_kodu", "firma_key"]].values))
        kept = keyed.loc[[(a, f) in survived
                          for a, f in zip(keyed["aa_kodu"], keyed["firma_key"])]]
        part_ids = {k: {int(v) for v in g["firma_id"].dropna()}
                    for k, g in kept.groupby("firma_key")}

    evidence = {
        "directory_keys": dir_keys, "bidder_keys": bidder_keys,
        "participant_keys": participant_keys, "buyer_keys": buyer_keys,
        "union_keys": dir_keys | bidder_keys | participant_keys | buyer_keys,
        "directory_ids": dir_ids, "participation_ids": part_ids,
        "id_evidence_available": bool(id_evidence_available),
    }
    return registry, audit, evidence


def _parse_ids(value) -> set:
    """Read an 'id|id' cell back into a set of ids for comparison against source evidence."""
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value).strip()
    return {int(p) for p in text.split("|") if p} if text else set()


def check_registry(registry: pd.DataFrame, audit: dict, evidence: dict) -> list[dict]:
    """Run every durable check against the registry and independent source evidence."""
    conflicts = audit["directory_attribute_conflicts"]
    collisions = audit["id_collisions"]
    counts = audit["counts"]
    keys = set(registry["firma_key"])
    failures: list[str] = []

    # 1. exact ordered schema and compatible types
    schema_ok = list(registry.columns) == FR.REGISTRY_COLUMNS
    types_ok = schema_ok and all(str(registry[c].dtype) == t
                                 for c, t in EXPECTED_DTYPES.items())
    if not schema_ok:
        failures.append(f"column order differs from the contract: {list(registry.columns)}")
    elif not types_ok:
        bad = {c: str(registry[c].dtype) for c, t in EXPECTED_DTYPES.items()
               if str(registry[c].dtype) != t}
        failures.append(f"dtypes differ from the contract: {bad}")

    # 2. unique, non-blank key
    blank = int((registry["firma_key"].fillna("").str.len() == 0).sum())
    dup = int(registry["firma_key"].duplicated().sum())
    key_ok = blank == 0 and dup == 0
    if not key_ok:
        failures.append(f"firma_key problems: {blank} blank, {dup} duplicated")

    # 3. the key set equals the union computed from the SOURCES
    union_ok = keys == evidence["union_keys"]
    if not union_ok:
        failures.append(f"registry keys != source union "
                        f"(missing {sorted(evidence['union_keys'] - keys)[:5]}, "
                        f"extra {sorted(keys - evidence['union_keys'])[:5]})")

    # 4. source-specific id sets and counts agree with the input evidence
    dir_bad, part_bad = [], []
    for _, r in registry.iterrows():
        k = r["firma_key"]
        if _parse_ids(r["directory_firm_ids"]) != evidence["directory_ids"].get(k, set()) \
                or int(r["n_directory_firm_ids"]) != len(evidence["directory_ids"].get(k, set())):
            dir_bad.append(k)
        if evidence["id_evidence_available"]:
            if _parse_ids(r["participation_firm_ids"]) != evidence["participation_ids"].get(k, set()) \
                    or int(r["n_participation_ids"]) != len(evidence["participation_ids"].get(k, set())):
                part_bad.append(k)
        elif int(r["n_participation_ids"]) != 0 or _parse_ids(r["participation_firm_ids"]):
            part_bad.append(k)
    ids_ok = not dir_bad and not part_bad
    if not ids_ok:
        failures.append(f"source id evidence mismatch (directory {dir_bad[:5]}, "
                        f"participation {part_bad[:5]})")

    # 5. audit tables: right columns even when empty, and keys that exist
    audit_shape_ok = (list(conflicts.columns) == FR.CONFLICT_COLUMNS
                      and list(collisions.columns) == FR.ID_COLLISION_COLUMNS)
    audit_keys_ok = (set(conflicts["firma_key"]) <= keys) and (set(collisions["firma_key"]) <= keys)
    if not audit_shape_ok:
        failures.append("audit tables do not carry the expected columns")
    if not audit_keys_ok:
        failures.append("audit tables reference firm keys that are not in the registry")

    # 6. conflicting descriptive fields are null, evidence kept; display name is the exception
    idx = registry.set_index("firma_key")
    null_bad, name_bad = [], []
    for field, grp in conflicts.groupby("field"):
        for k in grp["firma_key"].unique():
            if k not in idx.index:
                continue
            if field == "display_name":
                if pd.isna(idx.at[k, "firma_adi_display"]):
                    name_bad.append(k)          # the approved exception: still filled
            elif not pd.isna(idx.at[k, field]):
                null_bad.append((k, field))
    evidence_kept = bool(len(conflicts) == 0 or
                         (conflicts["value"].notna().all() and conflicts["firma_id"].notna().all()))
    conflict_ok = not null_bad and not name_bad and evidence_kept
    if not conflict_ok:
        failures.append(f"conflict handling wrong (not-nulled {null_bad[:5]}, "
                        f"display name lost {name_bad[:5]}, evidence complete={evidence_kept})")

    # 7. reported counts agree with the actual tables
    by_field = conflicts.groupby("field")["firma_key"].nunique().to_dict() if len(conflicts) else {}
    counts_ok = (counts["registry_keys"] == len(registry)
                 and counts["directory_keys"] == int(registry["in_directory"].sum())
                 and counts["activity_only_keys"] == int((~registry["in_directory"]).sum())
                 and counts["internal_keys"] == int(registry["is_internal"].sum())
                 and counts["keys_with_any_id_collision"] == int(registry["has_id_collision"].sum())
                 and counts["participation_id_evidence_available"] == evidence["id_evidence_available"]
                 and counts["directory_attribute_conflicts_by_field"] ==
                 {k: int(v) for k, v in sorted(by_field.items())})
    if not counts_ok:
        failures.append("reported counts disagree with the actual registry/audit tables")

    checks = [
        {"name": "exact ordered 18-column schema and types", "passed": bool(schema_ok and types_ok)},
        {"name": "firma_key unique and non-blank", "passed": bool(key_ok)},
        {"name": "registry keys equal the union of source keys", "passed": bool(union_ok)},
        {"name": "source-specific id sets/counts match input evidence", "passed": bool(ids_ok)},
        {"name": "audit tables well-formed and reference real keys",
         "passed": bool(audit_shape_ok and audit_keys_ok)},
        {"name": "conflicting fields nulled, evidence kept, display name filled",
         "passed": bool(conflict_ok)},
        {"name": "reported counts agree with the actual tables", "passed": bool(counts_ok)},
    ]
    for c in checks:
        print(f"[check] {'PASS' if c['passed'] else 'FAIL'}: {c['name']}")
    if failures:
        raise ValueError("[run_phase3_firm_registry] validation failed:\n  - "
                         + "\n  - ".join(failures))
    return checks


def publish(pairs: list, published: list) -> None:
    """Replace finals one at a time, appending to the caller's list so partial progress survives."""
    for name, tmp, final in pairs:
        os.replace(tmp, final)          # atomic per file, NOT across files
        published.append(name)


def _cleanup_temps(temps: list) -> None:
    """Remove only this run's temporary files; never touch a published final."""
    for p in temps:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def run(interim_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    """Build, check and publish the firm registry and its run summary."""
    run_id = _new_run_id()
    interim = Path(interim_dir) if interim_dir else DEFAULT_INTERIM
    reports = Path(reports_dir) if reports_dir else DEFAULT_REPORTS
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    final_registry = interim / "firm_registry.parquet"
    final_summary = reports / "firm_registry_run_summary.json"
    tmp_registry = interim / f".firm_registry.{run_id}.tmp.parquet"
    tmp_summary = reports / f".firm_registry_run_summary.{run_id}.tmp.json"
    temps = [tmp_registry, tmp_summary]

    stage = "starting"
    published: list = []
    counters: dict = {}
    try:
        stage = "reading_inputs"
        inputs = load_inputs(RAW_DIR, PHASE1_DIR, LOOKUPS)

        stage = "building_registry"
        registry, audit, evidence = build(inputs)
        counters = dict(audit["counts"])
        print(f"[registry] {len(registry):,} firm keys | "
              f"{counters['directory_keys']:,} in directory | "
              f"{counters['activity_only_keys']} activity-only")

        stage = "checking"
        checks = check_registry(registry, audit, evidence)

        stage = "preparing_outputs"
        table = pa.Table.from_pandas(registry, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               b"run_id": run_id.encode(),
                                               b"phase": b"phase3_firm_universe"})
        pq.write_table(table, tmp_registry)

        part_stats = {k: v for k, v in inputs["participation_stats"].items()
                      if k != "identity_audit"}
        payload = {
            "run_id": run_id, "status": "SUCCESS",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": inputs["input_records"],
            "outputs": [
                {"name": "firm_registry", "path": str(final_registry), "rows": len(registry),
                 "columns": len(registry.columns)},
                {"name": "firm_registry_run_summary", "path": str(final_summary), "rows": 1},
            ],
            "participation_filters": part_stats,
            "counts": counters,
            "checks": checks,
            # every audit record, in full: no truncation, value->source-id kept intact
            "directory_attribute_conflicts": _records(audit["directory_attribute_conflicts"]),
            "id_collisions": _records(audit["id_collisions"]),
            "audit_note": (
                "id_collisions has one row per (firm key, source) that carries more than one "
                "source id. Row count is NOT a count of distinct problematic firms, and a "
                "collision is not proof that separate real firms were merged."),
            "limitations": LIMITATIONS,
            "publication": {
                "order": ["firm_registry", "firm_registry_run_summary"],
                "note": ("Each replacement is atomic for its own file; the two together are NOT "
                         "a transaction and are not automatically rolled back. A reader must "
                         "check status == SUCCESS and that this run_id matches the parquet's "
                         "run_id metadata."),
            },
        }
        _write_json(tmp_summary, payload)

        stage = "publishing"
        publish([("firm_registry", tmp_registry, final_registry),
                 ("firm_registry_run_summary", tmp_summary, final_summary)], published)
        stage = "complete"
        _cleanup_temps(temps)
        print_summary(payload)
        return payload

    except BaseException as exc:                       # noqa: BLE001 - diagnose, then re-raise
        try:
            _write_json(reports / f"firm_registry_failed_{run_id}.json", {
                "run_id": run_id, "status": "FAILED", "stage": stage,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "exception": f"{type(exc).__name__}: {exc}",
                "counters": counters,
                "outputs_already_replaced": published,
                "publication_complete": False,
                "note": ("A failure before publication leaves every previous final untouched. "
                         "If an output was already replaced, publication is INCOMPLETE and this "
                         "run must not be treated as successful."),
            })
        except Exception:                              # noqa: BLE001
            pass                                       # never mask the original error
        try:
            _cleanup_temps(temps)                      # temps only; published finals are kept
        except Exception:                              # noqa: BLE001
            pass
        raise


def print_summary(payload: dict) -> None:
    """Print the firm-registry console summary."""
    c = payload["counts"]
    line = "=" * 70
    print("\n".join([
        line, "PHASE 3 — FIRM REGISTRY PUBLISHED", line,
        f"run_id              : {payload['run_id']}",
        f"registry keys       : {c['registry_keys']:,}  (source-observed firms, not a callable pool)",
        f"  in directory      : {c['directory_keys']:,}",
        f"  directory-only    : {c['directory_only_keys']:,}",
        f"  activity-only     : {c['activity_only_keys']:,}",
        f"  internal flagged  : {c['internal_keys']}",
        f"observed b/p/buyer  : {c['observed_as_bidder']:,} / "
        f"{c['observed_as_participant']:,} / {c['observed_as_buyer']:,}",
        f"attribute conflicts : {len(payload['directory_attribute_conflicts'])} rows "
        f"{c['directory_attribute_conflicts_by_field']}",
        f"id collision rows   : {len(payload['id_collisions'])} "
        f"(rows, not distinct problematic firms)",
        f"participation ids   : available={c['participation_id_evidence_available']}",
        f"checks              : {sum(1 for x in payload['checks'] if x['passed'])}/"
        f"{len(payload['checks'])} passed (all BEFORE publication)",
        f"outputs             : {payload['outputs'][0]['path']}",
        f"                      {payload['outputs'][1]['path']}",
        line,
    ]))


if __name__ == "__main__":
    run()

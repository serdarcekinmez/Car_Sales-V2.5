"""
run_phase2.py  —  Phase 2 orchestrator: the graded "who to call" training table.

Pipeline:
  read clean_sales + clean_reel_db (Phase 1) and the AA Katılım workbook  ->  clean
  the participation pool  ->  build the query table + its per-query audit  ->  collapse
  bids to (query, firm) pairs with gated win evidence  ->  for EVERY auction code
  separately, cross that auction's participants with its cars and fold in the bidders
  ->  check each block, stream it to a TEMPORARY parquet  ->  run the global checks  ->
  only then publish the outputs  ->  console summary.

Scope is ONLY Phase 2: NO features, NO eligibility, NO negative sampling, NO Model B,
NO training. The full candidate table is never held in memory — it is written one
auction at a time. Run from the repo root:  python3 pipeline/run_phase2.py

PUBLICATION IS NOT A TRANSACTION. Each output is replaced atomically on its own
(os.replace, same filesystem), but three replacements are three separate events. If
one fails midway we stop, report exactly which outputs were already replaced, and
mark the run as incompletely published — we do not pretend it rolled back.
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

import phase2_labels.participation as part          # noqa: E402
import phase2_labels.labels as lab                  # noqa: E402

RAW_DIR = ROOT / "Datasets" / "raw"
PHASE1_DIR = ROOT / "Datasets" / "interim" / "phase1_preparation"
LOOKUPS = ROOT / "Datasets" / "lookups"
DEFAULT_INTERIM = ROOT / "Datasets" / "interim" / "phase2_labels"
DEFAULT_REPORTS = ROOT / "reports" / "phase2_labels"

# Informational only — the row count produced by the earlier V1 pipeline on this
# export. It is compared and reported, never asserted as a production rule.
HISTORICAL_CANDIDATE_ROWS = 23_147_415

IDENTITY_NOTE = ("Counts firm keys that map to more than one source FirmaID. This "
                 "records where the name-derived key MERGED distinct source records; "
                 "it is not proof that separate real firms were wrongly combined, and "
                 "it makes no claim about vehicle or stock identity.")


def _new_run_id() -> str:
    """Make a unique id identifying this attempt in temp names, metadata and reports."""
    return f"{datetime.now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def _require_file(path: Path, what: str) -> Path:
    """Fail clearly and early when a required input is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"[run_phase2] required {what} not found: {path}")
    return path


def _jsonable(obj):
    """Make numpy/pandas scalars safe for json.dump."""
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON document, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable),
                    encoding="utf-8")


def _new_summary() -> dict:
    """Create the zeroed counters we grow one auction-block at a time."""
    return {"total_rows": 0, "n_queries_seen": 0, "n_queries_with_positive": 0,
            "grade": {0: 0, 1: 0, 2: 0},
            "source": {"participation": 0, "bid": 0, "both": 0},
            "split": {"train": 0, "test": 0},
            "violations": {"dup_pairs": 0, "bad_grade": 0, "residual_internal": 0,
                           "positive_not_bid": 0, "win_not_grade2": 0, "bad_source": 0}}


def _accumulate(summary: dict, block: pd.DataFrame, seen: set) -> None:
    """Fold one auction's block into the running totals and violation counters."""
    summary["total_rows"] += len(block)
    per_query = block.groupby("query_id")["has_positive"].first()
    summary["n_queries_seen"] += int(per_query.size)
    summary["n_queries_with_positive"] += int(per_query.sum())
    seen.update(per_query.index.tolist())
    for g, n in block["grade"].value_counts().items():
        summary["grade"][int(g)] = summary["grade"].get(int(g), 0) + int(n)
    for s, n in block["candidate_source"].value_counts().items():
        summary["source"][s] = summary["source"].get(s, 0) + int(n)
    for s, n in block["split"].value_counts().items():
        summary["split"][s] = summary["split"].get(s, 0) + int(n)
    for k, v in lab.block_assertion_counters(block).items():
        summary["violations"][k] = summary["violations"].get(k, 0) + int(v)


def _block_to_arrow(block: pd.DataFrame, aa_kodu: str, schema: pa.Schema) -> pa.Table:
    """Check one block's column contract and convert it safely to the Arrow schema."""
    if list(block.columns) != lab.OUTPUT_COLUMNS:
        raise ValueError(f"[run_phase2] auction {aa_kodu}: candidate columns/order differ "
                         f"from the contract\n  got: {list(block.columns)}")
    # pandas object/string dtypes are NOT Arrow types: validate the CONVERSION and the
    # resulting Arrow schema instead of comparing incompatible dtype representations.
    try:
        table = pa.Table.from_pandas(block, schema=schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise ValueError(f"[run_phase2] auction {aa_kodu}: block does not convert to the "
                         f"Arrow schema: {exc}") from exc
    if not table.schema.remove_metadata().equals(schema.remove_metadata()):
        raise ValueError(f"[run_phase2] auction {aa_kodu}: converted Arrow schema differs "
                         f"from arrow_schema()")
    violations = {k: v for k, v in lab.block_assertion_counters(block).items() if v}
    if violations:
        raise ValueError(f"[run_phase2] auction {aa_kodu}: block violates the label rules: "
                         f"{violations}")
    return table


def _expected_candidates(q_by_auction: dict, b_by_auction: dict,
                         participants: dict) -> tuple[int, set]:
    """Compute, from the inputs alone, how many rows and which zero-candidate queries to expect."""
    expected_rows = 0
    expected_zero = set()
    for aa_kodu, queries_aa in q_by_auction.items():
        keys = set(participants.get(aa_kodu, []))
        n_queries = int(queries_aa["query_id"].nunique())
        expected_rows += n_queries * len(keys)          # participants x cars
        bids_aa = b_by_auction.get(aa_kodu)
        bid_queries = set()
        if bids_aa is not None and len(bids_aa):
            # bidder pairs whose firm did NOT participate in THIS auction
            expected_rows += int((~bids_aa["firma_key"].isin(keys)).sum())
            bid_queries = set(bids_aa["query_id"])
        if not keys:
            # no participants: only queries that also have no bidder emit nothing
            expected_zero |= {q for q in queries_aa["query_id"] if q not in bid_queries}
    return expected_rows, expected_zero


def _check_global(summary: dict, query_table: pd.DataFrame, query_audit: pd.DataFrame,
                  bidder_pairs: pd.DataFrame, expected_rows: int,
                  expected_zero: set, seen: set) -> list:
    """Run the whole-run checks; raise on the first failure, before anything is published."""
    missing = set(query_table["query_id"]) - seen
    extra = seen - set(query_table["query_id"])
    checks = [
        ("no rule violations in any block", all(v == 0 for v in summary["violations"].values())),
        ("row count equals the expectation computed from inputs",
         summary["total_rows"] == expected_rows),
        ("positive candidates == bidder pairs",
         summary["grade"][1] + summary["grade"][2] == len(bidder_pairs)),
        ("winning candidates == bidder_pairs.is_win",
         summary["grade"][2] == int(bidder_pairs["is_win"].sum())),
        ("query_audit keys == query_table keys",
         set(query_audit["query_id"]) == set(query_table["query_id"])
         and len(query_audit) == len(query_table)),
        ("no unexpected query emitted", not extra),
        ("queries emitting no rows are exactly the expected zero-candidate set",
         missing == expected_zero),
    ]
    for name, ok in checks:
        print(f"[check] {'PASS' if ok else 'FAIL'}: {name}")
    for name, ok in checks:
        if not ok:
            detail = ""
            if name.startswith("queries emitting"):
                detail = (f"\n  unexpected missing: {sorted(missing - expected_zero)[:5]}"
                          f"\n  expected but present: {sorted(expected_zero - missing)[:5]}")
            if name.startswith("row count"):
                detail = f"\n  got {summary['total_rows']:,} expected {expected_rows:,}"
            raise ValueError(f"[run_phase2] global check failed: {name}{detail}")
    return [{"name": n, "passed": bool(ok)} for n, ok in checks]


def _publish(pairs: list, published: list) -> None:
    """Replace finals one at a time, appending to `published` so partial progress survives.

    `published` is supplied by the caller on purpose: if a replacement fails midway the
    caller still sees exactly which outputs were already replaced.
    """
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
    """Build and publish the Phase-2 candidate table, its query audit and the run summary."""
    run_id = _new_run_id()
    interim = Path(interim_dir) if interim_dir else DEFAULT_INTERIM
    reports = Path(reports_dir) if reports_dir else DEFAULT_REPORTS
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    final_cand = interim / "labeled_candidates.parquet"
    final_audit = interim / "query_audit.parquet"
    final_summary = reports / "phase2_run_summary.json"
    tmp_cand = interim / f".labeled_candidates.{run_id}.tmp.parquet"
    tmp_audit = interim / f".query_audit.{run_id}.tmp.parquet"
    tmp_summary = reports / f".phase2_run_summary.{run_id}.tmp.json"
    temps = [tmp_cand, tmp_audit, tmp_summary]

    stage = "starting"
    published: list = []
    summary = _new_summary()
    writer = None
    try:
        # ---- 1. inputs ----
        stage = "reading_inputs"
        sales_path = _require_file(PHASE1_DIR / "clean_sales.parquet", "Phase-1 clean_sales")
        bids_path = _require_file(PHASE1_DIR / "clean_reel_db.parquet", "Phase-1 clean_reel_db")
        # Phase 2 READS the shared lookup; it must never create or modify it.
        lookup_path = _require_file(LOOKUPS / "internal_firm_lookup.csv", "internal-firm lookup")
        clean_sales = pd.read_parquet(sales_path)
        clean_bids = pd.read_parquet(bids_path)
        internal_lookup = pd.read_csv(lookup_path)
        aa_path = part.find_aa_katilim(RAW_DIR)
        aa_raw = part.load_aa_katilim(aa_path)
        inputs = [{"role": "clean_sales", "path": str(sales_path), "rows": len(clean_sales)},
                  {"role": "clean_reel_db", "path": str(bids_path), "rows": len(clean_bids)},
                  {"role": "aa_katilim", "path": str(aa_path), "rows": len(aa_raw)},
                  {"role": "internal_firm_lookup", "path": str(lookup_path),
                   "rows": len(internal_lookup)}]

        # ---- 2. participation ----
        stage = "building_participation"
        participation, part_stats = part.clean_participation(aa_raw, internal_lookup)
        participants = part.participants_by_auction(participation)
        print(f"[participation] {part_stats['rows_in']:,} raw -> {part_stats['rows_out']:,} rows "
              f"({part_stats['n_auctions']} auctions, {part_stats['n_firms']:,} firms)")

        # ---- 3. queries, audit, bidder pairs ----
        stage = "building_queries"
        query_table, query_audit = lab.build_query_table(clean_sales, clean_bids)
        stage = "building_bidder_pairs"
        bidder_pairs = lab.build_bidder_pairs(clean_bids, query_table)
        print(f"[queries] {len(query_table):,} queries | "
              f"{int(query_audit['is_duplicate_listing'].sum())} duplicate listings | "
              f"{int(query_audit['is_orphan_query'].sum())} orphans")
        print(f"[bids] {len(bidder_pairs):,} pairs | "
              f"{int(bidder_pairs['is_win'].sum()):,} gated wins")

        q_by_auction = dict(tuple(query_table.groupby("aa_kodu")))
        b_by_auction = dict(tuple(bidder_pairs.groupby("aa_kodu")))
        empty_bids = bidder_pairs.iloc[0:0]
        expected_rows, expected_zero = _expected_candidates(q_by_auction, b_by_auction,
                                                            participants)
        print(f"[expect] {expected_rows:,} candidate rows computed from inputs "
              f"({len(expected_zero)} zero-candidate queries)")

        # ---- 4. stream one auction at a time to a TEMPORARY file ----
        stage = "streaming_blocks"
        schema = lab.arrow_schema().with_metadata({"run_id": run_id, "phase": "phase2_labels"})
        writer = pq.ParquetWriter(tmp_cand, schema)
        seen: set = set()
        for aa_kodu in sorted(q_by_auction):
            block = lab.candidate_block_for_auction(
                q_by_auction[aa_kodu], participants.get(aa_kodu, []),
                b_by_auction.get(aa_kodu, empty_bids))
            table = _block_to_arrow(block, aa_kodu, schema)
            _accumulate(summary, block, seen)
            writer.write_table(table)
        writer.close()
        writer = None

        # ---- 5. global checks BEFORE anything is published ----
        stage = "global_checks"
        checks = _check_global(summary, query_table, query_audit, bidder_pairs,
                               expected_rows, expected_zero, seen)
        zero_queries = sorted(set(query_table["query_id"]) - seen)

        # ---- 6. prepare the remaining outputs under temporary names ----
        stage = "preparing_outputs"
        audit_tbl = pa.Table.from_pandas(query_audit, preserve_index=False)
        audit_tbl = audit_tbl.replace_schema_metadata(
            {**(audit_tbl.schema.metadata or {}), b"run_id": run_id.encode(),
             b"phase": b"phase2_labels"})
        pq.write_table(audit_tbl, tmp_audit)

        payload = {
            "run_id": run_id, "status": "SUCCESS",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": inputs,
            "outputs": [{"name": "labeled_candidates", "path": str(final_cand),
                         "rows": summary["total_rows"]},
                        {"name": "query_audit", "path": str(final_audit),
                         "rows": len(query_audit)},
                        {"name": "run_summary", "path": str(final_summary), "rows": 1}],
            "participation": {k: v for k, v in part_stats.items() if k != "identity_audit"},
            "identity_audit": {**part_stats["identity_audit"], "interpretation": IDENTITY_NOTE},
            "queries": {"total": len(query_table),
                        "duplicate_listings": int(query_audit["is_duplicate_listing"].sum()),
                        "orphans": int(query_audit["is_orphan_query"].sum()),
                        "sale_type_state": query_table["sale_type_state"]
                                           .value_counts().to_dict(),
                        "zero_candidate_query_ids": zero_queries},
            "bidder_pairs": {"total": len(bidder_pairs),
                             "winning": int(bidder_pairs["is_win"].sum())},
            "candidates": {"total_rows": summary["total_rows"],
                           "expected_rows_from_inputs": expected_rows,
                           "historical_reference_rows": HISTORICAL_CANDIDATE_ROWS,
                           "matches_historical": summary["total_rows"] == HISTORICAL_CANDIDATE_ROWS,
                           "grade": summary["grade"], "source": summary["source"],
                           "split": summary["split"],
                           "queries_seen": summary["n_queries_seen"],
                           "queries_with_positive": summary["n_queries_with_positive"]},
            "checks": checks,
            "publication": {"order": ["labeled_candidates", "query_audit", "run_summary"],
                            "note": "each replacement is atomic per file; the three together "
                                    "are NOT a transaction. The summary is the completion marker."},
        }
        _write_json(tmp_summary, payload)

        # ---- 7. publish: candidates, then audit, then the summary LAST ----
        stage = "publishing"
        _publish([("labeled_candidates", tmp_cand, final_cand),
                  ("query_audit", tmp_audit, final_audit),
                  ("run_summary", tmp_summary, final_summary)], published)
        stage = "complete"
        _cleanup_temps(temps)
        print_summary(payload)
        return payload

    except BaseException as exc:                      # noqa: BLE001 - diagnose, then re-raise
        try:
            if writer is not None:
                writer.close()
        except Exception:                             # noqa: BLE001
            pass
        try:
            _write_json(reports / f"phase2_failed_{run_id}.json", {
                "run_id": run_id, "status": "FAILED", "stage": stage,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "exception": f"{type(exc).__name__}: {exc}",
                "counters": summary,
                "outputs_already_replaced": published,
                "publication_complete": stage == "complete",
                "note": ("Pre-publication failures leave every previous final untouched. "
                         "If some outputs were already replaced, publication is INCOMPLETE "
                         "and this run must not be treated as successful."),
            })
        except Exception:                             # noqa: BLE001
            pass                                      # never mask the original error
        try:
            _cleanup_temps(temps)                     # temps only; published finals are kept
        except Exception:                             # noqa: BLE001
            pass
        raise


def print_summary(payload: dict) -> None:
    """Print the Phase-2 console summary of the labeled training table."""
    c, q, b = payload["candidates"], payload["queries"], payload["bidder_pairs"]
    g, s, sp = c["grade"], c["source"], c["split"]
    line = "=" * 70
    pos = c["queries_with_positive"]
    print("\n".join([
        line, "PHASE 2 COMPLETE — graded 'who to call' training table", line,
        f"run_id            : {payload['run_id']}",
        f"total rows        : {c['total_rows']:,}  (one per query_id x firma_key)",
        f"  expected        : {c['expected_rows_from_inputs']:,} (computed from inputs)",
        f"  historical ref   : {c['historical_reference_rows']:,} "
        f"(match: {c['matches_historical']})",
        f"queries (cars)    : {q['total']:,}  "
        f"[{q['duplicate_listings']} duplicate listings, {q['orphans']} orphans]",
        f"  with positives  : {pos:,} ({100*pos/max(c['queries_seen'],1):.1f}%)",
        f"  zero-candidate  : {len(q['zero_candidate_query_ids'])}",
        f"grade 2 / 1 / 0   : {g[2]:,} / {g[1]:,} / {g[0]:,}",
        f"source both/bid/participation : {s['both']:,} / {s['bid']:,} / {s['participation']:,}",
        f"split train/test  : {sp['train']:,} / {sp['test']:,}",
        f"bidder pairs      : {b['total']:,} ({b['winning']:,} winning)",
        f"checks            : {sum(1 for x in payload['checks'] if x['passed'])}/"
        f"{len(payload['checks'])} passed (all BEFORE publication)",
        f"outputs           : {payload['outputs'][0]['path']}",
        f"                    {payload['outputs'][1]['path']}",
        f"                    {payload['outputs'][2]['path']}",
        line,
    ]))


if __name__ == "__main__":
    run()

"""
firm_registry.py  —  the source-observed firm registry.

One row per firma_key that EXISTS in at least one of our sources: the Üye-Kurumsal
member directory, the cleaned bid table, the cleaned participation table, or the
completed-sale buyers. It answers the plain question "which firms do we know about,
and what does each source say about them?"

WHAT THIS IS NOT — read this before using any column:
  * NOT a historically valid firm universe. The member directory is a SNAPSHOT
    (its filename carries a source label, which is not verified historical validity).
    It has no valid_from/valid_to and no attribute history, so it describes the firms
    as of the export, not as of any past auction date.
  * NOT a callable / eligible firm pool. No eligibility decision is made here.
  * NOT a point-in-time feature table. The observed_as_* flags are WHOLE-HISTORY and
    therefore reveal activity that would lie in the future of any given auction.
    No column here is approved for point-in-time features.
  * NOT a resolution of legal-entity identity. Where one folded key carries several
    source ids, or several records disagree, we RECORD the evidence; we never decide
    which record is the true firm.

Internal firms (UÇAR / CARINA family) are RETAINED and flagged, because a dimension
should be complete. The pools that exclude them are elsewhere.

observed_as_* semantics — important: False means "not observed in this input AFTER
its own filters", NOT "this firm never performed this activity". Internal bids were
removed in Phase 1 and internal participation in Phase 2, so an internal firm can show
observed_as_bidder=False while having bid many times.

Different values across duplicate directory records are CROSS-RECORD CONFLICTS. They
are not proof that an attribute changed over time.

This module returns everything in memory and WRITES NOTHING.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.normalize import ascii_key_series
from common.entity_resolution import flag_internal
from common.raw_files import RawFileError, find_single_raw

SHEET = "Rapor"

# Member-directory columns we depend on. The creation-date header contains a literal
# space ("F irma") in the source export; that is not a typo on our side.
DIR_FIRMA_ID = "Firma Id"
DIR_NAME = "Firma Ünvanı"
DIR_CREATED = "F irma Oluşturma Tarihi"
DIRECTORY_REQUIRED = [DIR_FIRMA_ID, DIR_NAME, "Profil", "Tip", "Detay Tip",
                      DIR_CREATED, "Şehir", "İlçe"]

# registry column -> directory source column, for the descriptive attributes that
# follow the agreement rule (sole value -> carry, none -> null, conflict -> null).
DESCRIPTIVE_FIELDS = {
    "detay_tip": "Detay Tip",
    "tip": "Tip",
    "profil": "Profil",
    "sehir": "Şehir",
    "ilce": "İlçe",
    "firma_olusturma_tarihi": DIR_CREATED,
}

# AA Katılım columns (participation raw). FirmaID is OPTIONAL.
AA_KODU_RAW = "Açık Artırma Kodu"
AA_FIRMA_RAW = "Firma Kısa Unvanı"
AA_FIRMA_ID = "FirmaID"

# The frozen output contract, in order.
REGISTRY_COLUMNS = [
    "firma_key", "firma_adi_display",
    "in_directory", "observed_as_bidder", "observed_as_participant",
    "observed_as_buyer", "is_internal",
    "directory_firm_ids", "n_directory_firm_ids",
    "participation_firm_ids", "n_participation_ids", "has_id_collision",
    "detay_tip", "tip", "profil", "sehir", "ilce", "firma_olusturma_tarihi",
]

CONFLICT_COLUMNS = ["firma_key", "field", "value", "firma_id"]
ID_COLLISION_COLUMNS = ["firma_key", "source", "firm_ids"]


def _distinct_nonblank(series: pd.Series) -> list[str]:
    """List a column's distinct, non-blank values as trimmed text, in sorted order."""
    vals = series.dropna().astype(str).str.strip()
    return sorted({v for v in vals if v != ""})


def _serialize_ids(series: pd.Series) -> tuple[str | None, int]:
    """Turn a column of source ids into a sorted 'id|id' string plus how many there are."""
    vals = series.dropna()
    if len(vals) == 0:
        return None, 0
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().all():
        # keep integer ids readable: 6246, never 6246.0, and sort numerically
        ids = sorted({int(v) for v in numeric})
        return "|".join(str(i) for i in ids), len(ids)
    txt = sorted({str(v).strip() for v in vals if str(v).strip() != ""})
    return ("|".join(txt), len(txt)) if txt else (None, 0)


def _empty(columns: list[str]) -> pd.DataFrame:
    """Build a correctly shaped empty table so callers never special-case 'no conflicts'."""
    return pd.DataFrame({c: pd.Series(dtype="string") for c in columns})


def find_member_directory(raw_dir: Path) -> Path:
    """The single Üye-Kurumsal member-directory workbook in raw_dir (by filename keyword)."""
    try:
        return find_single_raw(raw_dir, "member_directory")
    except RawFileError as exc:
        raise FileNotFoundError(f"[firm_registry] {exc}") from None


def load_member_directory(raw_dir: Path) -> pd.DataFrame:
    """Read the Üye-Kurumsal member directory and add the canonical firm key."""
    path = find_member_directory(raw_dir)
    df = pd.read_excel(path, sheet_name=SHEET)
    missing = [c for c in DIRECTORY_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"[firm_registry] member directory is missing columns: {missing}")
    df = df.copy()
    df["firma_key"] = ascii_key_series(df[DIR_NAME])
    return df


def collapse_directory(directory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the directory to one row per firm key, nulling any field its records disagree on."""
    usable = directory.loc[directory["firma_key"].fillna("").str.len() > 0]
    conflicts: list[dict] = []
    rows: list[dict] = []

    for key, grp in usable.groupby("firma_key", sort=True):
        row: dict = {"firma_key": key}

        # --- display name: the ONE field we still fill on conflict (readability only) ---
        names = _distinct_nonblank(grp[DIR_NAME])
        row["firma_adi_display"] = names[0] if names else None   # alphabetically first
        if len(names) > 1:
            for _, r in grp.iterrows():
                val = str(r[DIR_NAME]).strip()
                if val != "":
                    conflicts.append({"firma_key": key, "field": "display_name",
                                      "value": val, "firma_id": str(r[DIR_FIRMA_ID])})

        # --- descriptive fields: sole value -> carry, none -> null, conflict -> null ---
        for out_col, src_col in DESCRIPTIVE_FIELDS.items():
            vals = _distinct_nonblank(grp[src_col])
            if len(vals) == 1:
                row[out_col] = vals[0]
            else:
                row[out_col] = None
                if len(vals) > 1:
                    # keep the record-level value -> id association intact
                    for _, r in grp.iterrows():
                        val = str(r[src_col]).strip()
                        if val != "":
                            conflicts.append({"firma_key": key, "field": out_col,
                                              "value": val,
                                              "firma_id": str(r[DIR_FIRMA_ID])})

        ids, n_ids = _serialize_ids(grp[DIR_FIRMA_ID])
        row["directory_firm_ids"] = ids
        row["n_directory_firm_ids"] = n_ids
        rows.append(row)

    attrs = pd.DataFrame(rows)
    # the carried creation date becomes a real datetime; conflicts stay text evidence
    attrs["firma_olusturma_tarihi"] = pd.to_datetime(attrs["firma_olusturma_tarihi"],
                                                     errors="coerce")
    conf = pd.DataFrame(conflicts, columns=CONFLICT_COLUMNS) if conflicts else _empty(CONFLICT_COLUMNS)
    return attrs, conf.sort_values(CONFLICT_COLUMNS).reset_index(drop=True)


def participation_key_ids(aa_raw: pd.DataFrame,
                          clean_participation_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Recover each participant key's source ids from the raw rows that survived cleaning."""
    keyed = pd.DataFrame({
        # normalised exactly as participation.py does, using the shared helper
        "aa_kodu": aa_raw[AA_KODU_RAW].astype("string").str.strip(),
        "firma_key": ascii_key_series(aa_raw[AA_FIRMA_RAW]),
    })
    if AA_FIRMA_ID in aa_raw.columns:
        keyed["firma_id"] = aa_raw[AA_FIRMA_ID].values
    else:
        keyed["firma_id"] = pd.Series([pd.NA] * len(aa_raw), index=aa_raw.index)

    # membership in the surviving pairs reuses clean_participation's own exclusions;
    # we filter the RAW rows (not the deduped table) so no distinct id is lost.
    survived = set(map(tuple, clean_participation_df[["aa_kodu", "firma_key"]].values))
    kept = keyed.loc[[(a, f) in survived for a, f in zip(keyed["aa_kodu"], keyed["firma_key"])]]

    available = AA_FIRMA_ID in aa_raw.columns and kept["firma_id"].notna().any()
    out = []
    for key, grp in kept.groupby("firma_key", sort=True):
        ids, n_ids = _serialize_ids(grp["firma_id"]) if available else (None, 0)
        out.append({"firma_key": key, "participation_firm_ids": ids,
                    "n_participation_ids": n_ids})
    frame = pd.DataFrame(out, columns=["firma_key", "participation_firm_ids",
                                       "n_participation_ids"])
    return frame, bool(available)


def observed_firm_names(clean_bids: pd.DataFrame, clean_sales: pd.DataFrame,
                        clean_participation_df: pd.DataFrame) -> pd.DataFrame:
    """Record which filtered source observed each firm, and the display names it used there."""
    parts = []
    parts.append(pd.DataFrame({
        "firma_key": ascii_key_series(clean_bids["firma_adi"]),
        "name": clean_bids["firma_adi"].astype("string"),
        "source": "bidder"}))
    sold = clean_sales.loc[clean_sales["is_sold"] == 1]
    parts.append(pd.DataFrame({
        "firma_key": ascii_key_series(sold["alici_firma_kisa_unvani"]),
        "name": sold["alici_firma_kisa_unvani"].astype("string"),
        "source": "buyer"}))
    parts.append(pd.DataFrame({
        # participation carries BOTH the key and its own display name
        "firma_key": clean_participation_df["firma_key"].astype("string"),
        "name": clean_participation_df["firma_adi"].astype("string"),
        "source": "participant"}))

    allrows = pd.concat(parts, ignore_index=True)
    allrows = allrows.loc[allrows["firma_key"].fillna("").str.len() > 0]

    out = []
    for key, grp in allrows.groupby("firma_key", sort=True):
        seen = set(grp["source"])
        out.append({
            "firma_key": key,
            "observed_as_bidder": "bidder" in seen,
            "observed_as_participant": "participant" in seen,
            "observed_as_buyer": "buyer" in seen,
            "observed_names": _distinct_nonblank(grp["name"]),
        })
    return pd.DataFrame(out, columns=["firma_key", "observed_as_bidder",
                                      "observed_as_participant", "observed_as_buyer",
                                      "observed_names"])


def registry_stats(registry: pd.DataFrame, dir_conflicts: pd.DataFrame,
                   participation_id_evidence_available: bool) -> dict:
    """Summarise the registry: universe size, source coverage, conflicts and id collisions."""
    per_field = (dir_conflicts.groupby("field")["firma_key"].nunique().to_dict()
                 if len(dir_conflicts) else {})
    return {
        "registry_keys": int(len(registry)),
        "directory_keys": int(registry["in_directory"].sum()),
        "directory_only_keys": int((registry["in_directory"]
                                    & ~registry["observed_as_bidder"]
                                    & ~registry["observed_as_participant"]
                                    & ~registry["observed_as_buyer"]).sum()),
        "activity_only_keys": int((~registry["in_directory"]).sum()),
        "observed_as_bidder": int(registry["observed_as_bidder"].sum()),
        "observed_as_participant": int(registry["observed_as_participant"].sum()),
        "observed_as_buyer": int(registry["observed_as_buyer"].sum()),
        "internal_keys": int(registry["is_internal"].sum()),
        "keys_with_directory_id_collision": int((registry["n_directory_firm_ids"] > 1).sum()),
        "keys_with_participation_id_collision": int((registry["n_participation_ids"] > 1).sum()),
        "keys_with_any_id_collision": int(registry["has_id_collision"].sum()),
        "participation_id_evidence_available": bool(participation_id_evidence_available),
        "directory_attribute_conflicts_by_field": {k: int(v) for k, v in sorted(per_field.items())},
    }


def build_firm_registry(dir_attrs: pd.DataFrame, dir_conflicts: pd.DataFrame,
                        key_ids: pd.DataFrame, participation_id_evidence_available: bool,
                        observed: pd.DataFrame,
                        internal_lookup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Union every known firm key into one registry row each, and return it with its audit."""
    keys = sorted(set(dir_attrs["firma_key"]) | set(key_ids["firma_key"])
                  | set(observed["firma_key"]))
    reg = pd.DataFrame({"firma_key": pd.Series(keys, dtype="string")})

    reg = reg.merge(dir_attrs, on="firma_key", how="left")
    reg = reg.merge(key_ids, on="firma_key", how="left")
    reg = reg.merge(observed, on="firma_key", how="left")

    reg["in_directory"] = reg["n_directory_firm_ids"].notna() & (reg["n_directory_firm_ids"] > 0)
    for col in ("observed_as_bidder", "observed_as_participant", "observed_as_buyer"):
        # a key absent from `observed` was not seen in that source -> False, not missing
        reg[col] = reg[col].eq(True)
    reg["n_directory_firm_ids"] = reg["n_directory_firm_ids"].fillna(0).astype("int16")
    reg["n_participation_ids"] = reg["n_participation_ids"].fillna(0).astype("int16")

    # a directory name is preferred; otherwise fall back to the first observed name.
    fallback = reg["observed_names"].apply(lambda v: v[0] if isinstance(v, list) and v else None)
    reg["firma_adi_display"] = reg["firma_adi_display"].fillna(fallback)

    # reuse the Phase-1 internal rule; the key is already folded, so folding is idempotent
    is_internal, _ = flag_internal(reg["firma_key"], internal_lookup)
    reg["is_internal"] = (is_internal == 1).values

    # a collision means several ids WITHIN one source. It does not prove two real firms
    # were merged, and it cannot detect every cross-source disagreement.
    reg["has_id_collision"] = (reg["n_directory_firm_ids"] > 1) | (reg["n_participation_ids"] > 1)

    blank = reg["firma_key"].fillna("").str.len() == 0
    if blank.any():
        raise ValueError(f"[firm_registry] {int(blank.sum())} blank firma_key row(s) — "
                         "a registry key must be usable")
    if reg["firma_key"].duplicated().any():
        dup = reg.loc[reg["firma_key"].duplicated(), "firma_key"].tolist()[:5]
        raise ValueError(f"[firm_registry] duplicate firma_key rows: {dup}")

    for col in ("firma_key", "firma_adi_display", "directory_firm_ids",
                "participation_firm_ids", "detay_tip", "tip", "profil", "sehir", "ilce"):
        reg[col] = reg[col].astype("string")
    reg["firma_olusturma_tarihi"] = pd.to_datetime(reg["firma_olusturma_tarihi"], errors="coerce")
    registry = reg[REGISTRY_COLUMNS].sort_values("firma_key").reset_index(drop=True)

    coll_rows = []
    for src, ncol, idcol in (("directory", "n_directory_firm_ids", "directory_firm_ids"),
                             ("participation", "n_participation_ids", "participation_firm_ids")):
        hit = registry.loc[registry[ncol] > 1, ["firma_key", idcol]]
        for _, r in hit.iterrows():
            coll_rows.append({"firma_key": r["firma_key"], "source": src,
                              "firm_ids": r[idcol]})
    id_collisions = (pd.DataFrame(coll_rows, columns=ID_COLLISION_COLUMNS)
                     if coll_rows else _empty(ID_COLLISION_COLUMNS))

    audit = {
        "directory_attribute_conflicts": dir_conflicts,
        "id_collisions": id_collisions.sort_values(ID_COLLISION_COLUMNS).reset_index(drop=True),
        "counts": registry_stats(registry, dir_conflicts, participation_id_evidence_available),
    }
    return registry, audit

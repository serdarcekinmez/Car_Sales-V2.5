"""
io_load.py  —  load the 3 raw workbooks Phase 1 needs and assert their schema.

Every raw file has a single sheet called "Rapor" (confirmed crash-point #1 of the
old code, which hard-coded "Tüm Rapor"/"Reel DB" sheet names that no longer exist).
We locate files by keyword in the filename (spaces, not underscores) and check the
columns we depend on are actually present before any cleaning runs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SHEET = "Rapor"

# role -> keywords that identify the file (case-insensitive, accent-tolerant).
FILE_ROLES = {
    "satis": ["detay satış", "detay satis"],   # vehicle listings + sale outcome
    "teklif": ["detay teklif"],                 # individual bids
    "uye": ["üye", "kurumsal"],                 # member-firm directory (Detay Tip)
}

# Columns each downstream step relies on. If one is missing we stop loudly here
# rather than crash deep inside a cleaner with a confusing KeyError.
REQUIRED_COLUMNS = {
    "satis": [
        "Sıra", "AA Tarihi", "AA Kodu", "Plaka", "Satıcı Id", "Satıcı Kısa Ünvan",
        "Marka", "Model", "Model Uzantısı", "Model Yılı", "Renk", "Yakıt Tipi",
        "Vites Tipi", "Kasa Tipi", "KM",
        "Ekspertiz", "Eks.Puanı", "Tramer Adedi", "Tramer Tutarı", "Lokasyon",
        "Min.Fiyat", "Satış Tutarı", "Satış Tipi",
        "AA Girişi", "AA Esnası Tamamlama", "Alıcı Id", "Alıcı Firma Kısa Ünvanı",
    ],
    "teklif": [
        "Plaka", "AA Kodu", "AA Tarihi", "Teklif", "Firma Adı",
        "Teklif Tarihi Tam",
    ],
    "uye": ["Firma Ünvanı", "Detay Tip"],
}


def find_raw_files(raw_dir: Path) -> dict[str, Path]:
    """Map role -> Path by matching keywords inside each .xlsx filename."""
    found: dict[str, Path] = {}
    for path in sorted(raw_dir.glob("*.xlsx")):
        name = path.name.lower()
        for role, keys in FILE_ROLES.items():
            if role not in found and any(key in name for key in keys):
                found[role] = path
    return found


def load_rapor(path: Path) -> pd.DataFrame:
    """Read the 'Rapor' sheet of one workbook (raw files are read-only)."""
    return pd.read_excel(path, sheet_name=SHEET)


def assert_schema(df: pd.DataFrame, role: str) -> None:
    """Fail fast if any required column for this role is missing."""
    missing = [c for c in REQUIRED_COLUMNS[role] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[io_load] '{role}' is missing required columns: {missing}\n"
            f"  Got columns: {list(df.columns)}"
        )


def load_phase1_inputs(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load satis, teklif, uye from raw_dir, asserting each schema.
    'uye' (member directory) is optional: if absent we return None for it and
    entity_resolution falls back to its regex classifier.
    """
    files = find_raw_files(raw_dir)
    for role in ("satis", "teklif"):
        if role not in files:
            raise SystemExit(f"[io_load] could not locate required raw file for '{role}'")

    data: dict[str, pd.DataFrame] = {}
    for role in ("satis", "teklif"):
        df = load_rapor(files[role])
        assert_schema(df, role)
        data[role] = df

    if "uye" in files:
        uye = load_rapor(files["uye"])
        assert_schema(uye, "uye")
        data["uye"] = uye
    else:
        data["uye"] = None

    data["_paths"] = files  # keep paths for the report / version id
    return data

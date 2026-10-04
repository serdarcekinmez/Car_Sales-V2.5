"""
raw_files.py  —  find a raw export in Datasets/raw/ by its role, not by its exact name.

Exports are re-downloaded with a new date in the filename, so no caller names a file.
Each role has filename keywords (case-insensitive, Unicode-NFC so macOS-style decomposed
names match too). The raw folder must hold EXACTLY ONE workbook per role: none or several
is an error that lists the candidates. There is deliberately no "newest file" guess;
archive old exports outside the raw folder.

  find_single_raw(raw_dir, role)   -> Path  (raises RawFileError)
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

# role -> keywords that identify the workbook. "member_directory" matches
# phase1_preparation.io_load.FILE_ROLES["uye"] plus the ASCII spelling.
RAW_ROLES = {
    "aa_katilim": ("katılım", "katilim"),
    "member_directory": ("üye", "uye", "kurumsal"),
}


class RawFileError(FileNotFoundError):
    """No workbook, or more than one, matches a role."""


def _norm(name: str) -> str:
    return unicodedata.normalize("NFC", name).lower()


def find_single_raw(raw_dir, role: str) -> Path:
    """The one .xlsx in raw_dir whose filename carries a keyword of `role`."""
    keys = RAW_ROLES[role]
    hits = [p for p in sorted(Path(raw_dir).glob("*.xlsx"))
            if not p.name.startswith("~$") and any(k in _norm(p.name) for k in keys)]
    if not hits:
        raise RawFileError(f"no '{role}' workbook (.xlsx named with one of {list(keys)}) "
                           f"in {raw_dir}")
    if len(hits) > 1:
        raise RawFileError(f"several '{role}' workbooks in {raw_dir}: "
                           f"{[p.name for p in hits]}; keep exactly one (archive the others "
                           f"outside the raw folder)")
    return hits[0]

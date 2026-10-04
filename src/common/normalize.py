"""
normalize.py  —  Phase 1 PRIMITIVES (text / number / plate normalisation + book-keeping).

These are the small, boring building blocks every other Phase-1 module leans on.
They are deliberately tiny and plain-commented so each one can be read on its own.

Carried over (in behaviour, not copy-paste) from the old `clean_ml_datasets.py`:
  * ascii_key_series   — Turkish-aware folding, the heart of firm-name matching
  * parse_numeric_value — Turkish money/number parser ("1.234.567,89" -> 1234567.89)
  * to_snake            — column-name normalisation
  * standardize_plaka   — licence-plate cleanup + format flag
  * add_flag / add_log  — the "mark it, don't drop it" book-keeping helpers
"""
from __future__ import annotations

import re

import pandas as pd

# ----------------------------------------------------------------------------
# Turkish -> ASCII folding tables.
#
# THE TRAP (see data_profile.md): Python's str.upper() folds dotted İ -> I but
# leaves cedilla Ç un-folded, so a literal regex "UCAR" matches 0 rows even
# though the firm is spelled "UÇAR". We therefore fold the Turkish letters to
# ASCII *before* upper-casing / stripping. Every firm-name match runs on the
# folded key, never on the raw text.
# ----------------------------------------------------------------------------
_FOLD_PAIRS = {
    "Ç": "C", "ç": "C",
    "İ": "I", "ı": "I",
    "Ş": "S", "ş": "S",
    "Ğ": "G", "ğ": "G",
    "Ü": "U", "ü": "U",
    "Ö": "O", "ö": "O",
    "Â": "A", "â": "A", "Î": "I", "î": "I", "Û": "U", "û": "U",  # rare circumflex forms
}
# Uppercase fold map (for ASCII keys) and a lowercase one (for snake_case).
_FOLD_UPPER = {ord(k): v for k, v in _FOLD_PAIRS.items()}
_FOLD_LOWER = {ord(k): v.lower() for k, v in _FOLD_PAIRS.items()}


def ascii_key_series(series: pd.Series) -> pd.Series:
    """
    Fold a name column to a strict ASCII matching key.
    Steps: Turkish letters -> ASCII, UPPER-CASE, then drop everything that is
    not A-Z or 0-9. "UÇAR Otomotiv Ümraniye" -> "UCAROTOMOTIVUMRANIYE".
    NaN/blank become "" so they never accidentally match a real firm.
    """
    out = series.astype("string").fillna("")
    out = out.str.translate(_FOLD_UPPER)      # fold BEFORE upper (the trap fix)
    out = out.str.upper()
    out = out.str.replace(r"[^A-Z0-9]", "", regex=True)
    return out


def ascii_key(text: object) -> str:
    """Scalar version of ascii_key_series — handy for single lookup values."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).translate(_FOLD_UPPER).upper()
    return re.sub(r"[^A-Z0-9]", "", s)


# Translation table reused by to_snake / column normalisation.
SNAKE_TRANSLATION = _FOLD_LOWER


def to_snake(name: str) -> str:
    """
    Turn a raw (possibly Turkish) column header into a snake_case ASCII name.
    "AA Esnası Tamamlama" -> "aa_esnasi_tamamlama" ; "Min.Fiyat" -> "min_fiyat".
    """
    n = str(name).strip().translate(SNAKE_TRANSLATION).lower()
    n = re.sub(r"[^a-z0-9]+", "_", n)
    return n.strip("_")


def parse_numeric_value(series: pd.Series) -> pd.Series:
    """
    Parse messy numeric text into float, understanding Turkish formatting where
    '.' is the thousands separator and ',' is the decimal comma.

    Strategy: try a plain numeric parse first (handles "168691", "10.5", "-1").
    Whatever fails to parse is retried as Turkish format ("1.234.567,89"):
    drop the dots, swap the comma for a dot. This never corrupts a genuine
    "10.5" because that already parsed in the first pass.
    """
    s = series.astype("string").str.strip()
    direct = pd.to_numeric(s, errors="coerce")
    turkish = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    turkish_num = pd.to_numeric(turkish, errors="coerce")
    return direct.fillna(turkish_num)


# Valid Turkish plate: 2 province digits, 1-3 letters, 2-4 digits (e.g. 34BKT848).
_PLAKA_RE = r"^\d{2}[A-Z]{1,3}\d{2,4}$"


def standardize_plaka(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Clean a plate column (strip spaces/punctuation, upper-case) and return
    (clean_plaka, is_valid_format_flag). We never drop rows on a bad plate —
    we only raise the flag so it can be inspected later.
    """
    clean = (series.astype("string").fillna("")
             .str.upper()
             .str.replace(r"[^A-Z0-9]", "", regex=True))
    valid = clean.str.match(_PLAKA_RE, na=False).astype("int8")
    return clean, valid


# ----------------------------------------------------------------------------
# "Mark it, don't drop it" book-keeping.
#
# add_flag  -> one summary row per flag for cleaning_flags.csv (flag, count, why)
# add_log   -> one row per pipeline step for pipeline_log.csv (what happened)
# The per-row 1/0 flag columns themselves live inside the parquet outputs; these
# two helpers only accumulate the human-readable audit trail.
# ----------------------------------------------------------------------------
def add_flag(flags: list[dict], table: str, flag: str, mask: pd.Series,
             description: str) -> None:
    """Record how many rows a per-row flag hit, plus a one-line reason."""
    flags.append({
        "table": table,
        "flag": flag,
        "n_rows": int(pd.Series(mask).fillna(False).astype(bool).sum()),
        "description": description,
    })


def add_log(log: list[dict], step: str, action: str, detail: str,
            rows_in: int | None = None, rows_out: int | None = None) -> None:
    """Record one pipeline action (load, drop, join, write, assert …)."""
    log.append({
        "step": step,
        "action": action,
        "detail": detail,
        "rows_in": rows_in,
        "rows_out": rows_out,
    })

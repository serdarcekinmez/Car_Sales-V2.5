"""
store.py  —  Phase 8: SQLite store of immutable predictions and their feedback.

Standard library sqlite3 only; every statement is parameterized. Nothing happens at import
time; Store.initialize() creates the file and schema.

PREDICTIONS are written once with the exact server-calculated Phase-7 response
(predict.response_json), the validated request that produced it, the owner, the parent
prediction (a filter rerun of the same vehicle) and the identities that explain a change:
bundle_id (data + models + Phase-7 code + runtime), filters_version + rules_sha256 and
serving_code_sha256 (Phase-8 code). SQLite triggers forbid UPDATE and DELETE, so an
output can never be replaced after it was shown; a rerun is a new row.

FEEDBACK is append-only and linked to one prediction of the same owner. Its identity
columns (bundle, rules, serving code) are copied from that prediction inside the same
transaction, never taken from the client. A firm verdict is accepted only for a firm the
caller validated against that prediction's shown call list (app.py). The number of
feedback rows per prediction is bounded.

OWNERSHIP. Every read and write filters on owner: another account's prediction is
reported as not found (no existence oracle).

CONCURRENCY. One process (pipeline/run_demo.py runs one worker); a process-wide lock
serializes writes, WAL mode lets reads proceed.

PUBLIC API
  Store(path).initialize()
  save_prediction(record: PredictionRecord)
  get_prediction(prediction_id, owner)          -> dict | None
  list_predictions(owner, limit)                -> list[dict]
  add_feedback(prediction_id, owner, **fields)  -> dict
  feedback_for(prediction_id, owner)            -> list[dict]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "phase8-store-v1"
MAX_FEEDBACK_PER_PREDICTION = 400
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MODES = ("current", "historical_replay")
FIRM_VERDICTS = ("uygun", "uygun_degil")
PRICE_VERDICTS = ("dusuk", "makul", "yuksek")

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       TEXT PRIMARY KEY CHECK (length(prediction_id) = 32),
    owner               TEXT NOT NULL,
    created_at_utc      TEXT NOT NULL,
    stored_at_utc       TEXT NOT NULL,
    parent_id           TEXT REFERENCES predictions (prediction_id),
    mode                TEXT NOT NULL CHECK (mode IN ('current', 'historical_replay')),
    status              TEXT NOT NULL,
    prediction_date     TEXT NOT NULL,
    title               TEXT NOT NULL,
    bundle_id           TEXT NOT NULL,
    filters_version     TEXT NOT NULL,
    rules_sha256        TEXT NOT NULL,
    serving_code_sha256 TEXT NOT NULL,
    request_json        TEXT NOT NULL,
    response_json       TEXT NOT NULL,
    response_sha256     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_owner_created
    ON predictions (owner, stored_at_utc);
CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id       TEXT NOT NULL REFERENCES predictions (prediction_id),
    owner               TEXT NOT NULL,
    created_at_utc      TEXT NOT NULL,
    kind                TEXT NOT NULL CHECK (kind IN ('firm', 'price')),
    firma_key           TEXT,
    firma_adi           TEXT,
    firm_rank           INTEGER,
    verdict             TEXT NOT NULL,
    reason_code         TEXT,
    suggested_price_tl  REAL,
    comment             TEXT,
    shown_price_tl      REAL,
    bundle_id           TEXT NOT NULL,
    rules_sha256        TEXT NOT NULL,
    serving_code_sha256 TEXT NOT NULL,
    CHECK ((kind = 'firm' AND firma_key IS NOT NULL
            AND verdict IN ('uygun', 'uygun_degil') AND suggested_price_tl IS NULL)
        OR (kind = 'price' AND firma_key IS NULL
            AND verdict IN ('dusuk', 'makul', 'yuksek')))
);
CREATE INDEX IF NOT EXISTS feedback_prediction ON feedback (prediction_id);
CREATE TRIGGER IF NOT EXISTS feedback_no_update BEFORE UPDATE ON feedback
BEGIN SELECT RAISE(ABORT, 'feedback is append-only'); END;
CREATE TRIGGER IF NOT EXISTS feedback_no_delete BEFORE DELETE ON feedback
BEGIN SELECT RAISE(ABORT, 'feedback is append-only'); END;
"""

_FEEDBACK_COLUMNS = ("feedback_id", "prediction_id", "created_at_utc", "kind", "firma_key",
                     "firma_adi", "firm_rank", "verdict", "reason_code", "suggested_price_tl",
                     "comment", "shown_price_tl")


class StoreError(RuntimeError):
    """The store cannot complete the operation; the message is safe to show."""


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    owner: str
    created_at_utc: str
    parent_id: str | None
    mode: str
    status: str
    prediction_date: str
    title: str
    bundle_id: str
    filters_version: str
    rules_sha256: str
    serving_code_sha256: str
    request_json: str
    response_json: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        """Create the file / schema, or verify an existing store has this schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_DDL)
            with conn:
                row = conn.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
                if row is None:
                    conn.execute("INSERT INTO meta (key, value) VALUES ('schema', ?)",
                                 (SCHEMA_VERSION,))
                elif row["value"] != SCHEMA_VERSION:
                    raise StoreError(f"store schema {row['value']!r} is not {SCHEMA_VERSION!r}")

    # ------------------------------------------------------------------ predictions
    def save_prediction(self, record: PredictionRecord) -> None:
        if not isinstance(record, PredictionRecord):
            raise TypeError("record must be a PredictionRecord")
        if record.mode not in MODES:
            raise StoreError("unknown prediction mode")
        if len(record.response_json.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise StoreError("prediction response is too large to store")
        json.loads(record.response_json)                # must be valid JSON text
        json.loads(record.request_json)
        digest = hashlib.sha256(record.response_json.encode("utf-8")).hexdigest()
        values = asdict(record)
        with self._lock, closing(self._connect()) as conn, conn:
            if record.parent_id is not None:
                parent = conn.execute(
                    "SELECT 1 FROM predictions WHERE prediction_id = ? AND owner = ?",
                    (record.parent_id, record.owner)).fetchone()
                if parent is None:
                    raise StoreError("parent prediction not found")
            conn.execute(
                "INSERT INTO predictions (prediction_id, owner, created_at_utc, stored_at_utc, "
                "parent_id, mode, status, prediction_date, title, bundle_id, filters_version, "
                "rules_sha256, serving_code_sha256, request_json, response_json, "
                "response_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (values["prediction_id"], values["owner"], values["created_at_utc"], _now(),
                 values["parent_id"], values["mode"], values["status"],
                 values["prediction_date"], values["title"], values["bundle_id"],
                 values["filters_version"], values["rules_sha256"],
                 values["serving_code_sha256"], values["request_json"],
                 values["response_json"], digest))

    def get_prediction(self, prediction_id: str, owner: str) -> dict | None:
        """The stored row with parsed request / response, or None (absent or not owned)."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM predictions WHERE prediction_id = ? AND owner = ?",
                               (prediction_id, owner)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["request"] = json.loads(out.pop("request_json"))
        out["response"] = json.loads(out.pop("response_json"))
        return out

    def list_predictions(self, owner: str, limit: int = 20) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT prediction_id, parent_id, stored_at_utc, mode, status, "
                "prediction_date, title FROM predictions WHERE owner = ? "
                "ORDER BY stored_at_utc DESC, rowid DESC LIMIT ?", (owner, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ feedback
    def add_feedback(self, prediction_id: str, owner: str, *, kind: str, verdict: str,
                     firma_key: str | None = None, firma_adi: str | None = None,
                     firm_rank: int | None = None, reason_code: str | None = None,
                     suggested_price_tl: float | None = None, comment: str | None = None,
                     shown_price_tl: float | None = None) -> dict:
        """Append one feedback row; identities come from the stored prediction."""
        if kind == "firm" and (verdict not in FIRM_VERDICTS or not firma_key):
            raise StoreError("invalid firm feedback")
        if kind == "price" and (verdict not in PRICE_VERDICTS or firma_key is not None):
            raise StoreError("invalid price feedback")
        if kind not in ("firm", "price"):
            raise StoreError("invalid feedback kind")
        created = _now()
        with self._lock, closing(self._connect()) as conn, conn:
            pred = conn.execute(
                "SELECT bundle_id, rules_sha256, serving_code_sha256 FROM predictions "
                "WHERE prediction_id = ? AND owner = ?", (prediction_id, owner)).fetchone()
            if pred is None:
                raise StoreError("prediction not found")
            n = conn.execute("SELECT COUNT(*) FROM feedback WHERE prediction_id = ?",
                             (prediction_id,)).fetchone()[0]
            if n >= MAX_FEEDBACK_PER_PREDICTION:
                raise StoreError("feedback limit reached for this prediction")
            cur = conn.execute(
                "INSERT INTO feedback (prediction_id, owner, created_at_utc, kind, firma_key, "
                "firma_adi, firm_rank, verdict, reason_code, suggested_price_tl, comment, "
                "shown_price_tl, bundle_id, rules_sha256, serving_code_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (prediction_id, owner, created, kind, firma_key, firma_adi, firm_rank, verdict,
                 reason_code, suggested_price_tl, comment, shown_price_tl, pred["bundle_id"],
                 pred["rules_sha256"], pred["serving_code_sha256"]))
            feedback_id = cur.lastrowid
        return {"feedback_id": feedback_id, "prediction_id": prediction_id,
                "created_at_utc": created, "kind": kind, "firma_key": firma_key,
                "firma_adi": firma_adi, "firm_rank": firm_rank, "verdict": verdict,
                "reason_code": reason_code, "suggested_price_tl": suggested_price_tl,
                "comment": comment, "shown_price_tl": shown_price_tl}

    def feedback_for(self, prediction_id: str, owner: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_FEEDBACK_COLUMNS)} FROM feedback "
                "WHERE prediction_id = ? AND owner = ? ORDER BY feedback_id",
                (prediction_id, owner)).fetchall()
        return [dict(r) for r in rows]

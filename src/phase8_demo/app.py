"""
app.py  —  Phase 8: the private customer web demo (FastAPI) over the Phase-7 API.

One process serves one pilot customer account. The browser gets a static Turkish
workspace (static/) and talks to the JSON API below; every prediction is computed HERE by
phase7_inference.predict with the frozen models. Nothing is approximated in the browser.

STARTUP. create_app(settings) validates the configuration, opens the SQLite store and
returns the ASGI app wrapped in security.GuardMiddleware. The Phase-7 bundle (Excel /
parquet reads, history indexes, both models) is loaded ONCE in a background thread at
startup; until it is ready the API answers 503 with a readable state, and a failed load
shows a sanitized message (logical input names only, no server path).

PREDICTIONS. One prediction runs at a time (a process-wide lock: the frozen runtime is not
promised thread-safe). Endpoints are async only to read the body; model work runs in the
thread pool, so the event loop is never blocked. Each result is stored once, exactly as
Phase 7 returned it (predict.response_json), with the validated request, owner, parent
prediction and identities: bundle_id, filters version + rules_sha256 and this serving
code's SHA-256 (SERVING_FILES). Changing filters reruns the SAME stored request and
creates a NEW prediction whose parent is the old one; stored outputs never change.

BROWSER PROJECTION (client_view). The browser receives a reduced, display-only view of
the stored response: no ranking score (never a probability), no file hashes or runtime
library versions, at most MAX_EXCLUDED_VIEW excluded firms, and for a historical replay no
plate / query id and no recorded sale outcome. History snippets are the factual Phase-7
lines. Raw sales, bids, the member directory and model files never leave the server.

INPUT. Vehicle fields use the Phase-7 RAW_COLUMNS names. Numbers must arrive as finite
JSON numbers (not "120.000" strings: Phase 1 would read that as 120); booleans, NaN /
Infinity, nested values and over-long strings are refused before Phase 7 runs, and
Phase-7 InputError messages are passed through as 400. Optional fields are omitted, never
defaulted. Filters use only the Phase-7 rule fields.

HISTORICAL EXAMPLES. A bounded deterministic sample of recent recorded vehicles is offered
with opaque ids and a few display fields; the plate / query id stays on the server. A
replay is a demonstration only (the frozen models may have seen that period). A Phase-7
InputError on a replay becomes a generic 400 example_invalid (the Phase-7 text is dropped).

SHUTDOWN. A local run (pipeline/run_demo.py serve --allow-shutdown, loopback host only) gives
create_app an on_shutdown callback; the signed-in user can then stop the whole server from
the page ("Oturumu kapat", POST /api/shutdown). Without the callback the route answers 404,
so a public deployment can never be stopped from the browser.

ACCESS. Every /api route except /api/login, /api/session and /api/password-reset/*
requires a session; every mutating route requires CSRF + JSON + allowed Origin
(security.py). Predictions, CSV and feedback are filtered by owner (the configured account)
in the store. One account serves one customer; separate customers need separate
deployments. The account signs in with its user name or its e-mail (GARAJ_DEMO_EMAIL).

PASSWORD RESET ("Parolamı unuttum"). Only when GARAJ_DEMO_EMAIL and a mailer (mailer.py)
are configured. /request e-mails a 6-digit code when the identifier names the account (the
answer is the same either way, and sending runs in the background); /verify exchanges a
correct code for a reset token; /complete sets the new password (rules in security.py,
typed twice), saves its hash in the store (Account), signs every browser out and e-mails a
notice. Requests and wrong codes are throttled per client like logins.

PUBLIC API
  DemoSettings, settings_from_env(environ, ...),
  create_app(settings, on_shutdown=None, mailer=None), Account
  serving_code_sha256(), client_view(response, ...)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import re
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from phase7_inference import bundle as BD
from phase7_inference import filters as FL
from phase7_inference import predict as PR
from phase7_inference.vehicle_input import KEY_SEPARATOR, RAW_COLUMNS
from phase8_demo import security as SEC
from phase8_demo.mailer import valid_address
from phase8_demo.store import PredictionRecord, Store, StoreError

log = logging.getLogger(__name__)

SERVING_FORMAT = "phase8-demo-v1"
_PKG = Path(__file__).resolve().parent
_SRC = _PKG.parent
_REPO = _SRC.parent
SERVING_FILES = ("phase8_demo/__init__.py", "phase8_demo/app.py", "phase8_demo/store.py",
                 "phase8_demo/security.py", "phase8_demo/mailer.py",
                 "phase8_demo/static/index.html",
                 "phase8_demo/static/app.js", "phase8_demo/static/style.css")
RUNNER_FILE = "pipeline/run_demo.py"
PAGES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/static/style.css": ("style.css", "text/css; charset=utf-8")}
DEFAULT_STORE = "Datasets/phase8_demo/demo.sqlite3"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

MAX_BODY_BYTES = 64 * 1024
MAX_COMMENT_CHARS = 1000
MAX_EXCLUDED_VIEW = 50
MAX_EXCLUDE_KEYS = 200
MAX_FIRMA_KEY_CHARS = 200
MAX_CITIES = 100
MAX_CITY_CHARS = 80
PREDICT_LOCK_TIMEOUT_S = 120
LOGIN_CONCURRENCY = 2                   # simultaneous scrypt password checks per process
LOGIN_BUSY_RETRY_S = 5
RESET_SENT_TR = ("Bu bilgilere kayıtlı bir hesap varsa, hesabın e-posta adresine 6 haneli bir "
                 "kod gönderildi. Kod 10 dakika geçerlidir.")
EXAMPLE_COUNT = 60
EXAMPLE_DAYS = 180
EXAMPLE_SEED = 20261004
MIN_CATEGORY_COUNT = 3
MODELS_PER_MARKA = 80
SUGGEST_LIMITS = {"marka": 200, "yakit_tipi": 40, "vites_tipi": 40, "kasa_tipi": 60,
                  "renk": 60, "lokasyon": 200}

# Vehicle fields (RAW_COLUMNS): text -> max length; number -> (integer?, min, max).
TEXT_FIELDS = {"Plaka": 16, "Marka": 60, "Model": 100, "Model Uzantısı": 160,
               "Ekspertiz": 4000, "Satıcı Kısa Ünvan": 160, "Yakıt Tipi": 40,
               "Vites Tipi": 40, "Kasa Tipi": 60, "Renk": 40, "Lokasyon": 100}
NUMBER_FIELDS = {"Model Yılı": (True, 1950, 2100), "KM": (True, 0, 3_000_000),
                 "Eks.Puanı": (False, 0, 10), "Tramer Adedi": (True, 0, 500),
                 "Tramer Tutarı": (False, 0, 1_000_000_000),
                 "Min.Fiyat": (False, 0, 10_000_000_000)}
FIRM_REASONS = {"bolge_uygun_degil": "Bölge uygun değil",
                "arac_tipiyle_ilgilenmez": "Bu araç tipiyle ilgilenmez",
                "fiyat_seviyesi_uymaz": "Fiyat seviyesi uymaz",
                "calisilmiyor": "Bu firmayla çalışılmıyor",
                "iletisim_sorunu": "İletişim sorunu",
                "diger": "Diğer"}
HISTORY_LEVEL_TR = {"S1": "aynı marka-model-uzantı-yıl", "S2": "aynı marka-model-yıl",
                    "S3": "aynı marka-model", "S4": "aynı marka"}

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PID_RE = re.compile(r"[0-9a-f]{32}")
_EXAMPLE_RE = re.compile(r"[0-9a-f]{20}")
_CONTROL_LINE = re.compile(r"[\x00-\x1f\x7f]")
_CONTROL_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CSV_RISKY = ("=", "+", "-", "@", "\t", "\r", "\n")


class ConfigError(ValueError):
    """Invalid or missing demo configuration (raised before the server starts)."""


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


# ----------------------------------------------------------------------------
# Settings.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoSettings:
    root: Path                          # BundleConfig root (data + model files)
    store_path: Path                    # writable SQLite file
    username: str
    password_hash: str                  # security.hash_password format
    email: str = ""                     # sign-in alternative; password-reset address
    secure_cookies: bool = True
    allowed_hosts: frozenset = LOCAL_HOSTS
    trusted_proxies: frozenset = frozenset()
    session_hours: float = 8.0

    def validate(self) -> None:
        if not isinstance(self.username, str) or not self.username.strip() \
                or self.username != self.username.strip() \
                or len(self.username) > SEC.MAX_USERNAME_CHARS:
            raise ConfigError("GARAJ_DEMO_USERNAME must be a non-blank name of at most "
                              f"{SEC.MAX_USERNAME_CHARS} characters without outer spaces")
        try:
            SEC.parse_password_hash(self.password_hash)
        except ValueError as exc:
            raise ConfigError(f"GARAJ_DEMO_PASSWORD_HASH is invalid: {exc}") from None
        if self.email and not valid_address(self.email):
            raise ConfigError("GARAJ_DEMO_EMAIL must be an e-mail address (or empty)")
        if not Path(self.root).is_dir():
            raise ConfigError("the data root (GARAJ_DEMO_ROOT / --root) is not a directory")
        if not 0.25 <= float(self.session_hours) <= 24:
            raise ConfigError("GARAJ_DEMO_SESSION_HOURS must be between 0.25 and 24")
        if not self.allowed_hosts:
            raise ConfigError("at least one allowed host is required")


def _csv_env(text: str | None) -> list[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def settings_from_env(environ, *, root=None, store_path=None, secure_cookies: bool = True,
                      allowed_hosts=None) -> DemoSettings:
    """Settings from GARAJ_DEMO_* variables; explicit arguments (CLI) take precedence."""
    username = environ.get("GARAJ_DEMO_USERNAME", "")
    password_hash = environ.get("GARAJ_DEMO_PASSWORD_HASH", "")
    if not username or not password_hash:
        raise ConfigError("GARAJ_DEMO_USERNAME and GARAJ_DEMO_PASSWORD_HASH must be set "
                          "(python pipeline/run_demo.py hash-password)")
    root = Path(root or environ.get("GARAJ_DEMO_ROOT") or _REPO).resolve()
    store = Path(store_path or environ.get("GARAJ_DEMO_STORE") or root / DEFAULT_STORE)
    hosts = allowed_hosts if allowed_hosts is not None else (
        _csv_env(environ.get("GARAJ_DEMO_ALLOWED_HOSTS")) or sorted(LOCAL_HOSTS))
    hosts = frozenset(SEC.normalize_ip(h.strip().strip("[]")) or SEC.host_name(h)
                      for h in hosts)
    proxies = set()
    for p in _csv_env(environ.get("GARAJ_DEMO_TRUSTED_PROXIES")):
        ip = SEC.normalize_ip(p)
        if ip is None:
            raise ConfigError(f"GARAJ_DEMO_TRUSTED_PROXIES holds an invalid IP address {p!r}")
        proxies.add(ip)
    try:
        hours = float(environ.get("GARAJ_DEMO_SESSION_HOURS", "8"))
    except ValueError:
        raise ConfigError("GARAJ_DEMO_SESSION_HOURS must be a number") from None
    settings = DemoSettings(root=root, store_path=store, username=username,
                            password_hash=password_hash.strip(),
                            email=environ.get("GARAJ_DEMO_EMAIL", "").strip(),
                            secure_cookies=secure_cookies, allowed_hosts=hosts,
                            trusted_proxies=frozenset(proxies), session_hours=hours)
    settings.validate()
    return settings


class Account:
    """The pilot account. Its password hash is the configured one (GARAJ_DEMO_PASSWORD_HASH)
    until a password reset saves a newer one in the store; the saved hash is used only
    while the configured hash is still the one it replaced (see store.py)."""

    def __init__(self, username: str, email: str, configured_hash: str, store: Store):
        self.username, self.email = username, email
        self._store = store
        self._base = hashlib.sha256(configured_hash.encode("utf-8")).hexdigest()
        self._lock = threading.Lock()
        self._hash = configured_hash
        saved = store.get_password_hash(username, self._base)
        if saved is not None:
            try:
                SEC.parse_password_hash(saved)
                self._hash = saved
            except ValueError:
                log.warning("phase8 demo: ignoring an invalid saved password hash")

    def matches(self, identifier) -> bool:
        return SEC.identifier_matches(identifier, expected_username=self.username,
                                      expected_email=self.email)

    def check(self, identifier, password) -> bool:
        with self._lock:
            encoded = self._hash
        return SEC.check_credentials(identifier, password, expected_username=self.username,
                                     expected_email=self.email, password_hash=encoded)

    def set_password(self, password: str) -> None:
        """Hash (rules enforced: ValueError), save, and use from now on."""
        encoded = SEC.hash_password(password)
        self._store.save_password_hash(self.username, encoded, self._base)
        with self._lock:
            self._hash = encoded


def serving_code_sha256() -> str:
    """SHA-256 identity of the Phase-8 serving code (stored next to every bundle_id)."""
    files = {}
    for rel in SERVING_FILES:
        files[rel] = hashlib.sha256((_SRC / rel).read_bytes()).hexdigest()
    runner = _REPO / RUNNER_FILE
    files[RUNNER_FILE] = hashlib.sha256(runner.read_bytes()).hexdigest() \
        if runner.is_file() else None
    text = json.dumps({"format": SERVING_FORMAT, "files": files}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Request validation.
# ----------------------------------------------------------------------------
def _reject_constant(name):
    raise ValueError(f"non-finite number {name}")


def _keys(body: dict, allowed: set, what: str) -> None:
    unknown = sorted(str(k) for k in body if k not in allowed)
    if unknown:
        raise ApiError(400, "unknown_field", f"{what} içinde tanınmayan alan: "
                                             f"{', '.join(unknown[:5])}")


def _text(value, name: str, limit: int, *, multiline: bool = False) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError(400, "invalid_field", f"{name} metin olmalı.")
    text = value.strip()
    if not text:
        return None
    if len(text) > limit:
        raise ApiError(400, "too_long", f"{name} en fazla {limit} karakter olabilir.")
    if (_CONTROL_TEXT if multiline else _CONTROL_LINE).search(text):
        raise ApiError(400, "invalid_field", f"{name} geçersiz kontrol karakteri içeriyor.")
    return text


def _number(value, name: str, *, integer: bool, low: float, high: float):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, "invalid_number", f"{name} sayı olarak gönderilmeli (ör. 120000); "
                                              f"metin kabul edilmez.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ApiError(400, "invalid_number", f"{name} sonlu bir sayı olmalı.")
        if integer:
            if not value.is_integer():
                raise ApiError(400, "invalid_number", f"{name} tam sayı olmalı.")
            value = int(value)
    if not low <= value <= high:
        raise ApiError(400, "out_of_range", f"{name} {low:g} ile {high:g} arasında olmalı.")
    return value


def validate_vehicle(value) -> dict:
    """Validated vehicle mapping: blank fields omitted, never defaulted."""
    if not isinstance(value, dict):
        raise ApiError(400, "invalid_vehicle", "Araç bilgisi bir nesne olmalı.")
    _keys(value, set(RAW_COLUMNS), "Araç bilgisi")
    out = {}
    for name in RAW_COLUMNS:
        if name not in value or value[name] is None:
            continue
        if name in TEXT_FIELDS:
            v = _text(value[name], name, TEXT_FIELDS[name], multiline=name == "Ekspertiz")
        else:
            integer, low, high = NUMBER_FIELDS[name]
            v = _number(value[name], name, integer=integer, low=low, high=high)
        if v is not None:
            out[name] = v
    if "Marka" not in out:
        raise ApiError(400, "marka_required", "Marka zorunludur.")
    return out


def validate_date(value) -> str:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ApiError(400, "invalid_date", "Tahmin tarihi YYYY-AA-GG biçiminde olmalı.")
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise ApiError(400, "invalid_date", "Tahmin tarihi geçerli bir tarih değil.") from None
    if not date(2020, 1, 1) <= day <= date.today() + timedelta(days=366):
        raise ApiError(400, "invalid_date", "Tahmin tarihi makul aralıkta değil.")
    return day.isoformat()


def _string_list(value, name: str, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        raise ApiError(400, "invalid_filters", f"{name} bir liste olmalı.")
    if len(value) > max_items:
        raise ApiError(400, "invalid_filters", f"{name} en fazla {max_items} değer içerebilir.")
    out = []
    for v in value:
        text = _text(v, name, max_chars)
        if text is None:
            raise ApiError(400, "invalid_filters", f"{name} boş değer içeremez.")
        out.append(text)
    return out


def validate_filters(value) -> dict:
    """Shape / size checks; Phase-7 parse_filters then checks the meaning."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(400, "invalid_filters", "Filtreler bir nesne olmalı.")
    _keys(value, set(FL.RULE_FIELDS), "Filtreler")
    out = {}
    if "top_n" in value:
        out["top_n"] = _number(value["top_n"], "Liste uzunluğu", integer=True, low=1,
                               high=FL.MAX_TOP_N)
    if "exclude_firm_keys" in value:
        out["exclude_firm_keys"] = _string_list(value["exclude_firm_keys"], "Dışlanan firmalar",
                                                MAX_EXCLUDE_KEYS, MAX_FIRMA_KEY_CHARS)
    if "firm_cities" in value:
        out["firm_cities"] = _string_list(value["firm_cities"], "Firma şehirleri", MAX_CITIES,
                                          MAX_CITY_CHARS)
    if "active_365d_only" in value:
        if not isinstance(value["active_365d_only"], bool):
            raise ApiError(400, "invalid_filters", "Son 365 gün filtresi true/false olmalı.")
        out["active_365d_only"] = value["active_365d_only"]
    try:
        FL.parse_filters(out)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "invalid_filters", f"Filtre kabul edilmedi: {exc}") from None
    return out


# ----------------------------------------------------------------------------
# Browser projection and export.
# ----------------------------------------------------------------------------
def _view_model_a(a: dict) -> dict:
    if a.get("status") != "ok":
        return {k: a.get(k) for k in ("status", "reason", "message", "error_id")}
    calls = [{"rank": e["rank"], "rank_original": e["rank_original"],
              "firma_key": e["firma_key"], "firma_adi": e["firma_adi"],
              "directory": {k: e["directory"].get(k)
                            for k in ("city", "district", "detay_tip", "tip", "in_directory")},
              "history": e["history"], "snippets": e["snippets"]}
             for e in a["call_list"]]
    ahead = a["excluded_ahead"]
    return {"status": "ok", "score_note": a["score_note"],
            "explanation_note": a["explanation_note"], "contact_note": a["contact_note"],
            "directory_note": a["directory_note"], "call_list": calls,
            "excluded_ahead": ahead[:MAX_EXCLUDED_VIEW], "n_excluded_ahead": len(ahead),
            "filter_summary": a["filter_summary"], "warnings": a["warnings"]}


def _view_model_b(b: dict) -> dict:
    out = dict(b)
    sup = b.get("history_support")
    if sup:
        out["history_support_tr"] = [
            {"level": lvl, "label": label, "n": sup.get(f"{lvl}_n")}
            for lvl, label in HISTORY_LEVEL_TR.items() if f"{lvl}_n" in sup]
    return out


def client_view(response: dict, *, parent_id: str | None, serving_code: str,
                feedback: list, stored: bool) -> dict:
    """Display-only projection of a stored Phase-7 response (see module docstring)."""
    m = response["bundle"]
    vs = response["vehicle_support"]
    replay = response["mode"] == "historical_replay"
    return {
        "prediction_id": response["prediction_id"],
        "parent_id": parent_id,
        "stored": stored,
        "created_at_utc": response["created_at_utc"],
        "status": response["status"],
        "mode": response["mode"],
        "prediction_date": response["prediction_date"],
        "vehicle": response["vehicle"],
        "vehicle_input": None if replay else response["vehicle_input"],
        "replay_note": PR.REPLAY_NOTE if replay else None,
        "recorded_outcome_note": ("Kayıtlı satış sonucu bu ekranda gösterilmez; geçmiş örnek "
                                  "yalnızca çalışma biçimini gösterir.") if replay else None,
        "support": {"missing": vs["missing"], "min_fiyat": vs["min_fiyat"],
                    "inspection": {k: vs["inspection"].get(k)
                                   for k in ("n_hasar_compound_zero", "n_ekspertiz_missing",
                                             "note")},
                    "seller": {k: vs["seller"].get(k)
                               for k in ("satici_tipi", "n_seller_missing", "note")},
                    "wawasaky": vs["wawasaky"]["n"]},
        "limited_support": response["limited_support"],
        "warnings": response["warnings"],
        "data_recency": response["data_recency"],
        "filters": response["filters"],
        "model_a": _view_model_a(response["model_a"]),
        "model_b": _view_model_b(response["model_b"]),
        "identity": {"bundle_id": m["bundle_id"],
                     "dataset_version_id": m["dataset_version_id"],
                     "bundle_loaded_at_utc": m["loaded_at_utc"],
                     "data_ranges": m["data_ranges"], "fx": m["fx"], "models": m["models"],
                     "components": {k: m["components"][k] for k in ("model_a", "model_b")},
                     "filters_version": response["filters"]["version"],
                     "rules_sha256": response["filters"]["rules_sha256"],
                     "serving_code_sha256": serving_code},
        "feedback": feedback,
    }


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}".replace(".", ",")
    text = str(value)
    if text[:1] in _CSV_RISKY or text.lstrip()[:1] in _CSV_RISKY:
        text = "'" + text                    # spreadsheet formula injection guard
    return text


CSV_HEADER = ("Sıra", "Model sırası", "Firma", "Firma anahtarı", "Şehir (güncel dizin)",
              "İlçe", "Firma tipi", "Kayıtlı teklif (tahmin tarihinden önce)",
              "Son teklif (gün önce)", "Son katılım (gün önce)", "Geçmiş özeti",
              "Tahmin tarihi", "Tahmin no", "Filtre kuralları sha256")


def call_list_csv(response: dict) -> bytes:
    """The shown call list as Excel-friendly CSV (UTF-8 BOM, ';' separator)."""
    a = response["model_a"]
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    writer.writerow(CSV_HEADER)
    for e in a.get("call_list", []) if a.get("status") == "ok" else []:
        d, h = e["directory"], e["history"]
        writer.writerow([_csv_cell(v) for v in (
            e["rank"], e["rank_original"], e["firma_adi"] or e["firma_key"], e["firma_key"],
            d.get("city"), d.get("district"), d.get("detay_tip") or d.get("tip"),
            h.get("total_bids_prior"), h.get("days_since_last_bid"),
            h.get("days_since_last_participation"), " ".join(e["snippets"]),
            response["prediction_date"], response["prediction_id"],
            response["filters"]["rules_sha256"])])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


# ----------------------------------------------------------------------------
# Bundle-derived options and historical examples (computed once per bundle).
# ----------------------------------------------------------------------------
def _nonblank(series: pd.Series) -> np.ndarray:
    s = series.astype("string")
    return (s.notna() & s.fillna("").str.strip().ne("")).to_numpy(dtype=bool)


def _top_values(series: pd.Series, limit: int) -> list[str]:
    s = series.astype("string").str.strip()
    counts = s[_nonblank(s)].value_counts()
    counts = counts[counts >= MIN_CATEGORY_COUNT].head(limit)
    return sorted((str(v) for v in counts.index), key=str.casefold)


def build_options(bundle: BD.Bundle) -> dict:
    """Bounded category suggestions observed in history (frequent values only)."""
    cs = bundle.clean_sales
    suggestions = {f: _top_values(cs[f], k) for f, k in SUGGEST_LIMITS.items()}
    marka = cs["marka"].astype("string").str.strip()
    model = cs["model"].astype("string").str.strip()
    ok = _nonblank(marka) & _nonblank(model)
    pairs = pd.DataFrame({"marka": marka[ok].astype(str), "model": model[ok].astype(str)})
    known = set(suggestions["marka"])
    models: dict[str, list[str]] = {}
    for (mk, md), n in pairs.value_counts().items():
        if n >= MIN_CATEGORY_COUNT and mk in known:
            lst = models.setdefault(mk, [])
            if len(lst) < MODELS_PER_MARKA:
                lst.append(md)
    cities: list[str] = []
    if bundle.model_a is not None:
        c = bundle.model_a.firms["sehir"].astype("string").str.strip()
        cities = sorted({str(v) for v in c[_nonblank(c)]}, key=str.casefold)
    return {"suggestions": suggestions,
            "models_by_marka": {k: sorted(v, key=str.casefold) for k, v in models.items()},
            "firm_cities": cities,
            "top_n": {"default": FL.DEFAULT_TOP_N, "max": FL.MAX_TOP_N},
            "text_fields": TEXT_FIELDS,
            "number_fields": {k: {"integer": i, "min": lo, "max": hi}
                              for k, (i, lo, hi) in NUMBER_FIELDS.items()},
            "firm_reasons": FIRM_REASONS}


def _display(v) -> str | None:
    if v is None or v is pd.NA or (isinstance(v, float) and not math.isfinite(v)):
        return None
    text = str(v).strip()
    return text or None


def build_examples(bundle: BD.Bundle) -> tuple[list[dict], dict]:
    """(display list, example_id -> query_id). Recent recorded vehicles that from_history
    can replay unambiguously; the plate / query id never leaves the server."""
    cs = bundle.clean_sales
    day = pd.to_datetime(cs["aa_tarihi"], errors="coerce")
    qid = cs["plaka"].astype("string") + KEY_SEPARATOR + cs["aa_kodu"].astype("string")
    ok = (day.notna().to_numpy(dtype=bool) & _nonblank(cs["plaka"]) & _nonblank(cs["aa_kodu"])
          & _nonblank(cs["marka"]) & _nonblank(cs["model"])
          & pd.to_numeric(cs["model_yili"], errors="coerce").notna().to_numpy(dtype=bool)
          & pd.to_numeric(cs["aa_tarihi_invalid_flag"], errors="coerce").fillna(1).eq(0)
          .to_numpy(dtype=bool))
    # Uniqueness over ALL key-complete rows, as from_history counts it (not only valid ones).
    keyed = _nonblank(cs["plaka"]) & _nonblank(cs["aa_kodu"])
    keyed_q = qid.where(keyed)
    counts = keyed_q.dropna().value_counts()
    ok &= pd.to_numeric(keyed_q.map(counts), errors="coerce").fillna(0).eq(1) \
        .to_numpy(dtype=bool)
    if not ok.any():
        return [], {}
    last = day[ok].max()
    ok &= (day >= last - pd.Timedelta(days=EXAMPLE_DAYS)).to_numpy(dtype=bool)
    pos = np.flatnonzero(ok)
    rng = np.random.default_rng(EXAMPLE_SEED)
    pick = rng.choice(pos, size=min(EXAMPLE_COUNT, len(pos)), replace=False)
    pick = sorted(pick.tolist(), key=lambda p: (-day.iloc[p].value, p))
    items, index = [], {}
    for p in pick:
        q = str(qid.iloc[p])
        eid = hashlib.sha256(f"{bundle.bundle_id}|{q}".encode("utf-8")).hexdigest()[:20]
        r = cs.iloc[p]
        km = pd.to_numeric(pd.Series([r["km_clean"]]), errors="coerce").iloc[0]
        items.append({"example_id": eid, "date": str(day.iloc[p].date()),
                      "marka": _display(r["marka"]), "model": _display(r["model"]),
                      "model_uzantisi": _display(r["model_uzantisi"]),
                      "model_yili": int(pd.to_numeric(r["model_yili"])),
                      "km_rounded": None if pd.isna(km) else int(round(float(km), -3)),
                      "yakit_tipi": _display(r["yakit_tipi"]),
                      "vites_tipi": _display(r["vites_tipi"])})
        index[eid] = q
    return items, index


# ----------------------------------------------------------------------------
# Runtime state.
# ----------------------------------------------------------------------------
class DemoRuntime:
    """Bundle state, prediction lock and the store. One instance per app."""

    def __init__(self, settings: DemoSettings, store: Store, serving_code: str):
        self.settings, self.store, self.serving_code = settings, store, serving_code
        self._state_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self.state, self.message = "loading", "Modeller ve geçmiş veriler yükleniyor."
        self.bundle: BD.Bundle | None = None
        self.options: dict | None = None
        self.examples: list[dict] = []
        self.example_index: dict[str, str] = {}
        self.started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def load(self) -> None:
        try:
            bundle = BD.load_bundle(BD.BundleConfig(root=self.settings.root))
            options = build_options(bundle)
            examples, index = build_examples(bundle)
        except BD.BundleError as exc:
            log.error("phase8 demo: bundle failed to load: %s", exc)
            self._set("failed", f"Tahmin paketi yüklenemedi: {exc}")
            return
        except Exception:                               # noqa: BLE001 — shown sanitized
            log.exception("phase8 demo: bundle failed to load")
            self._set("failed", "Tahmin paketi yüklenemedi; ayrıntı sunucu kaydında.")
            return
        with self._state_lock:
            self.bundle, self.options = bundle, options
            self.examples, self.example_index = examples, index
            self.state, self.message = "ready", None
        log.info("phase8 demo: bundle %s ready, serving code %s", bundle.bundle_id,
                 self.serving_code)

    def _set(self, state: str, message: str) -> None:
        with self._state_lock:
            self.state, self.message = state, message

    def require_ready(self) -> BD.Bundle:
        with self._state_lock:
            if self.state != "ready" or self.bundle is None:
                raise ApiError(503, f"bundle_{self.state}", self.message or "Hazır değil.")
            return self.bundle

    def status(self) -> dict:
        with self._state_lock:
            b = self.bundle
            out = {"state": self.state, "message": self.message,
                   "started_at_utc": self.started_at_utc,
                   "serving_code_sha256": self.serving_code,
                   "today": date.today().isoformat(), "bundle": None}
            if b is not None:
                m = b.manifest
                out["bundle"] = {"bundle_id": m["bundle_id"],
                                 "dataset_version_id": m["dataset_version_id"],
                                 "loaded_at_utc": m["loaded_at_utc"],
                                 "data_ranges": m["data_ranges"], "fx": m["fx"],
                                 "models": m["models"],
                                 "components": {k: m["components"][k]
                                                for k in ("model_a", "model_b")},
                                 "component_errors": m["components"]["errors"],
                                 "stale_after_days": m["stale_after_days"]}
            return out

    # -------------------------------------------------------------- predictions
    def run(self, owner: str, request_record: dict, *, parent_id: str | None) -> dict:
        """Compute, store and project one prediction for a validated request record."""
        bundle = self.require_ready()
        filters = request_record["filters"]
        if not self._predict_lock.acquire(timeout=PREDICT_LOCK_TIMEOUT_S):
            raise ApiError(503, "busy", "Sunucu başka bir tahmini hesaplıyor; biraz sonra "
                                        "tekrar deneyin.")
        try:
            if request_record["mode"] == "current":
                response = PR.predict_vehicle(bundle, request_record["vehicle"],
                                              prediction_date=request_record["prediction_date"],
                                              filters=filters)
            else:
                response = PR.replay_history(bundle, request_record["query_id"],
                                             filters=filters)
        except PR.InputError as exc:
            if request_record["mode"] == "historical_replay":
                # The Phase-7 text can name the plate / query id: never shown, never logged.
                error_id = uuid.uuid4().hex[:12]
                log.warning("phase8 demo: historical example rejected at replay_history "
                            "(error %s)", error_id)
                raise ApiError(400, "example_invalid", "Bu geçmiş örnek kullanılamıyor; "
                                                       "lütfen başka bir örnek seçin (kayıt no "
                                                       f"{error_id}).") from None
            raise ApiError(400, "invalid_input", f"Girdi kabul edilmedi: {exc}") from None
        finally:
            self._predict_lock.release()

        stored = True
        try:
            self.store.save_prediction(PredictionRecord(
                prediction_id=response["prediction_id"], owner=owner,
                created_at_utc=response["created_at_utc"], parent_id=parent_id,
                mode=response["mode"], status=response["status"],
                prediction_date=response["prediction_date"],
                title=request_record["title"], bundle_id=response["bundle"]["bundle_id"],
                filters_version=response["filters"]["version"],
                rules_sha256=response["filters"]["rules_sha256"],
                serving_code_sha256=self.serving_code,
                request_json=json.dumps(request_record, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False),
                response_json=PR.response_json(response)))
        except Exception:                               # noqa: BLE001 — result still shown
            log.exception("phase8 demo: prediction %s could not be stored",
                          response.get("prediction_id"))
            stored = False
        return client_view(response, parent_id=parent_id, serving_code=self.serving_code,
                           feedback=[], stored=stored)

    def current_request(self, vehicle: dict, prediction_date: str, filters: dict) -> dict:
        title = " ".join(str(vehicle[k]) for k in ("Marka", "Model", "Model Yılı")
                         if k in vehicle)
        return {"mode": "current", "vehicle": vehicle, "prediction_date": prediction_date,
                "filters": filters, "title": title[:200]}

    def replay_request(self, example_id: str, filters: dict) -> dict:
        self.require_ready()
        with self._state_lock:
            query_id = self.example_index.get(example_id)
            item = next((e for e in self.examples if e["example_id"] == example_id), None)
        if query_id is None or item is None:
            raise ApiError(404, "example_not_found", "Geçmiş örnek bulunamadı; listeyi "
                                                     "yenileyin.")
        title = "Geçmiş örnek · " + " ".join(
            str(item[k]) for k in ("marka", "model", "model_yili") if item.get(k) is not None)
        return {"mode": "historical_replay", "example_id": example_id, "query_id": query_id,
                "filters": filters, "title": title[:200]}


# ----------------------------------------------------------------------------
# Application.
# ----------------------------------------------------------------------------
def _load_pages() -> dict:
    pages = {}
    for route, (name, ctype) in PAGES.items():
        pages[route] = ((_PKG / "static" / name).read_bytes(), ctype)
    return pages


def create_app(settings: DemoSettings, *, on_shutdown=None, mailer=None):
    """The ASGI application (FastAPI wrapped in GuardMiddleware). See module docstring.
    on_shutdown: optional zero-argument callable that stops the server (local runs only).
    mailer: optional object with send(to, subject, body) (mailer.py); with it and
    settings.email the password reset is available."""
    if not isinstance(settings, DemoSettings):
        raise TypeError("settings must be DemoSettings")
    settings.validate()
    if set(TEXT_FIELDS) | set(NUMBER_FIELDS) != set(RAW_COLUMNS) \
            or set(TEXT_FIELDS) & set(NUMBER_FIELDS):
        raise ConfigError("Phase-8 vehicle field rules do not match vehicle_input.RAW_COLUMNS")
    pages = _load_pages()
    serving = serving_code_sha256()
    store = Store(settings.store_path)
    store.initialize()
    rt = DemoRuntime(settings, store, serving)
    sessions = SEC.SessionStore(settings.session_hours * 3600)
    throttle = SEC.LoginThrottle()
    account = Account(settings.username, settings.email, settings.password_hash, store)
    resets = SEC.PasswordReset()
    reset_enabled = mailer is not None and bool(settings.email)
    reset_requests = SEC.LoginThrottle(max_failures=5, window_seconds=900.0)
    reset_failures = SEC.LoginThrottle(max_failures=10, window_seconds=900.0)
    password_slots = threading.BoundedSemaphore(LOGIN_CONCURRENCY)
    cookie =SEC.COOKIE_SECURE if settings.secure_cookies else SEC.COOKIE_LOCAL

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=rt.load, name="phase7-bundle-loader", daemon=True).start()
        yield

    api = FastAPI(title="Garaj demo", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)

    @api.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}},
                            status_code=exc.status, headers=exc.headers)

    @api.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception):
        error_id = uuid.uuid4().hex[:12]
        log.error("phase8 demo: unhandled error %s", error_id, exc_info=exc)
        return JSONResponse({"error": {"code": "server_error",
                                       "message": f"Beklenmeyen sunucu hatası (kayıt no "
                                                  f"{error_id})."}}, status_code=500)

    # ------------------------------------------------------------ helpers
    def session_of(request: Request, *, mutate: bool) -> SEC.Session:
        session = sessions.get(request.cookies.get(cookie))
        if session is None:
            raise ApiError(401, "auth_required", "Oturum açmanız gerekiyor.")
        if mutate:
            check_origin(request)
            if not SEC.csrf_ok(session, request.headers.get(SEC.CSRF_HEADER)):
                raise ApiError(403, "csrf", "Güvenlik belirteci geçersiz; sayfayı yenileyin.")
        return session

    def client_of(request: Request) -> str:
        return SEC.client_ip(request.client.host if request.client else None,
                             request.headers.get("x-forwarded-for"), settings.trusted_proxies)

    def refuse_if_throttled(limiter: SEC.LoginThrottle, client: str) -> None:
        wait = limiter.retry_after(client)
        if wait:
            raise ApiError(429, "too_many_attempts",
                           f"Çok fazla hatalı deneme; {math.ceil(wait / 60)} dakika sonra "
                           f"tekrar deneyin.", headers={"Retry-After": str(wait)})

    def send_mail_later(subject: str, body: str) -> None:
        """Send in the background: the answer must not reveal (by error or by timing)
        whether an e-mail went out. Failures are only logged."""
        def run():
            try:
                mailer.send(settings.email, subject, body)
            except Exception as exc:                    # noqa: BLE001 — logged, not shown
                log.error("phase8 demo: e-mail could not be sent (%s: %s)",
                          type(exc).__name__, exc)
        threading.Thread(target=run, name="phase8-mail", daemon=True).start()

    def check_origin(request: Request) -> None:
        if not SEC.origin_allowed(request.headers.get("origin"), settings.allowed_hosts):
            raise ApiError(403, "origin", "İstek kaynağı kabul edilmedi.")

    async def json_body(request: Request) -> dict:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError(415, "json_required", "İstek JSON olarak gönderilmeli.")
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise ApiError(413, "too_large", "İstek çok büyük.")
        try:
            data = json.loads(raw.decode("utf-8") or "{}", parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise ApiError(400, "invalid_json", "Geçersiz JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "invalid_json", "İstek bir JSON nesnesi olmalı.")
        return data

    def owned(prediction_id: str, owner: str) -> dict:
        if not _PID_RE.fullmatch(prediction_id or ""):
            raise ApiError(404, "not_found", "Tahmin bulunamadı.")
        record = store.get_prediction(prediction_id, owner)
        if record is None:
            raise ApiError(404, "not_found", "Tahmin bulunamadı.")
        return record

    def verify_login(username, password) -> bool | None:
        """check_credentials under a password slot; None = no slot free (scrypt not run).
        Runs in the worker thread: acquire and release both happen here, so a cancelled
        request cannot free the slot while scrypt is still computing."""
        if not password_slots.acquire(blocking=False):
            return None
        try:
            return account.check(username, password)
        finally:
            password_slots.release()

    def view_of(record: dict, owner: str) -> dict:
        return client_view(record["response"], parent_id=record["parent_id"],
                           serving_code=record["serving_code_sha256"],
                           feedback=store.feedback_for(record["prediction_id"], owner),
                           stored=True)

    # ------------------------------------------------------------ pages
    @api.get("/", include_in_schema=False)
    def index():
        body, ctype = pages["/"]
        return Response(body, media_type=ctype)

    @api.get("/static/{name}", include_in_schema=False)
    def static(name: str):
        page = pages.get(f"/static/{name}")
        if page is None:
            raise ApiError(404, "not_found", "Bulunamadı.")
        return Response(page[0], media_type=page[1])

    # ------------------------------------------------------------ session
    @api.get("/api/session")
    def get_session(request: Request):
        session = sessions.get(request.cookies.get(cookie))
        if session is None:
            return {"authenticated": False, "reset_enabled": reset_enabled}
        return {"authenticated": True, "username": session.username,
                "csrf_token": session.csrf_token, "can_shutdown": on_shutdown is not None}

    @api.post("/api/login")
    async def login(request: Request):
        check_origin(request)
        body = await json_body(request)
        _keys(body, {"username", "password"}, "Giriş")
        client = client_of(request)
        refuse_if_throttled(throttle, client)
        ok = await run_in_threadpool(verify_login, body.get("username"), body.get("password"))
        if ok is None:
            raise ApiError(429, "login_busy", "Şu anda çok sayıda giriş denemesi var; birkaç "
                                              "saniye sonra tekrar deneyin.",
                           headers={"Retry-After": str(LOGIN_BUSY_RETRY_S)})
        if not ok:
            throttle.failure(client)
            log.warning("phase8 demo: failed login from %s", client)
            raise ApiError(401, "invalid_credentials", "Kullanıcı adı veya parola hatalı.")
        throttle.success(client)
        sessions.revoke(request.cookies.get(cookie))
        token, session = sessions.create(settings.username)
        log.info("phase8 demo: login %s from %s", session.username, client)
        resp = JSONResponse({"authenticated": True, "username": session.username,
                             "csrf_token": session.csrf_token,
                             "can_shutdown": on_shutdown is not None})
        resp.set_cookie(cookie, token, max_age=int(sessions.ttl_seconds), path="/",
                        secure=settings.secure_cookies, httponly=True, samesite="strict")
        return resp

    # ------------------------------------------------------------ password reset
    def require_reset_enabled() -> None:
        if not reset_enabled:
            raise ApiError(503, "reset_disabled", "Parola sıfırlama bu kurulumda kapalı: "
                                                  "e-posta ayarları yapılmamış. Yöneticinize "
                                                  "başvurun.")

    @api.post("/api/password-reset/request")
    async def reset_request(request: Request):
        check_origin(request)
        body = await json_body(request)
        _keys(body, {"identifier"}, "Parola sıfırlama")
        require_reset_enabled()
        client = client_of(request)
        refuse_if_throttled(reset_requests, client)
        reset_requests.failure(client)                  # every request counts
        if account.matches(body.get("identifier")):
            code = resets.issue()
            if code is not None:
                log.info("phase8 demo: password-reset code issued (client %s)", client)
                send_mail_later(
                    "Garaj parola sıfırlama kodu",
                    f"Merhaba,\n\nGaraj çalışma alanı parola sıfırlama kodunuz: {code}\n\n"
                    "Kod 10 dakika geçerlidir ve yalnızca bir kez kullanılabilir.\n"
                    "Bu isteği siz yapmadıysanız bu e-postayı dikkate almayın; parolanız "
                    "değişmez.\n")
        else:
            log.info("phase8 demo: password-reset request for an unknown identifier "
                     "(client %s)", client)
        return {"sent": True, "message": RESET_SENT_TR}

    @api.post("/api/password-reset/verify")
    async def reset_verify(request: Request):
        check_origin(request)
        body = await json_body(request)
        _keys(body, {"code"}, "Kod doğrulama")
        require_reset_enabled()
        client = client_of(request)
        refuse_if_throttled(reset_failures, client)
        code = body.get("code")
        token = resets.verify(code.strip() if isinstance(code, str) else code)
        if token is None:
            reset_failures.failure(client)
            log.warning("phase8 demo: wrong or expired reset code from %s", client)
            raise ApiError(400, "invalid_code", "Kod hatalı ya da süresi dolmuş. Tekrar "
                                                "deneyin veya yeni bir kod isteyin.")
        return {"reset_token": token}

    @api.post("/api/password-reset/complete")
    async def reset_complete(request: Request):
        check_origin(request)
        body = await json_body(request)
        _keys(body, {"reset_token", "password", "password_confirm"}, "Yeni parola")
        require_reset_enabled()
        client = client_of(request)
        password, confirm = body.get("password"), body.get("password_confirm")
        if not isinstance(password, str) or not isinstance(confirm, str):
            raise ApiError(400, "invalid_field", "Yeni parola iki kez yazılmalı.")
        if password != confirm:
            raise ApiError(400, "password_mismatch", "Yazdığınız iki parola aynı değil.")
        missing = SEC.password_problems(password)
        if missing:
            raise ApiError(400, "weak_password", f"{SEC.PASSWORD_RULES_TR} Eksik: "
                                                 f"{', '.join(missing)}.")
        if not resets.consume_token(body.get("reset_token")):
            raise ApiError(400, "reset_expired", "Parola sıfırlama süresi doldu. Lütfen "
                                                 "yeni bir kod isteyin.")
        await run_in_threadpool(account.set_password, password)
        sessions.revoke_all()
        throttle.success(client)
        log.info("phase8 demo: password changed by reset (client %s)", client)
        send_mail_later("Garaj parolanız değiştirildi",
                        "Merhaba,\n\nGaraj çalışma alanı parolanız az önce değiştirildi. "
                        "Açık olan tüm oturumlar kapatıldı.\n\nBu değişikliği siz "
                        "yapmadıysanız hemen yöneticinize haber verin.\n")
        resp = JSONResponse({"changed": True})
        resp.delete_cookie(cookie, path="/", secure=settings.secure_cookies, httponly=True,
                           samesite="strict")
        return resp

    @api.post("/api/logout")
    async def logout(request: Request):
        session_of(request, mutate=True)
        sessions.revoke(request.cookies.get(cookie))
        resp = JSONResponse({"authenticated": False})
        resp.delete_cookie(cookie, path="/", secure=settings.secure_cookies, httponly=True,
                           samesite="strict")
        return resp

    @api.post("/api/shutdown")
    async def shutdown(request: Request):
        if on_shutdown is None:
            raise ApiError(404, "not_found", "Bulunamadı.")
        session = session_of(request, mutate=True)
        sessions.revoke(request.cookies.get(cookie))
        log.info("phase8 demo: shutdown requested by %s", session.username)
        # Stop slightly later so this response still reaches the browser.
        threading.Timer(0.5, on_shutdown).start()
        resp = JSONResponse({"stopping": True})
        resp.delete_cookie(cookie, path="/", secure=settings.secure_cookies, httponly=True,
                           samesite="strict")
        return resp

    # ------------------------------------------------------------ reference data
    @api.get("/api/status")
    def status(request: Request):
        session_of(request, mutate=False)
        return rt.status()

    @api.get("/api/options")
    def options(request: Request):
        session_of(request, mutate=False)
        rt.require_ready()
        return rt.options

    @api.get("/api/examples")
    def examples(request: Request):
        session_of(request, mutate=False)
        rt.require_ready()
        return {"items": rt.examples, "note": PR.REPLAY_NOTE}

    # ------------------------------------------------------------ predictions
    @api.post("/api/predict")
    async def predict(request: Request):
        session = session_of(request, mutate=True)
        body = await json_body(request)
        _keys(body, {"vehicle", "prediction_date", "filters"}, "İstek")
        record = rt.current_request(validate_vehicle(body.get("vehicle")),
                                    validate_date(body.get("prediction_date")),
                                    validate_filters(body.get("filters")))
        return await run_in_threadpool(rt.run, session.username, record, parent_id=None)

    @api.post("/api/replay")
    async def replay(request: Request):
        session = session_of(request, mutate=True)
        body = await json_body(request)
        _keys(body, {"example_id", "filters"}, "İstek")
        example_id = body.get("example_id")
        if not isinstance(example_id, str) or not _EXAMPLE_RE.fullmatch(example_id):
            raise ApiError(400, "invalid_example", "Geçmiş örnek seçilmedi.")
        record = rt.replay_request(example_id, validate_filters(body.get("filters")))
        return await run_in_threadpool(rt.run, session.username, record, parent_id=None)

    @api.post("/api/predictions/{prediction_id}/rerun")
    async def rerun(prediction_id: str, request: Request):
        session = session_of(request, mutate=True)
        body = await json_body(request)
        _keys(body, {"filters"}, "İstek")
        filters = validate_filters(body.get("filters"))
        parent = await run_in_threadpool(owned, prediction_id, session.username)
        old = parent["request"]
        if old["mode"] == "current":
            record = rt.current_request(validate_vehicle(old["vehicle"]),
                                        validate_date(old["prediction_date"]), filters)
        else:
            rt.require_ready()
            record = {**old, "filters": filters}
            if record["query_id"] != rt.example_index.get(record["example_id"]):
                raise ApiError(409, "example_changed", "Bu geçmiş örnek mevcut tahmin "
                                                       "paketinde yok; yeniden seçin.")
        return await run_in_threadpool(rt.run, session.username, record,
                                       parent_id=parent["prediction_id"])

    @api.get("/api/predictions")
    def list_predictions(request: Request):
        session = session_of(request, mutate=False)
        return {"items": store.list_predictions(session.username, 20)}

    @api.get("/api/predictions/{prediction_id}")
    def get_prediction(prediction_id: str, request: Request):
        session = session_of(request, mutate=False)
        return view_of(owned(prediction_id, session.username), session.username)

    @api.get("/api/predictions/{prediction_id}/call-list.csv")
    def export_csv(prediction_id: str, request: Request):
        session = session_of(request, mutate=False)
        record = owned(prediction_id, session.username)
        name = f"arama_listesi_{record['prediction_date']}_{prediction_id[:8]}.csv"
        return Response(call_list_csv(record["response"]), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @api.post("/api/predictions/{prediction_id}/feedback")
    async def feedback(prediction_id: str, request: Request):
        session = session_of(request, mutate=True)
        body = await json_body(request)
        kind = body.get("kind")
        if kind == "firm":
            _keys(body, {"kind", "firma_key", "verdict", "reason_code", "comment"},
                  "Geri bildirim")
            if body.get("verdict") not in ("uygun", "uygun_degil"):
                raise ApiError(400, "invalid_feedback", "Firma değerlendirmesi seçilmedi.")
            reason = body.get("reason_code")
            if reason is not None and reason not in FIRM_REASONS:
                raise ApiError(400, "invalid_feedback", "Gerekçe tanınmıyor.")
            firma_key = _text(body.get("firma_key"), "Firma", MAX_FIRMA_KEY_CHARS)
        elif kind == "price":
            _keys(body, {"kind", "verdict", "suggested_price_tl", "comment"}, "Geri bildirim")
            if body.get("verdict") not in ("dusuk", "makul", "yuksek"):
                raise ApiError(400, "invalid_feedback", "Fiyat değerlendirmesi seçilmedi.")
            suggested = body.get("suggested_price_tl")
            if suggested is not None:
                suggested = float(_number(suggested, "Önerilen fiyat", integer=False, low=1,
                                          high=NUMBER_FIELDS["Min.Fiyat"][2]))
        else:
            raise ApiError(400, "invalid_feedback", "Geri bildirim türü 'firm' veya 'price' "
                                                    "olmalı.")
        comment = _text(body.get("comment"), "Not", MAX_COMMENT_CHARS, multiline=True)

        def save():
            record = owned(prediction_id, session.username)
            response = record["response"]
            try:
                if kind == "firm":
                    shown = {e["firma_key"]: e for e in response["model_a"].get("call_list", [])} \
                        if response["model_a"].get("status") == "ok" else {}
                    entry = shown.get(firma_key)
                    if entry is None:
                        raise ApiError(400, "firm_not_shown", "Bu firma bu tahminin arama "
                                                              "listesinde yok.")
                    return store.add_feedback(prediction_id, session.username, kind="firm",
                                              verdict=body["verdict"], firma_key=firma_key,
                                              firma_adi=entry["firma_adi"],
                                              firm_rank=entry["rank"], reason_code=reason,
                                              comment=comment)
                b = response["model_b"]
                if b.get("status") != "ok":
                    raise ApiError(400, "price_not_shown", "Bu tahminde fiyat gösterilmedi.")
                return store.add_feedback(prediction_id, session.username, kind="price",
                                          verdict=body["verdict"], suggested_price_tl=suggested,
                                          comment=comment, shown_price_tl=b["price_tl"])
            except StoreError as exc:
                raise ApiError(400, "feedback_rejected", f"Geri bildirim kaydedilemedi: "
                                                         f"{exc}") from None

        return {"feedback": await run_in_threadpool(save)}

    return SEC.GuardMiddleware(api, allowed_hosts=settings.allowed_hosts,
                               secure=settings.secure_cookies, max_body_bytes=MAX_BODY_BYTES)

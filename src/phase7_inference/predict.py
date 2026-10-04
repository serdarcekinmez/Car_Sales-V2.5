"""
predict.py  —  Phase 7: ONE inference entry point for the customer demo.

For one vehicle and one prediction date it returns, in a single JSON-safe response:
  model_a  "Kimi arayalım?" — the frozen ranker's call list with real firm names after the
           versioned eligibility filters (filters.py), original and filtered ranks,
           exclusion reasons and factual history snippets. The score is a ranking score,
           never a purchase probability.
  model_b  "Tahmini ihale satış fiyatı" — the frozen lgbm_noR estimate in TL, conditional
           on a sale within the data's auction scope; no interval (none is valid for it).
The two outputs are never combined. Each model runs independently: an unavailable or
failing model is reported with its own status and never erases the other's result. A
per-model error is logged server-side with an error_id; the response carries only a
sanitized message (no traceback, no server path).

MODES
  current            predict_vehicle: a vehicle that has not been auctioned (no sale price,
                     buyer or outcome needed). Recency is judged PER SOURCE (sales, bids,
                     participation each end on their own day; bundle.MODEL_SOURCES says
                     which model reads which). A source whose last day is more than
                     bundle.config.stale_after_days before the prediction date is stale;
                     each model warns only about its own stale sources, quoting each
                     source's actual last day. Records after a source's last day are
                     invisible, and low activity indicators then reflect missing data, not
                     firm inactivity.
  historical_replay  replay_history: a recorded vehicle re-scored at its own date from its
                     original clean_sales position. A demonstration only: the frozen models
                     may have been trained on, or evaluated over, that date. It is never an
                     out-of-time validation and never untouched D6 data. The recorded
                     outcome is returned separately and is not a model input; its
                     in_price_population is real membership of the Phase-5 price
                     population built once from the same clean_sales (None when Model B is
                     unavailable), not a membership of any evaluation period.

INPUT (current): a mapping of vehicle_input.RAW_COLUMNS (exact Turkish export names).
Every ORIGINAL key is checked against that allow-list first, blank values included, so an
outcome or unknown key raises even when its value is None. Marka is required; None / blank
values then count as missing. Obvious invalid values (negative KM, Tramer or Min.Fiyat,
impossible model year, out-of-range Eks.Puanı) raise InputError before any model runs.
Min.Fiyat may be given but the primary price model does not use it.

PUBLIC API
  predict_vehicle(bundle, vehicle, *, prediction_date, filters=None, request_id=None) -> dict
  replay_history(bundle, query_id, *, filters=None, request_id=None)                  -> dict
  response_json(response)                                                             -> str
  InputError
"""
from __future__ import annotations

import json
import logging
import numbers
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from phase4_modeling.query_universe import SPLIT_BOUNDARY
from phase7_inference import filters as FL
from phase7_inference import model_a as MA
from phase7_inference import model_b as MB
from phase7_inference import vehicle_input as VI
from phase7_inference.bundle import MODEL_SOURCES, Bundle

log = logging.getLogger(__name__)

SCHEMA = "phase7-prediction-v1"
_ID_RE = re.compile(r"[A-Za-z0-9]{1,64}")
_SUMMARY_FIELDS = ["marka", "model", "model_uzantisi", "model_yili", "car_age", "km_clean",
                   "eks_puani_clean", "hasar_compound", "tramer_adedi", "tramer_tutari",
                   "min_fiyat_clean", "satici_tipi", "yakit_tipi", "vites_tipi", "kasa_tipi",
                   "renk", "lokasyon"]
_RANKER_CATEGORIES = {"veh_marka": "marka", "veh_kasa_tipi": "kasa_tipi",
                      "veh_lokasyon": "lokasyon"}
_SOURCE_TR = {"sales": "satış", "bids": "teklif", "participation": "katılım"}

REPLAY_NOTE = ("Geçmiş örnek gösterimi: kayıtlı araç kendi tarihinde yeniden puanlandı. "
               "Dondurulmuş modeller bu tarihten sonra eğitilmiş veya bu dönem daha önce "
               "incelenmiş olabilir; örneklem dışı doğrulama değildir ve D6 verisi değildir.")
EXPLANATION_NOTE = ("Geçmiş bilgileri tahmin tarihinden önceki kayıtlardan alınmıştır; "
                    "modelin karar gerekçesi veya satın alma garantisi değildir.")
CONTACT_NOTE = "Kaynak firma dizininde telefon / e-posta alanı yok; iletişim bilgisi gösterilmez."
DIRECTORY_NOTE = ("Şehir, ilçe ve firma tipi güncel üye dizini kaydıdır; geçmiş bir tarihteki "
                  "firma bilgisinin kanıtı değildir.")


class InputError(ValueError):
    """The request is invalid; no model was run. The message is safe to show."""


# ----------------------------------------------------------------------------
# Small helpers.
# ----------------------------------------------------------------------------
def _clean(obj):
    """JSON-safe copy: numpy / pandas scalars -> Python, NaN / NA -> None, dates -> ISO."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp):
        return str(obj.date()) if obj == obj.normalize() else obj.isoformat()
    return obj


def _w(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _num(v) -> float:
    v = pd.to_numeric(pd.Series([v]), errors="coerce").astype("float64").iloc[0]
    return float(v)


def _blank(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) \
        or (isinstance(v, str) and not v.strip())


def _check_request_id(request_id) -> str:
    if not isinstance(request_id, str) or not _ID_RE.fullmatch(request_id):
        raise InputError("request_id must be 1-64 ASCII letters / digits")
    return request_id


def _model_error(model: str, pid: str, message: str) -> dict:
    error_id = uuid.uuid4().hex[:12]
    log.exception("phase7 prediction %s: %s failed (error_id=%s)", pid, model, error_id)
    return {"status": "error", "error_id": error_id, "message": message}


# ----------------------------------------------------------------------------
# Input.
# ----------------------------------------------------------------------------
def _raw_frame(vehicle) -> tuple[pd.DataFrame, dict]:
    """One-row raw frame of the supplied, non-blank fields, and a JSON echo of the input."""
    if not isinstance(vehicle, Mapping):
        raise InputError("vehicle must be a mapping of field name -> value")
    keys = list(vehicle.keys())
    if not all(isinstance(k, str) for k in keys):
        raise InputError("vehicle field names must be strings")
    try:                                    # every original key, before blanks are dropped
        VI._check_raw(pd.DataFrame(columns=keys))
    except (TypeError, ValueError) as exc:
        raise InputError(str(exc)) from None
    echo = {}
    kept = {}
    for k, v in vehicle.items():
        if isinstance(v, (bool, np.bool_)) or not (
                v is None or isinstance(v, (str, numbers.Number))):
            raise InputError(f"vehicle field {k!r} must be a string, a number or empty")
        echo[k] = v
        if not _blank(v):
            kept[k] = [v]
    if "Marka" not in kept:
        raise InputError("Marka is required")
    return pd.DataFrame(kept, index=pd.RangeIndex(1)), _clean(echo)


def _check_values(rows: pd.DataFrame, day: pd.Timestamp) -> None:
    """Refuse obvious invalid vehicle values after Phase-1 cleaning."""
    r = rows.iloc[0]
    problems = []
    if _blank(r["marka"]):
        problems.append("Marka is required")
    year = _num(r["model_yili"])
    if np.isfinite(year) and not 1950 <= year <= day.year + 1:
        problems.append(f"Model Yılı {year:.0f} is not a plausible model year")
    for col, name in (("km_clean", "KM"), ("tramer_adedi", "Tramer Adedi"),
                      ("tramer_tutari", "Tramer Tutarı"), ("min_fiyat_clean", "Min.Fiyat")):
        v = _num(r[col])
        if np.isfinite(v) and v < 0:
            problems.append(f"{name} must not be negative")
    if _num(r["eks_puani_out_of_range_flag"]) == 1:
        problems.append("Eks.Puanı is outside its valid range")
    if problems:
        raise InputError("; ".join(problems))


# ----------------------------------------------------------------------------
# Warnings.
# ----------------------------------------------------------------------------
def _recency(bundle: Bundle, day: pd.Timestamp, mode: str) -> dict:
    """Per-source lag behind the prediction date and per-model stale status."""
    limit = bundle.config.stale_after_days
    sources = {}
    for name, (first, last) in sorted(bundle.data_ranges.items()):
        gap = int((day - last).days)
        sources[name] = {"first_day": str(first.date()), "last_day": str(last.date()),
                         "days_after_last_day": max(gap, 0),
                         "covers_prediction_date": gap <= 0,
                         "stale": mode == "current" and gap > limit}
    models = {}
    for model, names in MODEL_SOURCES.items():
        present = [n for n in names if n in sources]
        stale = [n for n in present if sources[n]["stale"]]
        models[model] = {"sources": present, "stale_sources": stale, "stale": bool(stale)}
    return {"stale_after_days": limit, "sources": sources, "models": models,
            "stale_history": any(s["stale"] for s in sources.values()),
            "prediction_date_within_history": mode == "current" and any(
                s["covers_prediction_date"] for s in sources.values())}


def _ends(recency: dict, names) -> str:
    """'satış 2026-06-04 (122 gün önce), teklif ...' — each source's own last day."""
    return ", ".join(f"{_SOURCE_TR.get(n, n)} {recency['sources'][n]['last_day']} "
                     f"({recency['sources'][n]['days_after_last_day']} gün önce)" for n in names)


def _input_warnings(rows: pd.DataFrame, support: dict, mode: str) -> list[dict]:
    out = []
    if support["wawasaky"]["n"]:
        out.append(_w("limited_support_wawasaky",
                      "WAWASAKY işaretli araç: ürün kapsamı henüz karara bağlanmadı. İki çıktı "
                      "da sınırlı destekli tahmindir; kayıt hariç tutulmadı."))
    eks_missing = support["inspection"]["n_ekspertiz_missing"]
    if eks_missing or (eks_missing is None and support["inspection"]["n_hasar_compound_zero"]):
        out.append(_w("damage_score_zero_not_evidence",
                      "Hasar puanı 0: ekspertiz metni olmadığında da 0 üretilir; bu, aracın "
                      "hasarsız olduğunun kanıtı değildir."))
    if support["seller"]["n_seller_missing"]:
        out.append(_w("seller_type_default",
                      "Satıcı adı girilmedi: satıcı tipi 'diger' varsayılanıdır, satıcı "
                      "tipinin kanıtı değildir."))
    if mode == "current" and support["min_fiyat"]["n_present"]:
        out.append(_w("min_fiyat_not_used",
                      "Min.Fiyat girildi; ana fiyat modeli (lgbm_noR) Min.Fiyat kullanmaz."))
    return out


def _replay_warnings(day: pd.Timestamp, train_end, spent_from, model: str) -> list[dict]:
    if train_end is not None and day <= train_end:
        return [_w("replay_in_training_data",
                   f"{model} bu tarihi içeren veriyle eğitildi ({train_end.date()} sonuna "
                   f"kadar); örneklem içi gösterimdir.")]
    if day >= spent_from:
        return [_w("replay_spent_period",
                   f"Bu tarih {model} için daha önce değerlendirmede kullanılmış dönemde; "
                   f"bağımsız doğrulama değildir.")]
    return []


# ----------------------------------------------------------------------------
# Models.
# ----------------------------------------------------------------------------
def _firm_entry(r: pd.Series, marka) -> dict:
    return {
        "rank": r["rank"], "rank_original": r["rank_original"],
        "firma_key": r["firma_key"], "firma_adi": r["firma_adi_display"],
        "score": r["score"],
        "directory": {"city": r["sehir"], "district": r["ilce"], "detay_tip": r["detay_tip"],
                      "tip": r["tip"], "profil": r["profil"],
                      "in_directory": r["in_directory"], "id_collision": r["has_id_collision"]},
        "contact": None,
        "history": {"lane": r["lane"], "known_via": r["known_via"],
                    "n_cars_bid_prior": r["firm_n_cars_bid_prior"],
                    "n_wins_prior": r["firm_n_wins_prior"],
                    "same_marka_cars_bid_prior": r["aff_marka_n_prior"],
                    "days_since_last_bid": r["firm_days_since_last_bid"],
                    "days_since_last_participation": r["firm_days_since_last_participation"],
                    "total_bids_prior": r["firm_total_bids_prior"],
                    "bid_in_last_365d": r["firm_is_active_365d"]},
        "snippets": MA.history_snippets(r, marka=marka),
    }


def _run_model_a(bundle: Bundle, rows: pd.DataFrame, spec: FL.FilterSpec, *, mode: str,
                 recency: dict, pid: str) -> dict:
    model = bundle.model_a
    if model is None:
        return {"status": "unavailable", "reason": "model_not_loaded",
                "message": "Firma sıralama modeli bu sunucuda yüklenemedi."}
    day = pd.Timestamp(rows["aa_tarihi"].iloc[0])
    try:
        ranked = MA.rank_pool(model, rows, new_vehicle=mode == "current")
        selected, ahead, summary = FL.apply_filters(ranked, spec)
    except Exception:                                   # noqa: BLE001 — isolated per model
        return _model_error("model_a", pid, "Firma sıralaması üretilemedi; ayrıntı sunucu "
                                            "kaydında.")
    marka = rows["marka"].iloc[0]
    warnings = []
    for col, src in _RANKER_CATEGORIES.items():
        v = rows[src].iloc[0]
        if not _blank(v) and v not in model.ranker.vocabulary[col]:
            warnings.append(_w(f"unseen_{src}",
                               f"{src} değeri '{v}' modelin eğitim sözlüğünde yok; bu bilgi "
                               f"sıralamada kullanılamadı."))
    if summary["pool_size"] == 0:
        warnings.append(_w("empty_pool", "Bu tarihte bilinen firma havuzu boş."))
    elif summary["n_selected"] < spec.top_n:
        warnings.append(_w("short_list", "Filtrelerden sonra uygun firma sayısı istenen liste "
                                         "uzunluğundan az."))
    stale = recency["models"]["model_a"]["stale_sources"]
    if stale:
        text = (f"Firma sıralamasının kullandığı eski kaynaklar: {_ends(recency, stale)}. Bu "
                f"tarihlerden sonraki kayıtlar sıralamaya görünmez.")
        if "bids" in stale:
            text += (" 'Son teklif' ve '365 günde teklif' göstergeleri bu boşluk nedeniyle "
                     "düşük görünebilir; firmanın faaliyetsiz olduğu anlamına gelmez."
                     + (" '365 günde teklif' filtresi bu nedenle uygun firmaları da "
                        "dışlayabilir." if spec.active_365d_only else ""))
        if "participation" in stale:
            text += " 'Son katılım' göstergesi de aynı nedenle eski görünebilir."
        warnings.append(_w("stale_activity", text))
    if mode == "historical_replay":
        warnings += _replay_warnings(day, model.train_end, SPLIT_BOUNDARY, "Model A")
    names = ranked.set_index("firma_key")["firma_adi_display"] if len(ranked) else None
    log.info("phase7 prediction %s: model_a pool=%d selected=%d excluded=%s", pid,
             summary["pool_size"], summary["n_selected"], summary["excluded_by_reason"])
    return {
        "status": "ok",
        "score_note": MA.SCORE_NOTE,
        "explanation_note": EXPLANATION_NOTE,
        "contact_note": CONTACT_NOTE,
        "directory_note": DIRECTORY_NOTE,
        "call_list": [_firm_entry(r, marka) for _, r in selected.iterrows()],
        "excluded_ahead": [{"firma_key": r["firma_key"],
                            "firma_adi": None if names is None else names.get(r["firma_key"]),
                            "rank_original": r["rank_original"], "reasons": r["reasons"]}
                           for _, r in ahead.iterrows()],
        "filter_summary": summary,
        "warnings": warnings,
    }


def _run_model_b(bundle: Bundle, rows: pd.DataFrame, *, mode: str, recency: dict,
                 source_rows, pid: str) -> dict:
    model = bundle.model_b
    base = {"label": MB.LABEL, "meaning": MB.MEANING, "model": MB.MODEL_NAME,
            "currency": "TRY", "interval": None,
            "interval_note": "Bu model için geçerli kalibrasyon yok; fiyat aralığı gösterilmez."}
    if model is None:
        return {**base, "status": "unavailable", "reason": "model_not_loaded",
                "message": "Fiyat modeli bu sunucuda yüklenemedi."}
    day = pd.Timestamp(rows["aa_tarihi"].iloc[0])
    fx = MB.fx_status(model, day)
    if not fx["available"]:
        return {**base, "status": "unavailable", "reason": "fx_unavailable", "fx": fx,
                "message": f"Tahmin tarihinden önceki en fazla {fx['max_lag_days']} günlük "
                           f"USDTRY kuru yok (kur kaydı {fx['snapshot_last_bar']} tarihinde "
                           f"bitiyor); fiyat gösterilmez. Kur otomatik çekilmez."}
    r = rows.iloc[0]
    missing = [c for c in MB.REQUIRED_FIELDS
               if _blank(r[c]) or (c == "model_yili" and not np.isfinite(_num(r[c])))]
    if missing:
        return {**base, "status": "unavailable", "reason": "insufficient_vehicle_fields",
                "fx": fx, "message": f"Fiyat için gerekli alanlar eksik: {missing}."}
    try:
        est = MB.estimate_price(model, rows, source_rows=source_rows)
    except Exception:                                   # noqa: BLE001 — isolated per model
        return {**base, **_model_error("model_b", pid, "Fiyat tahmini üretilemedi; ayrıntı "
                                                       "sunucu kaydında."), "fx": fx}
    warnings = []
    vocab = model.model.encoders.vocabulary
    for c in ("marka", "model"):
        v = r[c]
        if not _blank(v) and str(v).strip() not in vocab.get(c, ()):
            warnings.append(_w(f"unseen_{c}", f"{c} değeri '{v}' fiyat modelinin eğitim "
                                              f"sözlüğünde yok veya çok seyrek."))
    sup = est["history_support"]
    if sup["S3_n"] == 0:
        warnings.append(_w("no_segment_history",
                           "Bu marka-model için tahmin tarihinden önceki 365 günde kayıtlı "
                           "satış yok; tahmin daha belirsizdir."))
    if not (sup["market_median_30d_present"] and sup["market_median_90d_present"]):
        warnings.append(_w("no_recent_market",
                           "Tahmin tarihinden önceki 30/90 günde kayıtlı satış yok; fiyat güncel "
                           "piyasayı yansıtmayabilir."))
    stale = recency["models"]["model_b"]["stale_sources"]
    if stale:
        warnings.append(_w("stale_sales_history",
                           f"Fiyat geçmişinin kaynağı eski: {_ends(recency, stale)}. Bu tarihten "
                           f"sonraki satışlar fiyat geçmişine ve 30/90 günlük piyasa "
                           f"medyanlarına girmez."))
    if mode == "historical_replay":
        end = pd.Timestamp(model.fit_date_range[1]) if model.fit_date_range else None
        warnings += _replay_warnings(day, end, SPLIT_BOUNDARY, "Model B")
    return {**base, "status": "ok", "price_tl": est["price_tl"], "price_usd": est["price_usd"],
            "fx": {**fx, "rate": est["fx_rate"], "bar_date": est["fx_bar_date"]},
            "history_support": sup, "warnings": warnings}


# ----------------------------------------------------------------------------
# Response.
# ----------------------------------------------------------------------------
def _respond(bundle: Bundle, rows: pd.DataFrame, *, mode: str, spec: FL.FilterSpec,
             pid: str, request_id: str, echo: dict, support: dict, source_rows=None,
             outcome: dict | None = None) -> dict:
    day = pd.Timestamp(rows["aa_tarihi"].iloc[0])
    recency = _recency(bundle, day, mode)
    warnings = _input_warnings(rows, support, mode)
    if mode == "historical_replay":
        warnings.append(_w("historical_replay", REPLAY_NOTE))
    else:
        stale = [n for n, s in recency["sources"].items() if s["stale"]]
        if stale:
            warnings.append(_w("stale_history",
                               f"Bu kaynaklar tahmin tarihinden {recency['stale_after_days']} "
                               f"günden daha önce bitiyor: {_ends(recency, stale)}. Her kaynağın "
                               f"son gününden sonraki kayıtlar modellere görünmez; hangi modelin "
                               f"etkilendiği model uyarılarında belirtilir."))
        if recency["prediction_date_within_history"]:
            covered = [n for n, s in recency["sources"].items() if s["covers_prediction_date"]]
            warnings.append(_w("backdated_current_input",
                               "Tahmin tarihi şu kaynakların kayıtlı dönemi içinde: "
                               + ", ".join(_SOURCE_TR.get(n, n) for n in covered)
                               + ". Geriye tarihli yeni araç girişi."))
    a = _run_model_a(bundle, rows, spec, mode=mode, recency=recency, pid=pid)
    b = _run_model_b(bundle, rows, mode=mode, recency=recency, source_rows=source_rows,
                     pid=pid)
    ok = [m["status"] == "ok" for m in (a, b)]
    response = {
        "schema": SCHEMA,
        "prediction_id": pid,
        "request_id": request_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete" if all(ok) else "partial" if any(ok) else "failed",
        "mode": mode,
        "prediction_date": str(day.date()),
        "vehicle_input": echo,
        "vehicle": {c: rows[c].iloc[0] for c in _SUMMARY_FIELDS},
        "vehicle_support": support,
        "limited_support": bool(support["wawasaky"]["n"]),
        "bundle": bundle.public_manifest(),
        "data_recency": recency,
        "filters": FL.describe_filters(spec),
        "model_a": a,
        "model_b": b,
        "warnings": warnings,
        "recorded_outcome": outcome,
    }
    log.info("phase7 prediction %s: mode=%s date=%s model_a=%s model_b=%s filters=%s", pid,
             mode, day.date(), a["status"], b["status"], response["filters"]["rules_sha256"])
    return _clean(response)


def _parse_filters(filters) -> FL.FilterSpec:
    try:
        return FL.parse_filters(filters)
    except (TypeError, ValueError) as exc:
        raise InputError(str(exc)) from None


def predict_vehicle(bundle: Bundle, vehicle, *, prediction_date, filters=None,
                    request_id: str | None = None) -> dict:
    """A not-yet-auctioned vehicle -> call list + price (mode 'current'). See module doc."""
    if not isinstance(bundle, Bundle):
        raise TypeError("[predict] bundle must be a loaded Bundle")
    spec = _parse_filters(filters)
    raw, echo = _raw_frame(vehicle)
    pid = uuid.uuid4().hex
    req = pid if request_id is None else _check_request_id(request_id)
    try:
        rows = VI.build_inference_vehicles(raw, prediction_date=prediction_date,
                                           uye=bundle.member_directory,
                                           version_id=f"phase7-inference-{pid}",
                                           request_id=req)
    except (TypeError, ValueError) as exc:
        raise InputError(str(exc)) from None
    day = pd.Timestamp(rows["aa_tarihi"].iloc[0])
    start = min(a for a, _ in bundle.data_ranges.values())
    if day <= start:
        raise InputError(f"prediction_date must be after the history start {start.date()}")
    _check_values(rows, day)
    support = VI.describe_input(rows, raw=raw)
    return _respond(bundle, rows, mode="current", spec=spec, pid=pid, request_id=req,
                    echo=echo, support=support)


def replay_history(bundle: Bundle, query_id: str, *, filters=None,
                   request_id: str | None = None) -> dict:
    """A recorded vehicle re-scored at its own date (mode 'historical_replay')."""
    if not isinstance(bundle, Bundle):
        raise TypeError("[predict] bundle must be a loaded Bundle")
    spec = _parse_filters(filters)
    if not isinstance(query_id, str) or not query_id.strip():
        raise InputError("query_id must be a non-blank string")
    pid = uuid.uuid4().hex
    req = pid if request_id is None else _check_request_id(request_id)
    try:
        rows, src, outcomes = VI.from_history(bundle.clean_sales, [query_id])
    except (TypeError, ValueError) as exc:
        raise InputError(str(exc)) from None
    if rows["aa_tarihi"].isna().any() or _num(rows["aa_tarihi_invalid_flag"].iloc[0]) == 1:
        raise InputError("this recorded vehicle has no valid auction date; it cannot be replayed")
    o = outcomes.iloc[0]
    pop_rows = bundle.price_population_rows
    outcome = {"is_sold": o["is_sold"] == 1, "satis_tutari_tl": o["satis_tutari"],
               "in_price_population": None if pop_rows is None else int(src[0]) in pop_rows,
               "price_population_note": (
                   "Faz 5 fiyat popülasyonu üyeliği (target_population.build_price_population, "
                   "aynı clean_sales). Bir değerlendirme dönemine üyelik değildir; fiyat modeli "
                   "yüklenmediyse bilinmez (None)."),
               "note": "Kayıtlı sonuç; modellere girdi olarak verilmedi."}
    return _respond(bundle, rows, mode="historical_replay", spec=spec, pid=pid,
                    request_id=req, echo={"query_id": query_id},
                    support=VI.describe_input(rows), source_rows=src, outcome=outcome)


def response_json(response: dict) -> str:
    """Canonical JSON text of a response, for storage next to feedback."""
    return json.dumps(response, ensure_ascii=False, sort_keys=True, allow_nan=False)

"""
streamlit_app.py  —  Garaj demo on Streamlit (Community Cloud or `streamlit run`).

The same Phase-7 models and the same Phase-8 server logic (validation, DemoRuntime, CSV)
as the FastAPI demo; only the screen is Streamlit. Every prediction is computed on the
server by phase7_inference with the frozen models.

CONFIGURATION (Streamlit secrets — .streamlit/secrets.toml locally, "Secrets" on Cloud —
or environment variables of the same name):
  GARAJ_DEMO_USERNAME        pilot user name
  GARAJ_DEMO_PASSWORD_HASH   python pipeline/run_demo.py hash-password output
  GARAJ_DEMO_EMAIL           optional: the account's e-mail (sign-in alternative)
  GARAJ_DEMO_DATA_URL        optional: HTTPS address of the zip made by
                             pipeline/pack_demo_data.py (private storage)
  GARAJ_DEMO_DATA_TOKEN      optional: bearer token for that address (e.g. a read-only
                             Hugging Face token)
  GARAJ_DEMO_ROOT            optional: data root when the files are already on disk
Without GARAJ_DEMO_DATA_URL the data root is GARAJ_DEMO_ROOT or this repository.

Data, models and secrets are never committed: see docs/STREAMLIT_YAYIN.md.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from phase7_inference import predict as PR  # noqa: E402
from phase8_demo import app as A  # noqa: E402
from phase8_demo import security as SEC  # noqa: E402
from phase8_demo.store import Store  # noqa: E402

log = logging.getLogger("streamlit_app")

DATA_DIR = Path(tempfile.gettempdir()) / "garaj_demo_data"
STORE_PATH = Path(tempfile.gettempdir()) / "garaj_demo" / "demo.sqlite3"
MAX_ZIP_BYTES = 2 * 1024 ** 3
SUGGEST_FIELDS = {"Yakıt Tipi": "yakit_tipi", "Vites Tipi": "vites_tipi",
                  "Kasa Tipi": "kasa_tipi", "Renk": "renk", "Lokasyon": "lokasyon"}
STATUS_TR = {"complete": "Tamamlandı", "partial": "Kısmi sonuç", "failed": "Sonuç üretilemedi"}

st.set_page_config(page_title="Garaj · Müşteri çalışma alanı", page_icon="🚗",
                   layout="wide", initial_sidebar_state="collapsed")


# ----------------------------------------------------------------------------
# Configuration and data.
# ----------------------------------------------------------------------------
def setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except Exception:                                   # noqa: BLE001 — no secrets file
        value = None
    return str(value if value is not None else os.environ.get(name, default)).strip()


def _safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    total = 0
    for info in zf.infolist():
        dest = (target / info.filename).resolve()
        if not str(dest).startswith(str(target.resolve()) + os.sep):
            raise RuntimeError("veri arşivinde geçersiz yol var")
        total += info.file_size
        if total > MAX_ZIP_BYTES:
            raise RuntimeError("veri arşivi çok büyük")
    zf.extractall(target)


def data_root() -> Path:
    url = setting("GARAJ_DEMO_DATA_URL")
    if not url:
        return Path(setting("GARAJ_DEMO_ROOT") or REPO).resolve()
    if not url.lower().startswith("https://"):
        raise RuntimeError("GARAJ_DEMO_DATA_URL https:// ile başlamalı")
    marker = DATA_DIR / ".ready"
    if marker.is_file():
        return DATA_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    token = setting("GARAJ_DEMO_DATA_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=600) as resp:         # noqa: S310 — https
        payload = resp.read(MAX_ZIP_BYTES + 1)
    if len(payload) > MAX_ZIP_BYTES:
        raise RuntimeError("veri arşivi çok büyük")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        _safe_extract(zf, DATA_DIR)
    marker.write_text("ok", encoding="utf-8")
    return DATA_DIR


@st.cache_resource(show_spinner="Modeller ve geçmiş veriler yükleniyor… (ilk açılışta birkaç "
                                 "dakika sürebilir)")
def runtime() -> A.DemoRuntime:
    """One runtime per server process, shared by every browser session."""
    settings = A.DemoSettings(root=data_root(), store_path=STORE_PATH,
                              username=setting("GARAJ_DEMO_USERNAME"),
                              password_hash=setting("GARAJ_DEMO_PASSWORD_HASH"),
                              email=setting("GARAJ_DEMO_EMAIL"))
    settings.validate()
    store = Store(settings.store_path)
    store.initialize()
    rt = A.DemoRuntime(settings, store, A.serving_code_sha256())
    rt.load()
    return rt


# ----------------------------------------------------------------------------
# Login / logout.
# ----------------------------------------------------------------------------
@st.cache_resource
def account_store() -> Store:
    """The store that may hold a newer password hash (saved by a reset in the web demo)."""
    store = Store(STORE_PATH)
    store.initialize()
    return store


def login_view() -> None:
    username, password_hash = setting("GARAJ_DEMO_USERNAME"), setting("GARAJ_DEMO_PASSWORD_HASH")
    if not username or not password_hash:
        st.error("GARAJ_DEMO_USERNAME ve GARAJ_DEMO_PASSWORD_HASH ayarlanmamış "
                 "(Streamlit 'Secrets'). Ayrıntı: docs/STREAMLIT_YAYIN.md")
        st.stop()
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.subheader("Giriş")
        st.caption("Bu çalışma alanı size özel erişim içindir.")
        with st.form("login"):
            user = st.text_input("Kullanıcı adı ya da e-mail", max_chars=SEC.MAX_IDENTIFIER_CHARS)
            pw = st.text_input("Parola", type="password", max_chars=SEC.MAX_PASSWORD_CHARS,
                               placeholder="Lütfen 12 haneli parolayı girin.")
            st.markdown(f":red[**{SEC.PASSWORD_RULES_TR.replace('$', chr(92) + '$')}**]")
            ok = st.form_submit_button("Giriş yap", type="primary")
        st.caption("Parolanızı unuttuysanız: parola sıfırlama (e-posta kodu) bu Streamlit "
                   "sürümünde yok; yöneticinize başvurun.")
        if ok:
            fails = st.session_state.get("login_failures", 0)
            account = A.Account(username, setting("GARAJ_DEMO_EMAIL"), password_hash,
                                account_store())
            if fails >= 5:
                st.error("Çok fazla hatalı deneme; sayfayı daha sonra yeniden açın.")
            elif account.check(user, pw):
                st.session_state.clear()
                st.session_state["user"] = username
                st.rerun()
            else:
                st.session_state["login_failures"] = fails + 1
                st.error("Kullanıcı adı veya parola hatalı.")
    st.stop()


def logout() -> None:
    st.session_state.clear()
    st.session_state["logged_out"] = True


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------
def fmt_int(v) -> str:
    return f"{v:,.0f}".replace(",", ".")


def choice(label: str, options: list[str], key: str, *, disabled: bool = False):
    """Select from history values, or type a new value."""
    return st.selectbox(label, options, index=None, key=key, placeholder="Seçin veya yazın",
                        accept_new_options=True, disabled=disabled)


def current_filters() -> dict:
    f = {"top_n": int(st.session_state.get("f_topn", 20))}
    cities = st.session_state.get("f_cities") or []
    if cities:
        f["firm_cities"] = list(cities)
    excluded = st.session_state.setdefault("excluded", {})
    if excluded:
        f["exclude_firm_keys"] = list(excluded)
    if st.session_state.get("f_active"):
        f["active_365d_only"] = True
    return f


def run(rt: A.DemoRuntime, record: dict, parent_id: str | None = None) -> None:
    try:
        with st.spinner("Hesaplanıyor… İki model sunucuda çalışıyor."):
            view = rt.run(st.session_state["user"], record, parent_id=parent_id)
    except A.ApiError as exc:
        st.session_state["error"] = exc.message
        return
    st.session_state["error"] = None
    st.session_state["view"] = view
    st.session_state["record"] = record


def rerun_with_filters(rt: A.DemoRuntime) -> None:
    view, record = st.session_state.get("view"), st.session_state.get("record")
    if not view or not record:
        return
    try:
        filters = A.validate_filters(current_filters())
    except A.ApiError as exc:
        st.session_state["error"] = exc.message
        return
    run(rt, {**record, "filters": filters}, parent_id=view["prediction_id"])


# ----------------------------------------------------------------------------
# Screen.
# ----------------------------------------------------------------------------
def header() -> None:
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.markdown("### Garaj / Müşteri çalışma alanı  `Kontrollü demo`")
    with right:
        st.caption(f"👤 {st.session_state['user']}")
        st.button("Oturumu kapat", on_click=logout, type="secondary")


def status_strip(rt: A.DemoRuntime) -> None:
    s = rt.status()
    if s["state"] != "ready" or not s["bundle"]:
        st.error(s["message"] or "Tahmin paketi yüklenemedi.")
        st.stop()
    b = s["bundle"]
    parts = [f"{name}: {r[0]} – {r[1]}" for name, r in b["data_ranges"].items()]
    if b.get("fx") and b["fx"].get("bar_range"):
        parts.append(f"USDTRY kuru: {b['fx']['bar_range'][1]} tarihine kadar")
    comp = b["components"]
    parts.append("Firma modeli: " + ("hazır" if comp["model_a"] == "ready" else "yok"))
    parts.append("Fiyat modeli: " + ("hazır" if comp["model_b"] == "ready" else "yok"))
    st.caption("Veri durumu · " + " · ".join(parts))


def vehicle_form(rt: A.DemoRuntime) -> None:
    opts = rt.options
    sug = opts["suggestions"]
    mode = st.radio("Girdi", ["Yeni araç", "Geçmiş örnek"], horizontal=True,
                    label_visibility="collapsed", key="mode")
    if mode == "Geçmiş örnek":
        labels = {e["example_id"]: " · ".join(str(x) for x in (
            e["date"], e["marka"], e["model"], e["model_uzantisi"], e["model_yili"],
            f"{fmt_int(e['km_rounded'])} km" if e["km_rounded"] is not None else None) if x)
            for e in rt.examples}
        eid = st.selectbox("Kayıtlı araç", list(labels), format_func=labels.get, index=None,
                           placeholder="Seçin")
        st.caption(PR.REPLAY_NOTE)
        filters_row(rt)
        if st.button("Geçmiş örneği puanla", type="primary", disabled=eid is None):
            try:
                run(rt, rt.replay_request(eid, A.validate_filters(current_filters())))
            except A.ApiError as exc:
                st.session_state["error"] = exc.message
        return

    v: dict = {}
    st.markdown("**Araç**")
    c = st.columns(5)
    with c[0]:
        v["Marka"] = choice("Marka *", sug["marka"], "v_marka")
    with c[1]:
        v["Model"] = choice("Model", opts["models_by_marka"].get(v["Marka"] or "", []), "v_model")
    with c[2]:
        v["Model Uzantısı"] = st.text_input("Model uzantısı", max_chars=160, key="v_uzanti")
    with c[3]:
        v["Model Yılı"] = st.number_input("Model yılı", min_value=1950, max_value=2100,
                                          value=None, step=1, placeholder="ör. 2019",
                                          key="v_yil")
    with c[4]:
        v["KM"] = st.number_input("KM", min_value=0, max_value=3_000_000, value=None,
                                  step=1000, placeholder="ör. 120000", key="v_km")
    c = st.columns(5)
    for col, (field, sk) in zip(c, SUGGEST_FIELDS.items()):
        with col:
            v[field] = choice(field.replace(" Tipi", " tipi"), sug[sk], f"v_{sk}")

    st.markdown("**Durum, hasar ve satış**")
    c = st.columns(6)
    with c[0]:
        v["Eks.Puanı"] = st.number_input("Ekspertiz puanı (0–10)", min_value=0.0,
                                         max_value=10.0, value=None, step=0.5, key="v_eks")
    with c[1]:
        v["Tramer Adedi"] = st.number_input("Tramer adedi", min_value=0, max_value=500,
                                            value=None, step=1, key="v_tadet")
    with c[2]:
        v["Tramer Tutarı"] = st.number_input("Tramer tutarı (TL)", min_value=0.0,
                                             max_value=1e9, value=None, step=1000.0,
                                             key="v_ttutar")
    with c[3]:
        v["Min.Fiyat"] = st.number_input("Min.Fiyat (TL, bilgi)", min_value=0.0, max_value=1e10,
                                         value=None, step=10000.0, key="v_minf",
                                         help="Yalnızca kayıt içindir; fiyat tahmini "
                                              "Min.Fiyat kullanmaz.")
    with c[4]:
        v["Plaka"] = st.text_input("Plaka (isteğe bağlı)", max_chars=16, key="v_plaka")
    with c[5]:
        day = st.date_input("Tahmin tarihi *", value=date.today(), format="DD.MM.YYYY",
                            key="v_date")
    c = st.columns([1, 2])
    with c[0]:
        v["Satıcı Kısa Ünvan"] = st.text_input("Satıcı kısa ünvanı", max_chars=160,
                                               key="v_satici")
    with c[1]:
        v["Ekspertiz"] = st.text_area("Ekspertiz metni", max_chars=4000, height=68,
                                      key="v_ekspertiz",
                                      placeholder="Boşsa hasar bilgisi bilinmiyor sayılır; "
                                                  "hasarsız kabul edilmez.")
    filters_row(rt)
    if st.button("Önerileri getir", type="primary"):
        vehicle = {k: (x.strip() if isinstance(x, str) else x) for k, x in v.items()}
        vehicle = {k: x for k, x in vehicle.items() if x not in (None, "")}
        try:
            record = rt.current_request(A.validate_vehicle(vehicle),
                                        A.validate_date(day.isoformat() if day else None),
                                        A.validate_filters(current_filters()))
        except A.ApiError as exc:
            st.session_state["error"] = exc.message
            return
        run(rt, record)


def filters_row(rt: A.DemoRuntime) -> None:
    st.markdown("**Firma listesi filtreleri**")
    c = st.columns([1, 3, 2, 4])
    with c[0]:
        st.number_input("Liste uzunluğu", min_value=1, max_value=rt.options["top_n"]["max"],
                        value=rt.options["top_n"]["default"], step=1, key="f_topn")
    with c[1]:
        st.multiselect("Firma şehri (güncel dizin)", rt.options["firm_cities"], key="f_cities",
                       placeholder="Tüm şehirler")
    with c[2]:
        st.checkbox("Yalnızca son 365 günde teklifi olan firmalar", key="f_active")
    with c[3]:
        excluded: dict = st.session_state.setdefault("excluded", {})
        keep = st.multiselect("Dışlanan firmalar", list(excluded), default=list(excluded),
                              format_func=lambda k: excluded.get(k, k),
                              placeholder="Henüz dışlanan firma yok",
                              key=f"f_excluded_{st.session_state.get('excluded_ver', 0)}")
        for k in [k for k in excluded if k not in keep]:
            excluded.pop(k)


def firms_card(rt: A.DemoRuntime, view: dict) -> None:
    st.subheader("Kimi arayalım?")
    a = view["model_a"]
    for w in a.get("warnings") or []:
        st.warning(w["message"])
    if a.get("status") != "ok":
        st.error(a.get("message") or "Firma listesi üretilemedi.")
        return
    s = a.get("filter_summary") or {}
    st.caption(f"Tahmin tarihindeki firma havuzu {fmt_int(s.get('pool_size', 0))} firma; "
               f"filtre sonrası uygun {fmt_int(s.get('n_eligible', 0))}; listede "
               f"{fmt_int(s.get('n_selected', 0))}.")
    calls = a.get("call_list") or []
    if not calls:
        st.info("Filtrelerden sonra uygun firma yok. Şehir veya faaliyet filtresini gevşetin "
                "ya da dışlamaları kaldırın.")
        return
    rows = []
    for e in calls:
        d, h = e["directory"], e["history"]
        rows.append({"Çıkar": False, "Sıra": e["rank"], "Firma": e["firma_adi"] or e["firma_key"],
                     "Şehir": " / ".join(x for x in (d.get("city"), d.get("district")) if x),
                     "Tip": d.get("detay_tip") or d.get("tip"),
                     "Geçmiş": " ".join(e["snippets"][:2])})
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True,
        disabled=["Sıra", "Firma", "Şehir", "Tip", "Geçmiş"],
        column_config={"Çıkar": st.column_config.CheckboxColumn(
            "Çıkar", help="Listeden çıkarmak istediğiniz firmaları işaretleyin (birden fazla "
                          "seçilebilir)")},
        key=f"firms_{view['prediction_id']}")
    marked = [calls[i] for i in edited.index[edited["Çıkar"]].tolist()]
    c = st.columns([2, 1])
    with c[0]:
        if st.button(f"İşaretli {len(marked)} firmayı listeden çıkar ve yeniden hesapla",
                     disabled=not marked or not view["stored"], type="primary"):
            excluded = st.session_state.setdefault("excluded", {})
            for e in marked:
                excluded[e["firma_key"]] = e["firma_adi"] or e["firma_key"]
            st.session_state["excluded_ver"] = st.session_state.get("excluded_ver", 0) + 1
            rerun_with_filters(rt)
            st.rerun()
    with c[1]:
        if view["stored"]:
            rec = rt.store.get_prediction(view["prediction_id"], st.session_state["user"])
            if rec:
                st.download_button("Listeyi indir (CSV)", A.call_list_csv(rec["response"]),
                                   file_name=f"arama_listesi_{view['prediction_date']}.csv",
                                   mime="text/csv")
    for t in (a.get("score_note"), a.get("explanation_note"), a.get("directory_note"),
              a.get("contact_note")):
        if t:
            st.caption(t)


def price_card(view: dict) -> None:
    st.subheader("Tahmini ihale satış fiyatı")
    b = view["model_b"]
    for w in b.get("warnings") or []:
        st.warning(w["message"])
    if b.get("status") != "ok":
        st.info(b.get("message") or "Fiyat gösterilemiyor.")
        return
    fx = b.get("fx") or {}
    st.metric("Tahmini fiyat", f"{fmt_int(round(b['price_tl'] / 1000) * 1000)} TL")
    st.caption(f"Yaklaşık {fmt_int(b['price_usd'])} USD · USDTRY {fx.get('rate', '?')} "
               f"({fx.get('bar_date', '?')} açılış kuru) · 1.000 TL'ye yuvarlandı")
    if b.get("meaning"):
        st.info(b["meaning"])
    for h in b.get("history_support_tr") or []:
        st.caption(f"Önceki 365 günde {h['label']}: {fmt_int(h['n'] or 0)} satış")


def results(rt: A.DemoRuntime) -> None:
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    view = st.session_state.get("view")
    if not view:
        st.info("Araç bilgilerini girip **Önerileri getir** düğmesine basın. İki ayrı çıktı "
                "görürsünüz: aranacak firmaların öncelik listesi ve tahmini ihale satış fiyatı.")
        return
    st.divider()
    v = view.get("vehicle") or {}
    title = " ".join(str(x) for x in (v.get("marka"), v.get("model"), v.get("model_uzantisi"))
                     if x) or "Araç"
    st.markdown(f"#### {title}  ·  {STATUS_TR.get(view['status'], view['status'])}")
    st.caption(f"Tahmin tarihi {view['prediction_date']} · Tahmin no "
               f"{view['prediction_id'][:12]}")
    if view.get("replay_note"):
        st.info(view["replay_note"])
    for w in view.get("warnings") or []:
        st.warning(w["message"])
    left, right = st.columns([3, 2])
    with left:
        firms_card(rt, view)
    with right:
        price_card(view)


def main() -> None:
    if not st.session_state.get("user"):
        if st.session_state.pop("logged_out", False):
            st.success("Oturum kapatıldı. Bu sekmeyi kapatabilirsiniz.")
        login_view()
    header()
    try:
        rt = runtime()
    except Exception as exc:                            # noqa: BLE001 — shown sanitized
        log.exception("streamlit demo: runtime failed")
        st.error(f"Tahmin paketi yüklenemedi: {exc}")
        st.stop()
    status_strip(rt)
    with st.container(border=True):
        vehicle_form(rt)
    results(rt)


main()

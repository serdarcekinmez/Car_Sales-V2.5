/*
 * Garaj customer workspace (Phase 8). Vanilla JS, no build step, no CDN.
 * Every prediction is computed by the server (Phase 7 models); this file only collects
 * input, calls the JSON API and renders the server's display view. Data is rendered with
 * textContent / DOM nodes only (never innerHTML).
 */
'use strict';

(function () {
  const state = {
    csrf: null,
    user: null,
    status: null,
    options: null,
    examples: [],
    current: null,
    cities: [],
    excluded: new Map(),          // firma_key -> display name
    marked: new Map(),            // firms ticked in the shown list: firma_key -> display name
    priceVerdict: null,
    busy: false,
    statusTimer: null,
    tab: 'new',
  };

  const REASON_TR = {
    excluded_by_customer: 'sizin dışladığınız',
    city_unknown: 'şehri bilinmiyor',
    city_not_selected: 'seçili şehirlerde değil',
    no_bid_365d: 'son 365 günde kayıtlı teklifi yok',
  };
  const STATUS_TR = {
    complete: ['Tamamlandı', 'badge-ok'],
    partial: ['Kısmi sonuç', 'badge-warn'],
    failed: ['Sonuç üretilemedi', 'badge-err'],
  };
  const SOURCE_TR = { sales: 'Satış geçmişi', bids: 'Teklif geçmişi', participation: 'Katılım geçmişi' };
  const STRONG_WARNINGS = new Set(['limited_support_wawasaky', 'stale_history', 'historical_replay']);

  const fmtInt = new Intl.NumberFormat('tr-TR', { maximumFractionDigits: 0 });
  const fmtDec = new Intl.NumberFormat('tr-TR', { maximumFractionDigits: 2 });
  const fmtRate = new Intl.NumberFormat('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 4 });

  // ------------------------------------------------------------------ DOM helpers
  const $ = (id) => document.getElementById(id);

  function el(tag, opts, ...children) {
    const node = document.createElement(tag);
    if (opts) {
      if (opts.cls) node.className = opts.cls;
      if (opts.text !== undefined && opts.text !== null) node.textContent = String(opts.text);
      if (opts.attrs) {
        for (const [k, v] of Object.entries(opts.attrs)) {
          if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : String(v));
        }
      }
      if (opts.on) for (const [k, f] of Object.entries(opts.on)) node.addEventListener(k, f);
    }
    for (const c of children) {
      if (c === undefined || c === null || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function show(node, on) { node.hidden = !on; }

  function toast(text) {
    const t = $('toast');
    t.textContent = text;
    show(t, true);
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => show(t, false), 3200);
  }

  function localToday() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  }

  function daysBetween(isoA, isoB) {
    const a = Date.parse(isoA + 'T00:00:00Z');
    const b = Date.parse(isoB + 'T00:00:00Z');
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return Math.round((b - a) / 86400000);
  }

  // ------------------------------------------------------------------ API
  class ApiError extends Error {
    constructor(status, code, message) { super(message); this.status = status; this.code = code; }
  }

  async function api(method, path, body) {
    const opts = { method, headers: { Accept: 'application/json' }, credentials: 'same-origin', cache: 'no-store' };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    if (method !== 'GET' && state.csrf) opts.headers['X-CSRF-Token'] = state.csrf;
    let res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw new ApiError(0, 'network', 'Sunucuya ulaşılamadı. Bağlantıyı kontrol edin.');
    }
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const err = data && data.error ? data.error : null;
      const message = err && err.message ? err.message
        : res.status === 413 ? 'İstek çok büyük.'
        : res.status >= 500 ? 'Sunucu hatası.' : `İstek başarısız (${res.status}).`;
      if (res.status === 401 && path !== '/api/login') {
        state.csrf = null;
        showLogin('Oturumunuz sona erdi; yeniden giriş yapın.');
      }
      throw new ApiError(res.status, err ? err.code : 'http_' + res.status, message);
    }
    return data;
  }

  // ------------------------------------------------------------------ numbers
  // Turkish input: "120.000" = 120000, "1.250.000,50" = 1250000.5, "8,5" = 8.5, "8.5" = 8.5.
  function parseNumber(text, integer) {
    const s = String(text === undefined || text === null ? '' : text).trim().replace(/\s+/g, '');
    if (s === '') return { empty: true };
    let norm = null;
    if (/^\d{1,3}(\.\d{3})+(,\d+)?$/.test(s)) norm = s.replace(/\./g, '').replace(',', '.');
    else if (/^\d+(,\d+)?$/.test(s)) norm = s.replace(',', '.');
    else if (/^\d+\.\d+$/.test(s)) norm = s;
    if (norm === null) return { error: 'sayı biçimi anlaşılamadı' };
    const v = Number(norm);
    if (!Number.isFinite(v)) return { error: 'geçerli bir sayı değil' };
    if (integer && !Number.isInteger(v)) return { error: 'tam sayı olmalı' };
    return { value: v };
  }

  function numberRule(name) {
    return state.options && state.options.number_fields ? state.options.number_fields[name] : null;
  }

  function fieldLabel(input) {
    const label = input.closest('label');
    const span = label ? label.querySelector('.field-label') : null;
    return span ? span.firstChild.textContent.trim() : input.dataset.field;
  }

  function updatePreview(input) {
    const name = input.dataset.field;
    const preview = document.querySelector(`[data-preview-for="${CSS.escape(name)}"]`);
    if (!preview) return;
    const r = parseNumber(input.value, input.dataset.kind === 'int');
    preview.classList.remove('is-error');
    input.removeAttribute('aria-invalid');
    if (r.empty) { preview.textContent = ''; return; }
    if (r.error) {
      preview.textContent = r.error;
      preview.classList.add('is-error');
      input.setAttribute('aria-invalid', 'true');
      return;
    }
    const rule = numberRule(name);
    if (rule && (r.value < rule.min || r.value > rule.max)) {
      preview.textContent = `${fmtDec.format(rule.min)}–${fmtDec.format(rule.max)} aralığında olmalı`;
      preview.classList.add('is-error');
      input.setAttribute('aria-invalid', 'true');
      return;
    }
    preview.textContent = name === 'Model Yılı' ? `= ${r.value}` : `= ${fmtDec.format(r.value)}`;
  }

  // ------------------------------------------------------------------ views
  function showLogin(message) {
    show($('view-app'), false);
    show($('session-box'), false);
    show($('view-login'), true);
    const err = $('login-error');
    err.textContent = message || '';
    show(err, Boolean(message));
    clearInterval(state.statusTimer);
    state.statusTimer = null;
    $('login-user').focus();
  }

  function onLoggedIn(session) {
    state.csrf = session.csrf_token;
    state.user = session.username;
    $('session-user').textContent = session.username;
    show($('view-login'), false);
    show($('session-box'), true);
    show($('btn-shutdown'), Boolean(session.can_shutdown));
    show($('view-app'), true);
    if (!$('f-date').value) $('f-date').value = localToday();
    loadStatus();
    refreshRecent();
  }

  // ------------------------------------------------------------------ status / options
  async function loadStatus() {
    let st;
    try {
      st = await api('GET', '/api/status');
    } catch (e) {
      renderStatusError(e.message);
      return;
    }
    state.status = st;
    renderStatus(st);
    if (st.state === 'loading') {
      if (!state.statusTimer) state.statusTimer = setInterval(loadStatus, 4000);
      return;
    }
    clearInterval(state.statusTimer);
    state.statusTimer = null;
    if (st.state === 'ready' && !state.options) {
      try {
        state.options = await api('GET', '/api/options');
        fillOptions(state.options);
        const ex = await api('GET', '/api/examples');
        state.examples = ex.items || [];
        fillExamples(ex);
      } catch (e) {
        renderStatusError(e.message);
      }
    }
  }

  function renderStatusError(message) {
    const strip = $('status-strip');
    clear(strip);
    strip.append(el('span', { cls: 'status-item is-bad', text: message }));
  }

  function renderStatus(st) {
    const strip = $('status-strip');
    clear(strip);
    if (st.state === 'loading') {
      strip.append(el('span', { cls: 'status-item', text: 'Modeller ve geçmiş veriler yükleniyor… Bu birkaç dakika sürebilir.' }));
      return;
    }
    if (st.state === 'failed' || !st.bundle) {
      strip.append(el('span', { cls: 'status-item is-bad', text: st.message || 'Tahmin paketi yüklenemedi.' }));
      return;
    }
    const b = st.bundle;
    const today = $('f-date').value || st.today;
    strip.append(el('span', { cls: 'lead', text: 'Veri durumu:' }));
    for (const [name, range] of Object.entries(b.data_ranges || {})) {
      const lag = daysBetween(range[1], today);
      const stale = lag !== null && lag > b.stale_after_days;
      const text = `${SOURCE_TR[name] || name}: ${range[0]} – ${range[1]}` + (lag !== null && lag > 0 ? ` (${lag} gün önce bitti)` : '');
      strip.append(el('span', { cls: 'status-item' + (stale ? ' is-stale' : ''), text }));
    }
    if (b.fx && b.fx.bar_range) {
      const lag = daysBetween(b.fx.bar_range[1], today);
      const bad = lag !== null && lag > b.fx.max_lag_days;
      strip.append(el('span', {
        cls: 'status-item' + (bad ? ' is-bad' : ''),
        text: `USDTRY kuru: ${b.fx.bar_range[1]} tarihine kadar` + (bad ? ' (fiyat için çok eski)' : ''),
      }));
    }
    const comp = b.components || {};
    strip.append(el('span', { cls: 'status-item' + (comp.model_a === 'ready' ? '' : ' is-bad'), text: 'Firma modeli: ' + (comp.model_a === 'ready' ? 'hazır' : 'kullanılamıyor') }));
    strip.append(el('span', { cls: 'status-item' + (comp.model_b === 'ready' ? '' : ' is-bad'), text: 'Fiyat modeli: ' + (comp.model_b === 'ready' ? 'hazır' : 'kullanılamıyor') }));
  }

  function fillDatalist(id, values) {
    const dl = $(id);
    if (!dl) return;
    clear(dl);
    for (const v of values || []) dl.append(el('option', { attrs: { value: v } }));
  }

  function fillOptions(opts) {
    const s = opts.suggestions || {};
    for (const key of Object.keys(s)) fillDatalist('dl-' + key, s[key]);
    fillDatalist('dl-cities', opts.firm_cities || []);
    updateModelList();
    const topn = $('f-topn');
    if (!topn.value) topn.value = String(opts.top_n ? opts.top_n.default : 20);
  }

  function fillExamples(ex) {
    $('history-note').textContent = ex.note || '';
    const sel = $('example-select');
    clear(sel);
    if (!state.examples.length) {
      sel.append(el('option', { attrs: { value: '' }, text: 'Uygun geçmiş örnek yok' }));
      return;
    }
    for (const e of state.examples) {
      const parts = [e.date, [e.marka, e.model, e.model_uzantisi].filter(Boolean).join(' '), e.model_yili];
      if (e.km_rounded !== null && e.km_rounded !== undefined) parts.push(`~${fmtInt.format(e.km_rounded)} km`);
      sel.append(el('option', { attrs: { value: e.example_id }, text: parts.join(' · ') }));
    }
  }

  function currentMarka() {
    const input = document.querySelector('[data-field="Marka"]');
    return input ? input.value.trim() : '';
  }

  function modelsFor(marka) {
    const map = (state.options && state.options.models_by_marka) || {};
    if (map[marka]) return map[marka];
    const lower = marka.toLocaleLowerCase('tr-TR');
    for (const k of Object.keys(map)) if (k.toLocaleLowerCase('tr-TR') === lower) return map[k];
    return [];
  }

  function updateModelList() { fillDatalist('dl-model', modelsFor(currentMarka())); }

  function suggestionList(key) {
    if (!state.options) return null;
    if (key === 'model') return modelsFor(currentMarka());
    return (state.options.suggestions || {})[key] || null;
  }

  function updateHint(input) {
    const hint = document.querySelector(`[data-hint-for="${CSS.escape(input.dataset.field)}"]`);
    if (!hint) return;
    const list = suggestionList(input.dataset.suggest);
    const v = input.value.trim();
    if (!v || !list || list.includes(v)) { show(hint, false); return; }
    hint.textContent = 'Geçmiş veride sık görülen değerler arasında yok. Yazımı kontrol edin; yeni değer de girilebilir, ancak model bu bilgiyi kullanamayabilir.';
    show(hint, true);
  }

  // ------------------------------------------------------------------ input collection
  function collectVehicle() {
    const vehicle = {};
    const errors = [];
    for (const input of document.querySelectorAll('#pane-new [data-field]')) {
      const name = input.dataset.field;
      if (input.dataset.kind) {
        const r = parseNumber(input.value, input.dataset.kind === 'int');
        if (r.empty) continue;
        if (r.error) { errors.push(`${fieldLabel(input)}: ${r.error}.`); continue; }
        const rule = numberRule(name);
        if (rule && (r.value < rule.min || r.value > rule.max)) {
          errors.push(`${fieldLabel(input)}: ${fmtDec.format(rule.min)}–${fmtDec.format(rule.max)} aralığında olmalı.`);
          continue;
        }
        vehicle[name] = r.value;
      } else {
        const t = input.value.trim();
        if (t) vehicle[name] = t;
      }
    }
    if (!vehicle['Marka']) errors.push('Marka zorunludur.');
    return { vehicle, errors };
  }

  function collectFilters() {
    const errors = [];
    const filters = {};
    const max = state.options && state.options.top_n ? state.options.top_n.max : 200;
    const r = parseNumber($('f-topn').value, true);
    if (r.empty || r.error || r.value < 1 || r.value > max) errors.push(`Liste uzunluğu 1–${max} arasında bir tam sayı olmalı.`);
    else filters.top_n = r.value;
    if (state.cities.length) filters.firm_cities = state.cities.slice();
    if (state.excluded.size) filters.exclude_firm_keys = Array.from(state.excluded.keys());
    if ($('f-active').checked) filters.active_365d_only = true;
    return { filters, errors };
  }

  function showFormErrors(errors) {
    const box = $('form-error');
    clear(box);
    if (!errors.length) { show(box, false); return; }
    const ul = el('ul');
    for (const e of errors) ul.append(el('li', { text: e }));
    box.append(ul);
    show(box, true);
  }

  // ------------------------------------------------------------------ filters UI
  function renderCityChips() {
    const box = $('city-chips');
    clear(box);
    for (const c of state.cities) {
      box.append(el('span', { cls: 'chip' },
        el('span', { cls: 'chip-text', text: c }),
        el('button', { attrs: { type: 'button', 'aria-label': `${c} şehrini kaldır` }, text: '×', on: { click: () => { state.cities = state.cities.filter((x) => x !== c); renderCityChips(); } } })));
    }
  }

  function addCity() {
    const input = $('f-city');
    const v = input.value.trim();
    if (!v) return;
    if (state.cities.length >= 100) { showFormErrors(['En fazla 100 şehir seçilebilir.']); return; }
    if (!state.cities.some((c) => c.toLocaleLowerCase('tr-TR') === v.toLocaleLowerCase('tr-TR'))) state.cities.push(v);
    input.value = '';
    renderCityChips();
  }

  function renderExcludedChips() {
    const box = $('exclude-chips');
    clear(box);
    for (const [key, name] of state.excluded) {
      box.append(el('span', { cls: 'chip' },
        el('span', { cls: 'chip-text', text: name, attrs: { title: key } }),
        el('button', { attrs: { type: 'button', 'aria-label': `${name} dışlamasını kaldır` }, text: '×', on: { click: () => { state.excluded.delete(key); renderExcludedChips(); } } })));
    }
    show($('exclude-empty'), state.excluded.size === 0);
    show($('btn-clear-excluded'), state.excluded.size > 0);
  }

  function restoreFilters(view) {
    const rules = (view.filters && view.filters.rules) || {};
    if (rules.top_n) $('f-topn').value = String(rules.top_n);
    $('f-active').checked = Boolean(rules.active_365d_only);
    state.cities = (rules.firm_cities || []).slice();
    const names = new Map();
    const a = view.model_a || {};
    for (const e of (a.call_list || [])) names.set(e.firma_key, e.firma_adi);
    for (const e of (a.excluded_ahead || [])) names.set(e.firma_key, e.firma_adi);
    state.excluded = new Map();
    for (const k of (rules.exclude_firm_keys || [])) state.excluded.set(k, names.get(k) || state.excluded.get(k) || k);
    renderCityChips();
    renderExcludedChips();
  }

  function fillForm(vehicleInput) {
    for (const input of document.querySelectorAll('#pane-new [data-field]')) {
      const v = vehicleInput ? vehicleInput[input.dataset.field] : undefined;
      if (v === undefined || v === null) { input.value = ''; continue; }
      input.value = typeof v === 'number' ? (input.dataset.field === 'Model Yılı' ? String(v) : fmtDec.format(v)) : String(v);
      if (input.dataset.kind) updatePreview(input);
    }
    updateModelList();
  }

  // ------------------------------------------------------------------ requests
  function setBusy(on) {
    state.busy = on;
    for (const id of ['btn-predict', 'btn-replay']) $(id).disabled = on;
    $('btn-apply-filters').disabled = on || !state.current || !state.current.stored;
    for (const b of document.querySelectorAll('#firms-body [data-exclude]')) b.disabled = on || !(state.current && state.current.stored);
    updateBulkBar();
    show($('result-progress'), on);
    if (on) show($('result-empty'), false);
  }

  async function runRequest(path, body) {
    if (state.busy) return;
    setBusy(true);
    show($('result-error'), false);
    try {
      const view = await api('POST', path, body);
      state.current = view;
      renderResult(view);
      refreshRecent();
      $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      const box = $('result-error');
      box.textContent = (e.status === 400 ? 'Girdi kabul edilmedi: ' : 'Tahmin üretilemedi: ') + e.message
        + (state.current ? ' Aşağıda önceki sonuç görünüyor.' : '');
      show(box, true);
      if (!state.current) show($('result-empty'), true);
    } finally {
      setBusy(false);
    }
  }

  function predictNew() {
    const v = collectVehicle();
    const f = collectFilters();
    const errors = v.errors.concat(f.errors);
    const date = $('f-date').value;
    if (!date) errors.push('Tahmin tarihi seçilmedi.');
    showFormErrors(errors);
    if (errors.length) return;
    runRequest('/api/predict', { vehicle: v.vehicle, prediction_date: date, filters: f.filters });
  }

  function replayExample() {
    const f = collectFilters();
    const id = $('example-select').value;
    const errors = f.errors.slice();
    if (!id) errors.push('Geçmiş örnek seçilmedi.');
    showFormErrors(errors);
    if (errors.length) return;
    runRequest('/api/replay', { example_id: id, filters: f.filters });
  }

  function rerunWithFilters() {
    if (!state.current) return;
    if (!state.current.stored) { showFormErrors(['Bu sonuç kaydedilemediği için filtreyle yeniden hesaplanamıyor; yeni tahmin alın.']); return; }
    const f = collectFilters();
    showFormErrors(f.errors);
    if (f.errors.length) return;
    runRequest(`/api/predictions/${encodeURIComponent(state.current.prediction_id)}/rerun`, { filters: f.filters });
  }

  function excludeFirms(entries) {
    if (state.busy || !entries.length) return;
    for (const [key, name] of entries) state.excluded.set(key, name || key);
    renderExcludedChips();
    rerunWithFilters();
  }

  function updateBulkBar() {
    const n = state.marked.size;
    $('firms-bulk-count').textContent = n ? `${fmtInt.format(n)} firma işaretli` : 'Birden fazla firmayı işaretleyip birlikte çıkarabilirsiniz.';
    $('btn-exclude-selected').disabled = n === 0 || state.busy || !(state.current && state.current.stored);
  }

  async function refreshRecent() {
    let data;
    try { data = await api('GET', '/api/predictions'); } catch (e) { return; }
    const list = $('recent-list');
    clear(list);
    if (!data.items.length) { list.append(el('li', { cls: 'muted', text: 'Henüz tahmin yok.' })); return; }
    for (const item of data.items) {
      const status = STATUS_TR[item.status] ? STATUS_TR[item.status][0] : item.status;
      const when = item.stored_at_utc ? new Date(item.stored_at_utc).toLocaleString('tr-TR') : '';
      list.append(el('li', null, el('button', { attrs: { type: 'button' }, on: { click: () => openPrediction(item.prediction_id) } },
        item.title || 'Tahmin',
        el('small', { text: `${item.prediction_date} · ${status}${item.parent_id ? ' · filtreyle yeniden' : ''} · ${when}` }))));
    }
  }

  async function openPrediction(id) {
    if (state.busy) return;
    show($('result-error'), false);
    try {
      const view = await api('GET', `/api/predictions/${encodeURIComponent(id)}`);
      state.current = view;
      if (view.mode === 'current') { setTab('new'); fillForm(view.vehicle_input); $('f-date').value = view.prediction_date; }
      restoreFilters(view);
      renderResult(view);
      setBusy(false);
    } catch (e) {
      const box = $('result-error');
      box.textContent = 'Tahmin açılamadı: ' + e.message;
      show(box, true);
    }
  }

  // ------------------------------------------------------------------ rendering
  function warningList(node, warnings) {
    clear(node);
    for (const w of warnings || []) {
      node.append(el('li', { cls: STRONG_WARNINGS.has(w.code) ? 'is-strong' : '', text: w.message, attrs: { 'data-code': w.code } }));
    }
  }

  function vehicleLine(v) {
    const parts = [];
    if (v.model_yili !== null && v.model_yili !== undefined) parts.push(`${v.model_yili} model`);
    if (v.km_clean !== null && v.km_clean !== undefined) parts.push(`${fmtInt.format(v.km_clean)} km`);
    for (const k of ['yakit_tipi', 'vites_tipi', 'kasa_tipi', 'lokasyon']) if (v[k]) parts.push(v[k]);
    return parts.join(' · ');
  }

  function renderResult(view) {
    show($('result-empty'), false);
    show($('result'), true);
    const v = view.vehicle || {};
    $('res-mode').textContent = view.mode === 'historical_replay' ? 'Geçmiş örnek gösterimi' : 'Yeni araç';
    $('res-title').textContent = [v.marka, v.model, v.model_uzantisi].filter(Boolean).join(' ') || 'Araç';
    $('res-sub').textContent = [`Tahmin tarihi ${view.prediction_date}`, vehicleLine(v)].filter(Boolean).join(' · ');

    const badges = $('res-badges');
    clear(badges);
    const st = STATUS_TR[view.status] || [view.status, 'badge-neutral'];
    badges.append(el('span', { cls: 'badge ' + st[1], text: st[0] }));
    if (view.limited_support) badges.append(el('span', { cls: 'badge badge-warn', text: 'Sınırlı destek (WAWASAKY)' }));
    if (!view.stored) badges.append(el('span', { cls: 'badge badge-err', text: 'Kaydedilemedi' }));

    const lineage = $('res-lineage');
    if (view.parent_id) {
      lineage.textContent = `Filtre değişikliğiyle yeniden hesaplandı. Önceki tahmin (${view.parent_id.slice(0, 8)}) değiştirilmeden saklandı.`;
      show(lineage, true);
    } else if (!view.stored) {
      lineage.textContent = 'Bu sonuç sunucuda kaydedilemedi; geri bildirim, dışa aktarma ve filtreyle yeniden hesaplama kapalı.';
      show(lineage, true);
    } else {
      show(lineage, false);
    }
    const replay = $('res-replay');
    replay.textContent = view.replay_note ? `${view.replay_note} ${view.recorded_outcome_note || ''}` : '';
    show(replay, Boolean(view.replay_note));

    const rec = $('res-recency');
    clear(rec);
    const sources = (view.data_recency && view.data_recency.sources) || {};
    for (const [name, s] of Object.entries(sources)) {
      const text = `${SOURCE_TR[name] || name} ${s.last_day} tarihinde bitiyor` + (s.days_after_last_day > 0 ? ` (${s.days_after_last_day} gün önce)` : '');
      rec.append(el('li', { cls: s.stale ? 'is-stale' : '', text }));
    }
    warningList($('res-warnings'), view.warnings);

    renderFirms(view);
    renderPrice(view);
    renderMeta(view);
    $('btn-apply-filters').disabled = state.busy || !view.stored;
  }

  function stateBox(title, message, isError) {
    return el('div', { cls: 'empty-box' + (isError ? ' is-error' : '') }, el('strong', { text: title }), message || '');
  }

  function firmFeedbackMap(view) {
    const map = new Map();
    for (const f of view.feedback || []) if (f.kind === 'firm') map.set(f.firma_key, f);
    return map;
  }

  function feedbackText(f) {
    const verdict = f.verdict === 'uygun' ? 'Uygun' : 'Uygun değil';
    const reasons = (state.options && state.options.firm_reasons) || {};
    const reason = f.reason_code && reasons[f.reason_code] ? ` (${reasons[f.reason_code]})` : '';
    return `Geri bildiriminiz: ${verdict}${reason}`;
  }

  function renderFirms(view) {
    const a = view.model_a || {};
    const body = $('firms-body');
    clear(body);
    clear($('firms-notes'));
    warningList($('firms-warnings'), a.warnings);
    const csv = $('btn-csv');
    show($('firms-excluded'), false);
    show($('firms-bulk'), false);
    state.marked.clear();
    updateBulkBar();
    $('firms-summary').textContent = '';
    if (a.status !== 'ok') {
      show(csv, false);
      const title = a.status === 'unavailable' ? 'Firma listesi şu anda kullanılamıyor' : 'Firma listesi üretilemedi';
      body.append(stateBox(title, (a.message || '') + (a.error_id ? ` (kayıt no ${a.error_id})` : ''), a.status === 'error'));
      return;
    }
    const s = a.filter_summary || {};
    const reasonParts = Object.entries(s.excluded_by_reason || {}).filter(([, n]) => n > 0).map(([r, n]) => `${REASON_TR[r] || r}: ${fmtInt.format(n)}`);
    $('firms-summary').textContent = `Tahmin tarihindeki firma havuzu ${fmtInt.format(s.pool_size || 0)} firma; filtre sonrası uygun ${fmtInt.format(s.n_eligible || 0)}; listede ${fmtInt.format(s.n_selected || 0)}.`
      + (reasonParts.length ? ` Filtreyle çıkarılan — ${reasonParts.join(', ')}.` : ' Filtreyle çıkarılan firma yok.')
      + ((s.exclusion_keys_not_in_pool || []).length ? ` Dışladığınız ${s.exclusion_keys_not_in_pool.length} firma bu tarihte havuzda değildi.` : '');

    csv.href = `/api/predictions/${encodeURIComponent(view.prediction_id)}/call-list.csv`;
    show(csv, Boolean(view.stored) && (a.call_list || []).length > 0);

    if (!(a.call_list || []).length) {
      body.append(stateBox('Listede firma kalmadı', 'Filtrelerden sonra uygun firma yok. Şehir veya faaliyet filtresini gevşetin ya da dışlamaları kaldırın.'));
    } else {
      const fb = firmFeedbackMap(view);
      const ol = el('ol', { cls: 'firm-list' });
      for (const e of a.call_list) ol.append(firmRow(view, e, fb.get(e.firma_key)));
      body.append(ol);
      show($('firms-bulk'), Boolean(view.stored));
    }

    const ahead = a.excluded_ahead || [];
    if (ahead.length) {
      $('firms-excluded-title').textContent = `Listenin üstünde filtreyle çıkarılan ${fmtInt.format(a.n_excluded_ahead)} firma` + (a.n_excluded_ahead > ahead.length ? ` (ilk ${ahead.length} gösteriliyor)` : '');
      const ul = $('firms-excluded-list');
      clear(ul);
      for (const x of ahead) {
        const reasons = (x.reasons || []).map((r) => REASON_TR[r] || r).join(', ');
        ul.append(el('li', { text: `${x.firma_adi || x.firma_key} — model sırası ${x.rank_original} — ${reasons}` }));
      }
      show($('firms-excluded'), true);
    }
    const notes = $('firms-notes');
    for (const t of [a.score_note, a.explanation_note, a.directory_note, a.contact_note]) if (t) notes.append(el('p', { text: t }));
  }

  function firmRow(view, e, existing) {
    const name = e.firma_adi || e.firma_key;
    const d = e.directory || {};
    const meta = [d.city, d.district].filter(Boolean).join(' / ');
    const type = d.detay_tip || d.tip;
    const snippets = e.snippets || [];
    const li = el('li', { cls: 'firm' });

    const mark = el('input', { attrs: { type: 'checkbox', 'aria-label': `${name} firmasını işaretle`, disabled: !view.stored } });
    mark.addEventListener('change', () => {
      if (mark.checked) state.marked.set(e.firma_key, name); else state.marked.delete(e.firma_key);
      updateBulkBar();
    });
    li.append(el('div', { cls: 'firm-rank', attrs: { 'aria-label': `Sıra ${e.rank}` } }, String(e.rank),
      e.rank_original !== e.rank ? el('small', { text: `model ${e.rank_original}` }) : null,
      el('label', { attrs: { title: 'Birlikte listeden çıkarmak için işaretle' } }, mark)));

    const main = el('div', { cls: 'firm-main' });
    main.append(el('div', { cls: 'firm-name', text: name }));
    const metaText = [meta || 'Şehir bilgisi yok', type].filter(Boolean).join(' · ') + (d.in_directory === false ? ' · üye dizininde yok' : '');
    main.append(el('div', { cls: 'firm-meta', text: metaText }));
    const ul = el('ul', { cls: 'firm-snippets' });
    for (const s of snippets.slice(0, 2)) ul.append(el('li', { text: s }));
    main.append(ul);
    if (snippets.length > 2) {
      const more = el('details', { cls: 'firm-more' }, el('summary', { text: 'Tüm geçmiş bilgisi' }));
      const ul2 = el('ul', { cls: 'firm-snippets' });
      for (const s of snippets.slice(2)) ul2.append(el('li', { text: s }));
      more.append(ul2);
      main.append(more);
    }
    li.append(main);

    const status = el('div', { cls: 'firm-status', text: existing ? feedbackText(existing) : '' });
    const error = el('div', { cls: 'firm-error' });
    const fbBox = el('div', { cls: 'firm-feedback' }, status, error);
    const canAct = Boolean(view.stored);

    const actions = el('div', { cls: 'firm-actions' });
    actions.append(el('button', {
      cls: 'btn btn-secondary', text: 'Uygun', attrs: { type: 'button', disabled: !canAct },
      on: { click: () => sendFirmFeedback(view, e, { verdict: 'uygun' }, status, error) },
    }));
    actions.append(el('button', {
      cls: 'btn btn-secondary', text: 'Uygun değil', attrs: { type: 'button', disabled: !canAct },
      on: { click: () => toggleNotSuitable(view, e, fbBox, status, error) },
    }));
    actions.append(el('button', {
      cls: 'btn btn-quiet', text: 'Listeden çıkar', attrs: { type: 'button', disabled: !canAct || state.busy, 'data-exclude': '1', title: 'Firmayı dışla ve listeyi yeniden oluştur' },
      on: { click: () => excludeFirms([[e.firma_key, name]]) },
    }));
    li.append(actions);
    li.append(fbBox);
    return li;
  }

  function toggleNotSuitable(view, e, box, status, error) {
    const open = box.querySelector('.inline-form');
    if (open) { open.remove(); return; }
    const reasons = (state.options && state.options.firm_reasons) || {};
    const select = el('select', { attrs: { 'aria-label': 'Gerekçe' } }, el('option', { attrs: { value: '' }, text: 'Gerekçe seçin (isteğe bağlı)' }));
    for (const [code, label] of Object.entries(reasons)) select.append(el('option', { attrs: { value: code }, text: label }));
    const comment = el('textarea', { attrs: { rows: '2', maxlength: '1000', placeholder: 'Not (isteğe bağlı)', 'aria-label': 'Not' } });
    const form = el('form', { cls: 'inline-form', attrs: { novalidate: true } }, select, comment,
      el('div', { cls: 'row' },
        el('button', { cls: 'btn btn-secondary btn-small', text: 'Kaydet', attrs: { type: 'submit' } }),
        el('button', { cls: 'btn btn-quiet btn-small', text: 'Vazgeç', attrs: { type: 'button' }, on: { click: () => form.remove() } })));
    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const ok = await sendFirmFeedback(view, e, { verdict: 'uygun_degil', reason_code: select.value || null, comment: comment.value.trim() || null }, status, error);
      if (ok) form.remove();
    });
    box.append(form);
    select.focus();
  }

  async function sendFirmFeedback(view, e, payload, status, error) {
    error.textContent = '';
    try {
      const res = await api('POST', `/api/predictions/${encodeURIComponent(view.prediction_id)}/feedback`, { kind: 'firm', firma_key: e.firma_key, ...payload });
      status.textContent = feedbackText(res.feedback);
      (view.feedback = view.feedback || []).push(res.feedback);
      toast('Firma geri bildirimi kaydedildi.');
      return true;
    } catch (err) {
      error.textContent = 'Kaydedilemedi: ' + err.message;
      return false;
    }
  }

  function renderPrice(view) {
    const b = view.model_b || {};
    const body = $('price-body');
    clear(body);
    warningList($('price-warnings'), b.warnings);
    const form = $('price-feedback');
    if (b.status !== 'ok') {
      show(form, false);
      const titles = {
        fx_unavailable: 'Fiyat gösterilemiyor: uygun kur yok',
        insufficient_vehicle_fields: 'Fiyat için araç bilgisi eksik',
        model_not_loaded: 'Fiyat modeli şu anda kullanılamıyor',
      };
      const title = b.status === 'error' ? 'Fiyat tahmini üretilemedi' : (titles[b.reason] || 'Fiyat gösterilemiyor');
      body.append(stateBox(title, (b.message || '') + (b.error_id ? ` (kayıt no ${b.error_id})` : ''), b.status === 'error'));
      if (b.interval_note) body.append(el('p', { cls: 'fineprint', text: b.interval_note }));
      return;
    }
    const rounded = Math.round(b.price_tl / 1000) * 1000;
    body.append(el('p', { cls: 'price-value', text: `${fmtInt.format(rounded)} TL` }));
    const fx = b.fx || {};
    body.append(el('p', { cls: 'price-sub', text: `Yaklaşık ${fmtInt.format(Math.round(b.price_usd))} USD · USDTRY ${fx.rate ? fmtRate.format(fx.rate) : '?'} (${fx.bar_date || '?'} açılış kuru) · 1.000 TL'ye yuvarlandı` }));
    body.append(el('p', { cls: 'price-meaning', text: b.meaning }));

    const dl = el('dl', { cls: 'kv' });
    const add = (k, v) => { dl.append(el('dt', { text: k })); dl.append(el('dd', { text: v })); };
    add('Tahmin tarihi', view.prediction_date);
    const mb = view.identity && view.identity.models ? view.identity.models.model_b : null;
    add('Model', `${b.model}` + (mb && mb.fit_date_range && mb.fit_date_range.length ? ` (eğitim verisi ${mb.fit_date_range[0]} – ${mb.fit_date_range[1]})` : ''));
    add('Fiyat aralığı', b.interval_note || 'Gösterilmez.');
    for (const h of b.history_support_tr || []) add(`Önceki 365 günde ${h.label}`, `${fmtInt.format(h.n || 0)} satış`);
    const mf = view.vehicle ? view.vehicle.min_fiyat_clean : null;
    if (view.mode === 'current' && typeof mf === 'number' && mf > 0) add('Girilen Min.Fiyat', `${fmtInt.format(mf)} TL — yalnızca bilgi; bu tahmini etkilemez`);
    body.append(dl);

    // feedback
    state.priceVerdict = null;
    for (const btn of form.querySelectorAll('.seg')) btn.setAttribute('aria-pressed', 'false');
    $('price-suggest').value = '';
    $('price-suggest-preview').textContent = '';
    $('price-comment').value = '';
    show($('price-feedback-error'), false);
    const last = (view.feedback || []).filter((f) => f.kind === 'price').pop();
    $('price-feedback-status').textContent = last ? `Kayıtlı geri bildirim: ${priceVerdictTr(last.verdict)}` : '';
    $('btn-price-feedback').disabled = !view.stored;
    show(form, true);
  }

  function priceVerdictTr(v) { return { dusuk: 'Düşük', makul: 'Makul', yuksek: 'Yüksek' }[v] || v; }

  async function sendPriceFeedback(ev) {
    ev.preventDefault();
    const view = state.current;
    const err = $('price-feedback-error');
    show(err, false);
    if (!view || !view.stored) return;
    if (!state.priceVerdict) { err.textContent = 'Düşük, Makul veya Yüksek seçin.'; show(err, true); return; }
    const payload = { kind: 'price', verdict: state.priceVerdict };
    const r = parseNumber($('price-suggest').value, false);
    if (r.error) { err.textContent = 'Önerilen fiyat: ' + r.error + '.'; show(err, true); return; }
    if (!r.empty) payload.suggested_price_tl = r.value;
    const c = $('price-comment').value.trim();
    if (c) payload.comment = c;
    const btn = $('btn-price-feedback');
    btn.disabled = true;
    try {
      const res = await api('POST', `/api/predictions/${encodeURIComponent(view.prediction_id)}/feedback`, payload);
      (view.feedback = view.feedback || []).push(res.feedback);
      $('price-feedback-status').textContent = `Kaydedildi: ${priceVerdictTr(res.feedback.verdict)}`;
      toast('Fiyat geri bildirimi kaydedildi.');
    } catch (e) {
      err.textContent = 'Kaydedilemedi: ' + e.message;
      show(err, true);
    } finally {
      btn.disabled = false;
    }
  }

  function renderMeta(view) {
    const meta = $('res-meta');
    clear(meta);
    const id = view.identity || {};
    const item = (label, value) => el('span', null, `${label}: `, el('code', { text: value || '—' }));
    meta.append(item('Tahmin no', view.prediction_id));
    meta.append(item('Oluşturma (UTC)', view.created_at_utc));
    meta.append(item('Tahmin paketi', (id.bundle_id || '').slice(0, 16)));
    meta.append(item('Veri sürümü', id.dataset_version_id));
    meta.append(item('Filtre', `${id.filters_version || ''} / ${(id.rules_sha256 || '').slice(0, 12)}`));
    meta.append(item('Sunucu kodu', (id.serving_code_sha256 || '').slice(0, 16)));
  }

  // ------------------------------------------------------------------ tabs
  function setTab(tab) {
    state.tab = tab;
    const isNew = tab === 'new';
    $('tab-new').classList.toggle('is-active', isNew);
    $('tab-history').classList.toggle('is-active', !isNew);
    $('tab-new').setAttribute('aria-selected', String(isNew));
    $('tab-history').setAttribute('aria-selected', String(!isNew));
    show($('pane-new'), isNew);
    show($('pane-history'), !isNew);
    show($('btn-predict'), isNew);
    show($('btn-replay'), !isNew);
    showFormErrors([]);
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    $('login-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const user = $('login-user').value.trim();
      const pass = $('login-pass').value;
      const err = $('login-error');
      show(err, false);
      if (!user || !pass) { err.textContent = 'Kullanıcı adı ve parola gerekli.'; show(err, true); return; }
      $('btn-login').disabled = true;
      try {
        const s = await api('POST', '/api/login', { username: user, password: pass });
        $('login-pass').value = '';
        onLoggedIn(s);
      } catch (e) {
        err.textContent = e.message;
        show(err, true);
      } finally {
        $('btn-login').disabled = false;
      }
    });

    $('btn-logout').addEventListener('click', async () => {
      try { await api('POST', '/api/logout', {}); } catch (e) { /* session may already be gone */ }
      state.csrf = null;
      state.current = null;
      show($('result'), false);
      show($('result-empty'), true);
      showLogin('Çıkış yapıldı.');
    });

    $('btn-shutdown').addEventListener('click', async () => {
      if (!window.confirm('Uygulama tamamen kapatılsın mı? Sunucu durur; yeniden açmak için başlatma komutunu tekrar çalıştırmanız gerekir.')) return;
      try {
        await api('POST', '/api/shutdown', {});
      } catch (e) {
        toast('Kapatılamadı: ' + e.message);
        return;
      }
      clearInterval(state.statusTimer);
      state.statusTimer = null;
      document.body.replaceChildren(el('main', null, el('div', { cls: 'card state-card' },
        el('h2', { text: 'Uygulama kapatıldı' }),
        el('p', { cls: 'muted', text: 'Sunucu durduruldu. Bu sekmeyi kapatabilirsiniz. Yeniden kullanmak için başlatma komutunu tekrar çalıştırın.' }))));
      window.close();
    });
    $('btn-exclude-selected').addEventListener('click', () => excludeFirms(Array.from(state.marked.entries())));

    $('tab-new').addEventListener('click', () => setTab('new'));
    $('tab-history').addEventListener('click', () => setTab('history'));
    $('pane-new').addEventListener('submit', (ev) => { ev.preventDefault(); predictNew(); });
    $('btn-predict').addEventListener('click', predictNew);
    $('btn-replay').addEventListener('click', replayExample);
    $('btn-apply-filters').addEventListener('click', rerunWithFilters);
    $('btn-add-city').addEventListener('click', addCity);
    $('f-city').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); addCity(); } });
    $('btn-clear-excluded').addEventListener('click', () => { state.excluded.clear(); renderExcludedChips(); });
    $('f-date').addEventListener('change', () => { if (state.status) renderStatus(state.status); });

    for (const input of document.querySelectorAll('#pane-new [data-kind]')) {
      input.addEventListener('input', () => updatePreview(input));
    }
    for (const input of document.querySelectorAll('#pane-new [data-suggest]')) {
      input.addEventListener('change', () => updateHint(input));
    }
    const marka = document.querySelector('[data-field="Marka"]');
    marka.addEventListener('input', updateModelList);

    for (const btn of document.querySelectorAll('#price-feedback .seg')) {
      btn.addEventListener('click', () => {
        state.priceVerdict = btn.dataset.verdict;
        for (const b of document.querySelectorAll('#price-feedback .seg')) b.setAttribute('aria-pressed', String(b === btn));
      });
    }
    $('price-suggest').addEventListener('input', () => {
      const r = parseNumber($('price-suggest').value, false);
      const p = $('price-suggest-preview');
      p.classList.toggle('is-error', Boolean(r.error));
      p.textContent = r.empty ? '' : r.error ? r.error : `= ${fmtInt.format(r.value)} TL`;
    });
    $('price-feedback').addEventListener('submit', sendPriceFeedback);
    renderExcludedChips();
  }

  async function boot() {
    wire();
    try {
      const s = await api('GET', '/api/session');
      if (s.authenticated) onLoggedIn(s);
      else showLogin('');
    } catch (e) {
      showLogin(e.message);
    }
  }

  document.addEventListener('DOMContentLoaded', boot);
})();

"""
bundle.py  —  Phase 7: the frozen inference bundle, loaded once per server process.

A bundle binds ONE version of every input the two models need: the cleaned history
(Phase 1–3 outputs), the participation export, the member directory, the FX snapshot and
the two frozen model artifacts. Everything is read and indexed once at load; a prediction
never rebuilds history, trains, evaluates or touches the network. Nothing happens at
import time.

PATHS. BundleConfig takes an explicit project root; every file is a path relative to it
with a default naming the current accepted file EXACTLY (no glob, no "newest file"
guess). A missing file is a clear initialization error naming the logical input.

PARTIAL LOADING. The shared inputs (clean_sales, member directory) must load, or
load_bundle raises BundleError. Model A inputs (bids, positives, registry, participation,
internal lookup, ranker) and Model B inputs (FX, lgbm_noR) load independently: a failure
disables only that model, and the bundle records a sanitized reason (logical input names
only: no server path, no traceback) in components; the full error goes to the server log.

IDENTITY. bundle_id = SHA-256 over
  inputs    the SHA-256 of every loaded data / model file, by logical name;
  sources   the SHA-256 of every source file that shapes a prediction (SOURCE_FILES: this
            module and predict.py, read by PATH so no circular import is needed; the
            Phase 1-5 modules they call, transitively; common/* helpers);
  rules     stale_after_days, filters_version, the FX rule and lag, MODEL_SOURCES;
  runtime   Python and the pandas / numpy / LightGBM / pyarrow / openpyxl versions.
Two bundles with the same id therefore load the same data and models with the same code,
rules and library versions. The id does not cover Phase-8 server code; a Phase-8 response
store must record its own code version next to bundle_id.

RECENCY. data_ranges holds each source's own first / last day ("sales", and "bids",
"participation" when Model A loads). Sources end on different days; MODEL_SOURCES says
which sources each model reads, so staleness is judged per source and per model.

MANIFEST (public_manifest). Logical input names, SHA-256 digests, source / runtime /
rule identity, dataset_version_id, per-source date ranges, model training ranges and the
FX rule — never server paths, raw rows or firm-level history.

PUBLIC API
  BundleConfig(root, ...)          frozen; paths relative to root
  load_bundle(config)              -> Bundle (raises BundleError if shared inputs fail)
  cached_bundle(config)            -> Bundle, loaded once per config per process
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from phase1_preparation.clean_sales import FROZEN_SALES_COLUMNS
from phase2_labels import participation as _participation
from phase5_pricing import fx_rates as _fx_rates
from phase5_pricing import price_features as _price_features
from phase5_pricing import target_population as _target_population
from phase7_inference import filters as _filters
from phase7_inference import model_a as _model_a
from phase7_inference import model_b as _model_b
from phase7_inference import vehicle_input as _vehicle_input

log = logging.getLogger(__name__)

FORMAT = "phase7-bundle-v2"
SHEET = "Rapor"
MEMBER_DIRECTORY_COLUMNS = ("Firma Ünvanı", "Detay Tip")      # io_load.REQUIRED_COLUMNS["uye"]

# Every source file that shapes a prediction, relative to src/: this package (bundle.py
# and predict.py included), the Phase 1-5 modules it calls and their transitive imports.
# Hashed by path, so predict.py needs no import here. Extend when an import is added.
_SRC = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "common/__init__.py", "common/normalize.py", "common/entity_resolution.py",
    "common/anomaly_flags.py",
    "phase1_preparation/__init__.py", "phase1_preparation/clean_sales.py",
    "phase2_labels/__init__.py", "phase2_labels/participation.py",
    "phase3_firm_universe/candidate_pool.py",
    "phase4_modeling/features.py", "phase4_modeling/query_universe.py",
    "phase4_modeling/ranker.py", "phase4_modeling/evaluation.py",
    "phase5_pricing/fx_rates.py", "phase5_pricing/price_features.py",
    "phase5_pricing/price_model.py", "phase5_pricing/target_population.py",
    "phase7_inference/__init__.py", "phase7_inference/vehicle_input.py",
    "phase7_inference/filters.py", "phase7_inference/model_a.py",
    "phase7_inference/model_b.py", "phase7_inference/bundle.py",
    "phase7_inference/predict.py",
)
RUNTIME_PACKAGES = ("pandas", "numpy", "lightgbm", "pyarrow", "openpyxl")

# Which history sources each model reads (data_ranges keys).
MODEL_SOURCES = {"model_a": ("sales", "bids", "participation"), "model_b": ("sales",)}


class BundleError(RuntimeError):
    """The bundle cannot serve at all (shared inputs failed)."""


@dataclass(frozen=True)
class BundleConfig:
    root: Path
    clean_sales: str = "Datasets/interim/phase1_preparation/clean_sales.parquet"
    clean_reel_db: str = "Datasets/interim/phase1_preparation/clean_reel_db.parquet"
    labeled_candidates: str = "Datasets/interim/phase2_labels/labeled_candidates.parquet"
    firm_registry: str = "Datasets/interim/phase3_firm_universe/firm_registry.parquet"
    internal_lookup: str = "Datasets/lookups/internal_firm_lookup.csv"
    aa_katilim: str = "Datasets/raw/2501-260604 AA Katılım Raporu.xlsx"
    member_directory: str = "Datasets/raw/Üye-Kurumsal Raporu_2026-06-07.xlsx"
    fx_file: str = ("Datasets/external/fx/"
                    "USDTRY_yahoo_daily_2024-11-01_2026-10-02_fetched20261003T074901.csv")
    model_a: str = "workflow/runs/phase4_experiment_build/results/final_model.json"
    model_b: str = "workflow/runs/phase5_experiment_build/results/models/lgbm_noR.json"
    stale_after_days: int = 30


@dataclass(frozen=True)
class Bundle:
    config: BundleConfig
    manifest: dict                      # public, JSON-safe; see public_manifest
    clean_sales: pd.DataFrame
    member_directory: pd.DataFrame
    data_ranges: dict                   # source -> (first day, last day) Timestamps
    model_a: _model_a.ModelA | None
    model_b: _model_b.ModelB | None
    component_errors: dict = field(default_factory=dict)   # name -> sanitized message
    # clean_sales positions in the Phase-5 price population (build_price_population over
    # this clean_sales, as Model B's history); None when Model B is unavailable.
    price_population_rows: frozenset | None = None

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def history_end(self) -> pd.Timestamp:
        """Latest last day over ALL sources. Sources end on different days: never present
        this as every source's end; use data_ranges (and MODEL_SOURCES) per source."""
        return max(end for _, end in self.data_ranges.values())

    def public_manifest(self) -> dict:
        return json.loads(json.dumps(self.manifest))


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(config: BundleConfig, name: str) -> Path:
    """The file of logical input `name`. Errors name the input only; the path is logged."""
    rel = getattr(config, name)
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise BundleError(f"config.{name} must be a non-empty path relative to config.root")
    path = Path(config.root) / rel
    if not path.is_file():
        log.error("phase7 bundle: input %s not found at %s", name, path)
        raise BundleError(f"input {name!r} is missing; see server log")
    return path


def _span(dates, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.to_datetime(pd.Series(dates)).dropna()
    if d.empty:
        raise BundleError(f"input {name!r} has no dated rows")
    return d.min().normalize(), d.max().normalize()


def _day(ts) -> str | None:
    return None if ts is None else str(pd.Timestamp(ts).date())


def _source_hashes() -> dict:
    out = {}
    for rel in SOURCE_FILES:
        path = _SRC / rel
        if not path.is_file():
            log.error("phase7 bundle: source file %s not found under %s", rel, _SRC)
            raise BundleError(f"source file {rel} is missing; the bundle cannot be identified")
        out[rel] = _sha256_file(path)
    return dict(sorted(out.items()))


def _runtime_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in RUNTIME_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _rules(config: BundleConfig) -> dict:
    return {"stale_after_days": config.stale_after_days,
            "filters_version": _filters.FILTER_VERSION,
            "fx_rule": _price_features.FX_RULE,
            "max_fx_lag_days": _price_features.MAX_FX_LAG_DAYS,
            "model_sources": {k: list(v) for k, v in MODEL_SOURCES.items()}}


def _component(name: str, errors: dict, load):
    """Run one model's loader; a failure disables only that model (reported as
    unavailable in the manifest, never as ready)."""
    try:
        return load()
    except Exception as exc:                            # noqa: BLE001 — isolated per model
        log.exception("phase7 bundle: %s failed to load", name)
        errors[name] = (str(exc) if isinstance(exc, BundleError)
                        else f"{name} could not be loaded ({type(exc).__name__}); see server log")
        return None


def _read_member_directory(path: Path) -> pd.DataFrame:
    uye = pd.read_excel(path, sheet_name=SHEET)
    missing = [c for c in MEMBER_DIRECTORY_COLUMNS if c not in uye.columns]
    if missing:
        raise BundleError(f"member directory lacks columns {missing}")
    return uye


def _read_participation(path: Path, lookup: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=SHEET)
    need = [_participation.AA_KODU_RAW, _participation.AA_TARIHI_RAW, _participation.FIRMA_RAW]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise BundleError(f"AA Katılım export lacks columns {missing}")
    part, _ = _participation.clean_participation(raw, lookup)
    return part


# ----------------------------------------------------------------------------
# Loading.
# ----------------------------------------------------------------------------
def load_bundle(config: BundleConfig) -> Bundle:
    """Read, verify and index every input once. See the module docstring."""
    if not isinstance(config, BundleConfig):
        raise TypeError("[bundle] config must be a BundleConfig")
    if not Path(config.root).is_dir():
        raise BundleError("config.root is not a directory")
    if isinstance(config.stale_after_days, bool) or not isinstance(config.stale_after_days, int) \
            or config.stale_after_days < 0:
        raise BundleError("config.stale_after_days must be an int >= 0")

    sources = _source_hashes()                          # before any data: fail fast
    hashes: dict[str, str] = {}
    errors: dict[str, str] = {}
    ranges: dict[str, tuple] = {}
    price_rows: dict[str, frozenset] = {}

    # --- shared inputs: required ---
    cs_path, uye_path = _resolve(config, "clean_sales"), _resolve(config, "member_directory")
    hashes["clean_sales"] = _sha256_file(cs_path)
    hashes["member_directory"] = _sha256_file(uye_path)
    clean_sales = pd.read_parquet(cs_path).reset_index(drop=True)
    if list(clean_sales.columns) != FROZEN_SALES_COLUMNS:
        raise BundleError("clean_sales does not have FROZEN_SALES_COLUMNS in order")
    versions = clean_sales["dataset_version_id"].astype("string").dropna().unique()
    if len(versions) != 1:
        raise BundleError("clean_sales must carry exactly one dataset_version_id")
    if clean_sales["split"].astype("string").fillna("").eq(_vehicle_input.SPLIT).any():
        raise BundleError("clean_sales holds inference rows; load the historical export")
    member_directory = _read_member_directory(uye_path)
    ranges["sales"] = _span(clean_sales["aa_tarihi"], "clean_sales")

    # --- Model A ---
    def load_a():
        paths = {n: _resolve(config, n) for n in ("clean_reel_db", "labeled_candidates",
                                                  "firm_registry", "internal_lookup",
                                                  "aa_katilim", "model_a")}
        digests = {n: _sha256_file(p) for n, p in paths.items()}
        import pyarrow.parquet as pq
        bids = pd.read_parquet(paths["clean_reel_db"])
        positives = pq.read_table(paths["labeled_candidates"],
                                  columns=["query_id", "firma_key", "grade"],
                                  filters=[("grade", ">=", 1)]).to_pandas()
        registry = pd.read_parquet(paths["firm_registry"])
        part = _read_participation(paths["aa_katilim"], pd.read_csv(paths["internal_lookup"]))
        model = _model_a.load_model_a(clean_sales=clean_sales, bids=bids, participation=part,
                                      registry=registry, positives=positives,
                                      model_path=paths["model_a"])
        bids_range = _span(bids["aa_tarihi"], "clean_reel_db")
        part_range = _span(part["aa_tarihi"], "aa_katilim")
        hashes.update(digests)
        ranges["bids"], ranges["participation"] = bids_range, part_range
        return model

    # --- Model B ---
    def load_b():
        paths = {n: _resolve(config, n) for n in ("fx_file", "model_b")}
        digests = {n: _sha256_file(p) for n, p in paths.items()}
        fx = _fx_rates.load_fx(paths["fx_file"])
        model = _model_b.load_model_b(clean_sales=clean_sales, fx=fx,
                                      model_path=paths["model_b"])
        # The same population call load_model_b makes (model_b does not expose it): done
        # once here so a replay can state real price-population membership.
        pop, _ = _target_population.build_price_population(clean_sales)
        price_rows["rows"] = frozenset(int(i) for i in pop["source_row"].to_numpy())
        hashes.update(digests)
        return model

    model_a = _component("model_a", errors, load_a)
    model_b = _component("model_b", errors, load_b)
    if model_a is None and model_b is None:
        raise BundleError(f"neither model could be loaded: {errors}")

    manifest = _manifest(config, hashes, sources, str(versions[0]), ranges, model_a, model_b,
                         errors)
    log.info("phase7 bundle %s loaded (model_a=%s, model_b=%s)", manifest["bundle_id"],
             model_a is not None, model_b is not None)
    return Bundle(config=config, manifest=manifest, clean_sales=clean_sales,
                  member_directory=member_directory, data_ranges=ranges,
                  model_a=model_a, model_b=model_b, component_errors=dict(errors),
                  price_population_rows=price_rows.get("rows") if model_b is not None else None)


def _manifest(config, hashes, sources, version, ranges, model_a, model_b, errors) -> dict:
    runtime, rules = _runtime_versions(), _rules(config)
    identity = {"format": FORMAT, "inputs": dict(sorted(hashes.items())), "sources": sources,
                "rules": rules, "runtime": runtime}
    bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    fx_info = None
    if model_b is not None:
        bars = model_b.fx.loc[pd.to_numeric(model_b.fx["open"], errors="coerce").notna(),
                              "bar_date"]
        fx_info = {"bar_range": [_day(bars.min()), _day(bars.max())],
                   "rule": _price_features.FX_RULE,
                   "max_lag_days": _price_features.MAX_FX_LAG_DAYS,
                   "note": "Snapshot only; inference never fetches rates."}
    return {
        "format": FORMAT,
        "bundle_id": bundle_id,
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version_id": version,
        "input_sha256": dict(sorted(hashes.items())),
        "source_sha256": sources,
        "runtime_versions": runtime,
        "rules": rules,
        "data_ranges": {k: [_day(a), _day(b)] for k, (a, b) in sorted(ranges.items())},
        "fx": fx_info,
        "models": {
            "model_a": None if model_a is None else {
                "artifact": "phase4 final_model (LambdaRank)",
                "training_periods": list(model_a.training_periods),
                "training_data_end": _day(model_a.train_end),
                "score_meaning": _model_a.SCORE_NOTE},
            "model_b": None if model_b is None else {
                "artifact": _model_b.MODEL_NAME,
                "fit_date_range": list(model_b.fit_date_range or []),
                "label": _model_b.LABEL,
                "interval": None,
                "interval_note": "Bu model için geçerli kalibrasyon yok; lgbm_R aralıkları "
                                 "aktarılmaz."},
        },
        "components": {"model_a": "ready" if model_a is not None else "unavailable",
                       "model_b": "ready" if model_b is not None else "unavailable",
                       "errors": dict(errors)},
        "filters_version": _filters.FILTER_VERSION,
        "stale_after_days": config.stale_after_days,
    }


_CACHE: dict = {}
_LOCK = threading.Lock()


def cached_bundle(config: BundleConfig) -> Bundle:
    """load_bundle once per (frozen) config per process; later calls reuse it."""
    key = json.dumps({k: str(v) for k, v in asdict(config).items()}, sort_keys=True)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = load_bundle(config)
        return _CACHE[key]

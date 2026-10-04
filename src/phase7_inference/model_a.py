"""
model_a.py  —  Phase 7: the frozen Model A ranker served on one vehicle.

"Whom should we call for this vehicle?" For a vehicle on its prediction date this module
builds the accepted point-in-time candidate pool (candidate_pool_as_of, exclude=None as in
every accepted run), attaches the accepted Phase-4 features (features.attach_features)
and scores the WHOLE pool with the frozen ranker. Filtering and the top-N cut happen
afterwards (filters.py), so an excluded firm is backfilled from the scored pool. It
trains nothing and evaluates nothing.

STATE (ModelA, built once per bundle): the EventIndex and FeatureIndex over the whole
cleaned history and the loaded ranker. A new vehicle is attached with
features.extend_feature_index (historical firm histories reused, the cached index never
modified); a historical vehicle is scored with the original index at its own date.
Evidence is admitted strictly before the prediction date, exactly as in training.

OUTPUT is the full ranking in model order: score (a ranking score, NOT a purchase
probability), rank_original (ties: firma_key ascending, the Phase-4 metric rule), the
point-in-time features used for factual history snippets, the pool lane and the firm's
current member-directory attributes. The registry has no phone or e-mail field, so no
contact is ever returned. Directory city / type are a present-day snapshot, not proof of
the firm's attributes on a past date.

PUBLIC API
  load_model_a(*, clean_sales, bids, participation, registry, positives, model_path)
                                              -> ModelA
  rank_pool(model, rows, *, new_vehicle)      -> DataFrame (full pool, model order)
  history_snippets(row, *, marka)             -> list[str] (Turkish, factual; a "no recorded
                                                 bid" line only when total_bids_prior = 0)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phase3_firm_universe.candidate_pool import (EventIndex, build_event_index,
                                                 candidate_pool_as_of)
from phase4_modeling import features as F
from phase4_modeling import ranker as R
from phase4_modeling.query_universe import SPLIT_BOUNDARY, VALIDATION_START

REGISTRY_ATTRS = ["firma_adi_display", "sehir", "ilce", "detay_tip", "tip", "profil",
                  "in_directory", "has_id_collision"]
POOL_META = ["lane", "known_via"]
SCORE_NOTE = ("Sıralama skorudur; satın alma olasılığı değildir. Yalnızca aynı listedeki "
              "firmaları sıralamak için anlamlıdır.")


def _fail(msg: str) -> None:
    raise ValueError(f"[model_a] {msg}")


@dataclass(frozen=True)
class ModelA:
    ranker: R.Ranker
    event_index: EventIndex
    feature_index: F.FeatureIndex
    firms: pd.DataFrame                 # REGISTRY_ATTRS indexed by firma_key
    training_periods: tuple[str, ...]
    train_end: pd.Timestamp             # last day of the frozen model's training data


def load_model_a(*, clean_sales: pd.DataFrame, bids: pd.DataFrame,
                 participation: pd.DataFrame, registry: pd.DataFrame,
                 positives: pd.DataFrame, model_path) -> ModelA:
    """Indexes over the whole cleaned history (as pipeline/run_phase4.build_inputs) and
    the verified frozen ranker. No ledger, key stream or training set is built."""
    ranker = R.load_ranker(model_path)
    ev = build_event_index(bids, participation, registry)
    fi = F.build_feature_index(bids, positives, clean_sales)
    missing = [c for c in ["firma_key"] + REGISTRY_ATTRS if c not in registry.columns]
    if missing:
        _fail(f"registry lacks {missing}")
    firms = registry[["firma_key"] + REGISTRY_ATTRS].copy()
    firms["firma_key"] = firms["firma_key"].astype("string")
    firms = firms.set_index("firma_key")
    periods = tuple(ranker.provenance["periods"])
    end = (SPLIT_BOUNDARY if "validation" in periods else VALIDATION_START) - pd.Timedelta(days=1)
    return ModelA(ranker=ranker, event_index=ev, feature_index=fi, firms=firms,
                  training_periods=periods, train_end=end)


def rank_pool(model: ModelA, rows: pd.DataFrame, *, new_vehicle: bool) -> pd.DataFrame:
    """
    Score the full point-in-time pool for ONE vehicle row (FROZEN_SALES_COLUMNS, from
    vehicle_input.build_inference_vehicles when new_vehicle, else from_history). Returns
    one row per pool firm in model order with rank_original 1..n.
    """
    if not isinstance(model, ModelA):
        raise TypeError("[model_a] model must be a ModelA")
    if len(rows) != 1:
        _fail("rank_pool scores exactly one vehicle")
    fi = (F.extend_feature_index(model.feature_index, rows[F.VEHICLE_COLUMNS])
          if new_vehicle else model.feature_index)
    qid = str(rows["plaka"].iloc[0]) + "::" + str(rows["aa_kodu"].iloc[0])
    day = pd.Timestamp(rows["aa_tarihi"].iloc[0])
    pool = candidate_pool_as_of(model.event_index, day)
    n = len(pool)
    keys = pd.DataFrame({"query_id": pd.array([qid] * n, dtype="string"),
                         "aa_tarihi": np.full(n, day.to_datetime64(), dtype="datetime64[ns]"),
                         "firma_key": pool["firma_key"].astype("string").to_numpy()})
    feats = F.attach_features(fi, model.event_index, keys)
    score = R.score(model.ranker, feats)
    feats["score"] = score
    for c in POOL_META:
        feats[c] = pool[c].astype(object).to_numpy()
    order = np.lexsort((feats["firma_key"].astype(str).to_numpy(), -score))
    out = feats.iloc[order].reset_index(drop=True)
    out["rank_original"] = np.arange(1, n + 1)
    attrs = model.firms.reindex(out["firma_key"].astype("string"))
    for c in REGISTRY_ATTRS:
        out[c] = attrs[c].to_numpy()
    return out


def _int(v):
    return None if pd.isna(v) else int(v)


def history_snippets(row: pd.Series, *, marka) -> list[str]:
    """Factual Turkish lines from the point-in-time features of one ranked row. They
    describe recorded history before the prediction date, not the model's reasons.

    Two counts differ (features.py CUTOFF): firm_total_bids_prior counts single bids
    admitted before the date (candidate_pool); firm_n_cars_bid_prior / firm_n_wins_prior /
    aff_marka_n_prior count only cars whose bids by this firm were ALL admitted before it.
    "No recorded bid" is therefore claimed only from total_bids_prior = 0, and the car /
    win / marka lines are qualified as complete-engagement counts."""
    out = []
    total = _int(row["firm_total_bids_prior"])
    cars, wins = int(row["firm_n_cars_bid_prior"]), int(row["firm_n_wins_prior"])
    if not total:
        out.append("Tahmin tarihinden önce kayıtlı teklifi yok.")
    else:
        out.append(f"Tahmin tarihinden önce {total} kayıtlı teklif.")
        if cars:
            out.append(f"Teklifleri tamamen bu tarihten önce kayda geçmiş {cars} araçta teklif "
                       f"vermiş; bu araçlarda kayıtlı ihale kazanımı {wins}.")
        else:
            out.append("Teklifleri tamamen bu tarihten önce kayda geçmiş bir araç yok; araç ve "
                       "kazanım sayıları bu nedenle 0.")
    same = _int(row["aff_marka_n_prior"])
    if same and cars and isinstance(marka, str) and marka.strip():
        out.append(f"Bu {cars} aracın {same} tanesi {marka} markasında.")
    last = _int(row["firm_days_since_last_bid"])
    if last is not None:
        out.append(f"Son teklifi tahmin tarihinden {last} gün önce.")
    part = _int(row["firm_days_since_last_participation"])
    if part is not None:
        out.append(f"Son ihale katılımı tahmin tarihinden {part} gün önce.")
    if row["lane"] == "C":
        out.append("Yalnızca üye dizini kaydıyla biliniyor; tahmin tarihinden önce kayıtlı "
                   "teklif veya katılımı yok.")
    return out

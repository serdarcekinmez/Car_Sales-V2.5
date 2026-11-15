"""Daily FX refresh shared by FastAPI and Streamlit; never modifies the data bundle."""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase5_pricing import fx_rates as FX

log = logging.getLogger(__name__)


class FxRefresh:
    """One owner per DemoRuntime. Successful checks persist across restarts.

    Only append new dates (or fill missing opens). Historical opens are preserved so
    refreshing today's rate cannot revise historical model features. A new snapshot
    becomes visible only after all its files and the atomic checkpoint are written.
    """

    def __init__(self, root: Path, store_path: Path):
        self.root = Path(root).resolve()
        key = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        self.directory = Path(store_path).resolve().parent / "fx-cache" / key
        self.path = None
        self.checked_day = None
        self.message = None
        self.seed_sha = None
        self.initialized = False

    def _initialize(self):
        try:
            seed = FX.latest_fx_snapshot(self.root / "Datasets/external/fx")
            self.seed_sha = hashlib.sha256(seed.read_bytes()).hexdigest()
            self.path = seed
        except FileNotFoundError:
            pass
        try:
            marker = json.loads((self.directory / "current.json").read_text())
            candidate = (self.directory / marker["snapshot"]).resolve()
            if (marker["seed_sha256"] == self.seed_sha
                    and candidate.is_relative_to(self.directory.resolve())
                    and hashlib.sha256(candidate.read_bytes()).hexdigest() == marker["sha256"]):
                FX.load_fx(candidate)
                self.path, self.checked_day = candidate, marker["checked_day"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.initialized = True

    def refresh(self, today=None) -> Path | None:
        """Return the last good snapshot even when Yahoo or disk writes fail.

        Failed attempts are retried on the runtime's next hourly tick. Weekends with
        no new bars count as successful checks, not failures or fabricated rates.
        """
        day = pd.Timestamp(today or datetime.now(timezone.utc).date()).normalize()
        try:
            if not self.initialized:
                self._initialize()
            if self.checked_day == str(day.date()):
                return self.path
            old = FX.load_fx(self.path) if self.path else pd.DataFrame(columns=FX.COLUMNS)
            start = old["bar_date"].max() if len(old) else pd.Timestamp("2024-11-01")
            if start > day:
                raise ValueError("FX snapshot contains future dates")
            fetched = FX.fetch_usdtry(start, day)
            new = fetched.bars.copy()
            new["bar_date"] = pd.to_datetime(new["bar_date"]).dt.normalize()
            opens = pd.to_numeric(new["open"], errors="coerce")
            if (new["bar_date"].isna().any() or new["bar_date"].duplicated().any()
                    or (new["bar_date"] > day).any() or (new["bar_date"] < start).any()
                    or (~np.isfinite(opens[opens.notna()])).any()
                    or (opens.dropna() <= 0).any()):
                raise ValueError("Invalid FX response")
            new = new.loc[opens.notna()].copy()
            # Existing valid bars always win over revised provider history.
            valid_old = old.loc[pd.to_numeric(old["open"], errors="coerce").notna()]
            additions = new.loc[~new["bar_date"].isin(valid_old["bar_date"])]
            if old.empty and additions.empty:
                raise ValueError("No usable FX bars received")
            candidate = self.path
            self.directory.mkdir(parents=True, exist_ok=True)
            if len(additions) or candidate is None or candidate.parent == self.root / "Datasets/external/fx":
                retained = old.loc[~old["bar_date"].isin(additions["bar_date"])]
                parts = [frame for frame in (retained, additions) if not frame.empty]
                merged = pd.concat(parts, ignore_index=True).sort_values("bar_date")
                prov = {**fetched.provenance,
                        "covered": [str(merged["bar_date"].min().date()), str(merged["bar_date"].max().date())],
                        "rows": len(merged), "rows_missing_open": int(merged["open"].isna().sum()),
                        "previous_csv_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path else None,
                        "merge_rule": "preserve existing valid opens; append new dates or fill missing opens",
                        "note": "CSV combines previous snapshot with downloaded bars; raw JSON is incremental."}
                folder = Path(tempfile.mkdtemp(prefix="snapshot-", dir=self.directory))
                candidate = FX.save_fx_snapshot(FX.FxFetch(merged, fetched.raw, prov), folder)
                FX.load_fx(candidate)
            marker = {"snapshot": str(candidate.relative_to(self.directory)),
                      "seed_sha256": self.seed_sha,
                      "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                      "checked_day": str(day.date())}
            with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as fh:
                json.dump(marker, fh)
                temporary = Path(fh.name)
            temporary.replace(self.directory / "current.json")
            self.path, self.checked_day, self.message = candidate, str(day.date()), None
            log.info("FX checked through %s; snapshot %s", self.checked_day, candidate.name)
        except Exception:  # Network/disk/provider failures must not take the demo down.
            log.exception("FX refresh failed; keeping the previous snapshot")
            self.message = "Döviz kuru güncellenemedi; son kayıt kullanılıyor. Otomatik olarak tekrar denenecek."
        return self.path

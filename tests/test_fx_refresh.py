"""Offline regressions for daily updates, failure recovery and immutable predictions."""
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phase5_pricing import fx_rates as FX, price_features as PF
from phase8_demo.fx_refresh import FxRefresh
from phase8_demo import app as A
from phase7_inference import bundle as BD, model_b as MB


def fetched(days, opens):
    frame = pd.DataFrame({'bar_date': pd.to_datetime(days), 'open': opens})
    for col in ('high', 'low', 'close'):
        frame[col] = opens
    frame['bar_start_utc'] = frame['bar_date'].dt.strftime('%Y-%m-%dT00:00:00Z')
    return FX.FxFetch(frame[FX.COLUMNS], b'{"synthetic":true}', {
        'covered': [days[0], days[-1]], 'fetched_at_utc': '2026-11-15T12:00:00+00:00'})


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'bundle'
        self.seed = FX.save_fx_snapshot(fetched(['2026-10-08', '2026-10-09'], [40., 41.]),
                                        self.root / 'Datasets/external/fx')
        self.seed_bytes = self.seed.read_bytes()
        self.store = self.base / 'state/demo.sqlite3'
        self.service = FxRefresh(self.root, self.store)

    def test_update_preserves_history_and_reuses_cache_after_restart(self):
        result = fetched(['2026-10-09', '2026-11-13'], [999., 43.])
        with patch.object(FX, 'fetch_usdtry', return_value=result) as fetch:
            path = self.service.refresh('2026-11-15')
            self.assertEqual(FX.load_fx(path)['open'].tolist(), [40., 41., 43.])
            self.assertEqual(self.seed.read_bytes(), self.seed_bytes)
            self.assertFalse(path.is_relative_to(self.root))
            self.service.refresh('2026-11-15')
            restarted = FxRefresh(self.root, self.store)
            self.assertEqual(restarted.refresh('2026-11-15'), path)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(fetch.call_args.args, (pd.Timestamp('2026-10-09'), pd.Timestamp('2026-11-15')))
        rates = PF.asof_fx_rate(['2026-10-10', '2026-11-15'], FX.load_fx(path))
        self.assertEqual(rates['rate'].tolist(), [41., 43.])
        with self.assertRaises(ValueError):
            PF.asof_fx_rate(['2026-11-15'], FX.load_fx(self.seed))

    def test_network_failure_retries_and_preserves_last_good_cache(self):
        with patch.object(FX, 'fetch_usdtry', side_effect=OSError('offline')), self.assertLogs(level='ERROR'):
            self.assertEqual(self.service.refresh('2026-11-15'), self.seed)
        self.assertIsNone(self.service.checked_day)
        self.assertIsNotNone(self.service.message)
        with patch.object(FX, 'fetch_usdtry', return_value=fetched(['2026-11-13'], [43.])):
            path = self.service.refresh('2026-11-15')
        restarted = FxRefresh(self.root, self.store)
        with patch.object(FX, 'fetch_usdtry', side_effect=OSError('offline')), self.assertLogs(level='ERROR'):
            self.assertEqual(restarted.refresh('2026-11-16'), path)
        self.assertIsNone(self.service.message)

    def test_weekend_no_new_bar_and_next_day_recheck(self):
        with patch.object(FX, 'fetch_usdtry', return_value=fetched(['2026-11-13'], [43.])) as fetch:
            path = self.service.refresh('2026-11-14')
            self.assertEqual(self.service.refresh('2026-11-15'), path)
            self.assertEqual(fetch.call_count, 2)
        self.assertEqual(self.service.checked_day, '2026-11-15')
        self.assertEqual(FX.load_fx(path)['bar_date'].max(), pd.Timestamp('2026-11-13'))

    def test_invalid_provider_response_and_write_failure_do_not_publish(self):
        for bad in [fetched(['2026-11-13'], [-1.]), fetched(['2026-11-13'], [float('inf')]),
                    fetched(['2026-11-16'], [43.])]:
            with patch.object(FX, 'fetch_usdtry', return_value=bad), self.assertLogs(level='ERROR'):
                self.assertEqual(self.service.refresh('2026-11-15'), self.seed)
        with patch.object(FX, 'fetch_usdtry', return_value=fetched(['2026-11-13'], [43.])), \
                patch.object(FX, 'save_fx_snapshot', side_effect=PermissionError()), self.assertLogs(level='ERROR'):
            self.assertEqual(self.service.refresh('2026-11-15'), self.seed)
        self.assertFalse((self.service.directory / 'current.json').exists())
        self.assertEqual(self.seed.read_bytes(), self.seed_bytes)

    def test_no_seed_can_bootstrap(self):
        service = FxRefresh(self.base / 'empty', self.store)
        with patch.object(FX, 'fetch_usdtry', return_value=fetched(['2024-11-01', '2026-11-13'], [34., 43.])):
            self.assertIsNotNone(service.refresh('2026-11-15'))
        self.assertIsNone(service.message)

    def test_replaced_seed_invalidates_cached_snapshot(self):
        with patch.object(FX, 'fetch_usdtry', return_value=fetched(['2026-11-13'], [43.])):
            self.service.refresh('2026-11-15')
        self.seed.write_bytes(self.seed_bytes.replace(b'40.0', b'39.0'))
        restarted = FxRefresh(self.root, self.store)
        with patch.object(FX, 'fetch_usdtry', side_effect=OSError('offline')), self.assertLogs(level='ERROR'):
            self.assertEqual(restarted.refresh('2026-11-15'), self.seed)

    def test_bundle_activation_changes_identity_without_mutating_old_bundle(self):
        model_path = self.root / 'model.json'
        model_path.write_text('{}')
        config = BD.BundleConfig(root=self.root, model_b='model.json')
        model = SimpleNamespace(fx=FX.load_fx(self.seed), fit_date_range=None)
        hashes = {'fx_file': hashlib.sha256(self.seed_bytes).hexdigest()}
        manifest = BD._manifest(config, hashes, {}, 'test', {}, None, model, {})
        old = BD.Bundle(config, manifest, pd.DataFrame(), pd.DataFrame(), {}, None, model,
                        price_population_rows=frozenset())
        with patch.object(FX, 'fetch_usdtry', return_value=fetched(['2026-11-13'], [43.])):
            path = self.service.refresh('2026-11-15')
        with patch.object(MB, 'load_model_b', side_effect=lambda **kw: SimpleNamespace(fx=kw['fx'], fit_date_range=None)):
            new = BD.with_fx_snapshot(old, path)
        self.assertNotEqual(old.bundle_id, new.bundle_id)
        self.assertIs(new.clean_sales, old.clean_sales)
        self.assertEqual(old.manifest['fx']['bar_range'][1], '2026-10-09')
        self.assertEqual(new.manifest['fx']['bar_range'][1], '2026-11-13')
        self.assertEqual(MB.fx_status(new.model_b, pd.Timestamp('2026-11-15'))['rate'], 43.)
        self.assertIs(BD.with_fx_snapshot(new, path), new)

    def test_runtime_activates_updates_retains_old_on_failure_and_worker_stops(self):
        settings = SimpleNamespace(root=self.root, store_path=self.store)
        rt = A.DemoRuntime(settings, Mock(), 'test')
        old, updated = object(), object()
        rt.bundle = old
        with patch.object(rt.fx_refresh, 'refresh', return_value=self.seed), \
                patch.object(BD, 'with_fx_snapshot', return_value=updated):
            rt.refresh_fx()
        self.assertIs(rt.bundle, updated)
        with patch.object(rt.fx_refresh, 'refresh', return_value=self.seed), \
                patch.object(BD, 'with_fx_snapshot', side_effect=ValueError()), self.assertLogs(level='ERROR'):
            rt.refresh_fx()
        self.assertIs(rt.bundle, updated)
        self.assertIsNotNone(rt._fx_activation_message)
        with patch.object(rt.fx_refresh, 'refresh', return_value=self.seed), \
                patch.object(BD, 'with_fx_snapshot', return_value=updated):
            rt.refresh_fx()
        self.assertIsNone(rt._fx_activation_message)
        called = threading.Event()
        with patch.object(rt, 'load', side_effect=called.set):
            rt.start()
            self.assertTrue(called.wait(2))
            rt.stop()
            rt._refresh_thread.join(2)
            self.assertFalse(rt._refresh_thread.is_alive())

    def test_startup_load_and_periodic_worker_use_shared_refresh(self):
        rt = A.DemoRuntime(SimpleNamespace(root=self.root, store_path=self.store), Mock(), 'test')
        bundle = SimpleNamespace(bundle_id='synthetic')
        with patch.object(rt.fx_refresh, 'refresh', return_value=self.seed), \
                patch.object(BD, 'load_bundle', return_value=bundle) as load, \
                patch.object(A, 'build_options', return_value={}), \
                patch.object(A, 'build_examples', return_value=([], {})):
            rt.load()
        self.assertEqual(load.call_args.args[0].fx_snapshot, self.seed)
        self.assertIs(rt.bundle, bundle)
        self.assertEqual(rt.state, 'ready')
        rt._refresh_stop = Mock()
        rt._refresh_stop.wait.side_effect = [False, True]
        with patch.object(rt, 'refresh_fx') as refresh, patch.object(rt, 'load') as load:
            rt.start(initial_load=False)  # Same lifecycle used by cached Streamlit runtime.
            rt._refresh_thread.join(2)
            self.assertFalse(rt._refresh_thread.is_alive())
            refresh.assert_called_once()
            load.assert_not_called()
            rt._refresh_stop.wait.assert_called_with(3600)


if __name__ == '__main__':
    unittest.main()

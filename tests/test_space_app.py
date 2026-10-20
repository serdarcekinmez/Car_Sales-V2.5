"""Synthetic HTTP tests; optional real bundle load is local-only and opt-in."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
AVAILABLE = all(importlib.util.find_spec(m) for m in ('fastapi', 'httpx', 'pandas', 'lightgbm'))


@unittest.skipUnless(AVAILABLE, 'BLOCKED: install requirements-space.txt and httpx')
class AppTests(unittest.TestCase):
    def setUp(self):
        from phase8_demo import app, security
        self.app, self.security = app, security
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.env = {'GARAJ_DEMO_USERNAME': 'synthetic-test',
                    'GARAJ_DEMO_PASSWORD_HASH': security.hash_password('Synthetic-Test-123!'),
                    'GARAJ_DEMO_ROOT': str(self.base),
                    'GARAJ_DEMO_STORE': str(self.base / 'test.sqlite3'),
                    'GARAJ_DEMO_ALLOWED_HOSTS': 'testserver'}

    def test_settings_validation_and_hash_compatibility(self):
        self.app.settings_from_env(self.env)
        for key, value in [('GARAJ_DEMO_USERNAME', ''), ('GARAJ_DEMO_PASSWORD_HASH', 'invalid'),
                           ('GARAJ_DEMO_SESSION_HOURS', 'nan'), ('GARAJ_DEMO_SESSION_HOURS', 'bad'),
                           ('GARAJ_DEMO_EMAIL', 'bad'), ('GARAJ_DEMO_ROOT', str(self.base / 'absent'))]:
            with self.subTest(key=key), self.assertRaises(self.app.ConfigError):
                self.app.settings_from_env({**self.env, key: value})

    def test_login_logout_routes_headers_and_status(self):
        from fastapi.testclient import TestClient
        # No private data or model execution in HTTP security tests.
        with patch.object(self.app.DemoRuntime, 'load') as load:
            application = self.app.create_app(self.app.settings_from_env(self.env))
            with TestClient(application, base_url='https://testserver') as client:
                load.assert_called_once()
                page = client.get('/')
                self.assertEqual(page.status_code, 200)
                self.assertEqual(page.headers['x-frame-options'], 'DENY')
                self.assertEqual(client.get('/', headers={'host': 'evil.invalid'}).status_code, 400)
                for path in ('/api/status', '/api/options', '/api/predictions/abc/call-list.csv'):
                    self.assertEqual(client.get(path).status_code, 401)
                self.assertEqual(client.post('/api/predict', json={}).status_code, 401)
                for path in ('/Datasets/raw/test.xlsx', '/workflow/runs/model.json', '/static/../data'):
                    self.assertEqual(client.get(path).status_code, 404)
                self.assertEqual(client.post('/api/login', json={'username': 'synthetic-test', 'password': 'wrong'}).status_code, 401)
                login = client.post('/api/login', json={'username': 'synthetic-test', 'password': 'Synthetic-Test-123!'})
                self.assertEqual(login.status_code, 200)
                for flag in ('Secure', 'HttpOnly', 'SameSite=strict'):
                    self.assertIn(flag, login.headers['set-cookie'])
                self.assertEqual(client.get('/api/status').json()['state'], 'loading')
                self.assertEqual(client.post('/api/predict', json={}).status_code, 403)
                csrf = {self.security.CSRF_HEADER: login.json()['csrf_token'], 'origin': 'https://testserver'}
                self.assertEqual(client.post('/api/logout', json={}, headers={**csrf, 'origin': 'https://evil.invalid'}).status_code, 403)
                self.assertEqual(client.post('/api/logout', json={}, headers=csrf).status_code, 200)
                self.assertEqual(client.get('/api/status').status_code, 401)

    def test_space_bootstrap_rejects_settings_before_download(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
        import run_space
        for override in ({'GARAJ_DEMO_ALLOWED_HOSTS': '*'}, {'GARAJ_DEMO_ROOT': 'relative'},
                         {'GARAJ_DEMO_USERNAME': ''}, {'GARAJ_DEMO_STORE': str(self.base / 'inside.sqlite3')}):
            env = {**self.env, 'GARAJ_DEMO_STORE': str(self.base.parent / 'outside.sqlite3'), **override}
            with self.subTest(override=next(iter(override))), patch.dict(os.environ, env, clear=True), \
                    patch.object(run_space, 'download') as download:
                self.assertEqual(run_space.main(), 2)
                download.assert_not_called()

    def test_space_bootstrap_execs_existing_secure_runner(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
        import run_space
        env = {**self.env, 'GARAJ_DEMO_STORE': str(self.base.parent / 'outside.sqlite3'),
               'GARAJ_DEMO_DATA_TOKEN': 'synthetic', 'GARAJ_DEMO_DATA_REVISION': 'test-revision'}
        with patch.dict(os.environ, env, clear=True), patch.object(run_space, 'download') as download, \
                patch.object(run_space.os, 'execv') as execute:
            run_space.main()
            self.assertEqual(download.call_args.kwargs['revision'], 'test-revision')
            self.assertNotIn('GARAJ_DEMO_DATA_TOKEN', os.environ)
            self.assertEqual(execute.call_args.args[1][-6:],
                             ['serve', '--host', '0.0.0.0', '--port', '8000', '--behind-https'])

    @unittest.skipUnless(os.environ.get('GARAJ_TEST_DATA_ROOT'), 'BLOCKED: set GARAJ_TEST_DATA_ROOT for local private bundle test')
    def test_real_models_load_locally(self):
        from phase7_inference.bundle import BundleConfig, load_bundle
        bundle = load_bundle(BundleConfig(root=Path(os.environ['GARAJ_TEST_DATA_ROOT'])))
        self.assertIsNotNone(bundle.model_a)
        self.assertIsNotNone(bundle.model_b)


if __name__ == '__main__':
    unittest.main()

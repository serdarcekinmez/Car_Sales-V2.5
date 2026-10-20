import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from pipeline import download_demo_data as D
from pipeline.prepare_space import prepare


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'bundle'
        self.archive = self.base / 'data.zip'
        self.manifest = {'Datasets/test.csv': {'size': 4, 'sha256': hashlib.sha256(b'test').hexdigest()}}
        path = self.base / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        self.patch = patch.object(D, 'MANIFEST', path)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def zip(self, entries):
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name, data in entries:
                z.writestr(name, data)

    def test_extract_and_reuse_without_token_or_network(self):
        self.zip([('Datasets/test.csv', b'test')])
        D.download(self.root, archive=self.archive)
        self.assertTrue(D.valid_data(self.root, self.manifest))
        D.download(self.root, fetch=lambda **kw: self.fail('download repeated'))

    def test_private_auth_arguments(self):
        self.zip([('Datasets/test.csv', b'test')])
        def fetch(**kwargs):
            self.assertEqual((kwargs['repo_id'], kwargs['repo_type'], kwargs['filename'],
                              kwargs['revision'], kwargs['token']),
                             (D.REPO, 'dataset', D.FILENAME, 'abc123', 'synthetic-test-token'))
            return self.archive
        D.download(self.root, token='synthetic-test-token', revision='abc123', fetch=fetch)

    def test_missing_token_and_revision(self):
        for kwargs in ({}, {'revision': ''}):
            with self.assertRaises(D.DataError):
                D.download(self.root, **kwargs)

    def test_auth_error_redacted(self):
        class Failure(Exception):
            response = type('Response', (), {'status_code': 403})()
        def fetch(**kwargs):
            raise Failure('sensitive-token')
        with self.assertRaisesRegex(D.DataError, 'authentication failed') as ctx:
            D.download(self.root, token='synthetic', fetch=fetch)
        self.assertNotIn('sensitive', str(ctx.exception))

    def test_missing_remote_file_and_network_errors_are_redacted(self):
        for status in (404, 500):
            class Failure(Exception):
                response = type('Response', (), {'status_code': status})()
            def fetch(**kwargs):
                raise Failure('private-content')
            with self.assertRaises(D.DataError) as ctx:
                D.download(self.root, token='synthetic', fetch=fetch)
            self.assertNotIn('private-content', str(ctx.exception))

    def test_unsafe_names(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'a\\b', 'a/../b', 'a//b', 'a./b'):
            with self.subTest(name=name), self.assertRaises(D.DataError):
                D.safe_name(name)

    def test_reject_incomplete_extra_duplicate_and_corrupt_content(self):
        cases = [[], [('extra', b'')], [('Datasets/test.csv', b'evil')],
                 [('Datasets/test.csv', b'test'), ('Datasets/test.csv', b'test')]]
        for entries in cases:
            with self.subTest(entries=len(entries)):
                self.zip(entries)
                with self.assertRaises(D.DataError):
                    D.extract_archive(self.archive, self.root, self.manifest)
                self.assertFalse(self.root.exists())
                self.assertEqual(list(self.base.glob('.garaj-stage-*')), [])

    def test_reject_symlink_and_limits(self):
        info = zipfile.ZipInfo('Datasets/test.csv')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.zip([(info, b'test')])
        with self.assertRaises(D.DataError):
            D.extract_archive(self.archive, self.root, self.manifest)
        self.zip([('Datasets/test.csv', b'test')])
        with patch.object(D, 'MAX_TOTAL', 1), self.assertRaises(D.DataError):
            D.extract_archive(self.archive, self.root, self.manifest)

    def test_html_is_not_zip(self):
        self.archive.write_text('<html>login</html>')
        with self.assertRaisesRegex(D.DataError, 'Invalid or corrupt'):
            D.extract_archive(self.archive, self.root, self.manifest)

    def test_existing_modified_data_not_overwritten(self):
        self.root.mkdir()
        (self.root / 'keep').write_text('keep')
        with self.assertRaises(D.DataError):
            D.download(self.root)
        self.assertEqual((self.root / 'keep').read_text(), 'keep')

    def test_code_only_package(self):
        stage = prepare(self.base / 'space')
        paths = [p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()]
        self.assertIn('Dockerfile', paths)
        self.assertIn('src/phase8_demo/static/index.html', paths)
        self.assertFalse(any(p.startswith(('Datasets/', 'workflow/')) or p.endswith(('.zip', '.sqlite3'))
                             or p == '.env' for p in paths))


if __name__ == '__main__':
    unittest.main()

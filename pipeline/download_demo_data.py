"""Fetch the private, manifest-pinned inference bundle at runtime only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

MANIFEST = Path(__file__).resolve().parents[1] / 'deploy/huggingface/data-manifest.json'
REPO = 'SerdarCZ/Garaj-demo-data'
FILENAME = 'garaj_demo_data.zip'
MAX_ARCHIVE = 64 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_FILE = 128 * 1024**2
MAX_ENTRIES = 100


class DataError(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or '\x00' in name
            or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/'))
            or any(x.endswith((' ', '.')) for x in p.parts)):
        raise DataError('Archive contains an unsafe path.')
    return p


def valid_data(root, manifest):
    if root.is_symlink() or not root.is_dir():
        return False
    for name, entry in manifest.items():
        parts = safe_name(name).parts
        path = root
        for part in parts:
            path = path / part
            if path.is_symlink():
                return False
        if (not path.is_file() or path.stat().st_size != entry['size']
                or digest(path) != entry['sha256']):
            return False
    # Reject extra files: automatic workbook/FX discovery must stay unambiguous.
    paths = list(root.rglob('*'))
    return (not any(p.is_symlink() for p in paths)
            and {p.relative_to(root).as_posix() for p in paths if p.is_file()} == set(manifest))


def extract_archive(archive, root, manifest):
    """Publish a new directory by rename; never merge with or overwrite existing data."""
    root = Path(root).absolute()
    if any(p.is_symlink() for p in root.parents):
        raise DataError('Data root must not have symlink ancestors.')
    if root.exists() or root.is_symlink():
        raise DataError('Data root already exists but is invalid; choose a new empty path.')
    if Path(archive).stat().st_size > MAX_ARCHIVE:
        raise DataError('Archive exceeds compressed size limit.')
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf, tempfile.TemporaryDirectory(
                prefix='.garaj-stage-', dir=root.parent) as temp:
            stage = Path(temp) / 'bundle'
            stage.mkdir(mode=0o700)
            entries = zf.infolist()
            if len(entries) > MAX_ENTRIES or sum(i.file_size for i in entries) > MAX_TOTAL:
                raise DataError('Archive exceeds content limits.')
            seen = set()
            for info in entries:
                safe_name(info.filename)
                mode = info.external_attr >> 16
                if (info.filename.casefold() in seen or info.is_dir()
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                        or info.flag_bits & 1 or info.file_size > MAX_FILE
                        or info.file_size > max(1, info.compress_size) * 1000):
                    raise DataError('Archive contains duplicate, special, encrypted or oversized entries.')
                seen.add(info.filename.casefold())
                if info.filename not in manifest:
                    raise DataError('Archive contains an unexpected file; update the reviewed manifest first.')
            if {i.filename for i in entries} != set(manifest):
                raise DataError('Archive is missing required inference files.')
            for info in entries:
                expected = manifest[info.filename]
                if info.file_size != expected['size']:
                    raise DataError('Archive file size differs from the reviewed manifest.')
                target = stage / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                count = 0
                with zf.open(info) as source, target.open('xb') as dest:
                    for block in iter(lambda: source.read(1024**2), b''):
                        count += len(block)
                        if count > expected['size']:
                            raise DataError('Extracted file exceeds expected size.')
                        dest.write(block)
                target.chmod(0o600)
            if not valid_data(stage, manifest):
                raise DataError('Data integrity check failed (SHA-256).')
            stage.rename(root)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError('Invalid or corrupt ZIP archive.') from None


def download(root, *, revision='main', token=None, archive=None, fetch=None):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    root = Path(root).absolute()
    if not revision or revision != revision.strip():
        raise DataError('GARAJ_DEMO_DATA_REVISION must be non-empty without outer spaces.')
    if valid_data(root, manifest):
        return root
    if root.exists() or root.is_symlink():
        raise DataError('Existing data is incomplete or modified; select a new GARAJ_DEMO_ROOT.')
    if archive:
        extract_archive(archive, root, manifest)
        return root
    if not token or not token.strip():
        raise DataError('GARAJ_DEMO_DATA_TOKEN is required for the private dataset.')
    if fetch is None:
        from huggingface_hub import hf_hub_download
        fetch = hf_hub_download
    # Cache is temporary: no extra archive copy remains after successful startup.
    with tempfile.TemporaryDirectory(prefix='garaj-download-') as cache:
        try:
            archive = fetch(repo_id=REPO, repo_type='dataset', filename=FILENAME,
                            revision=revision, token=token, cache_dir=cache)
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401, 403):
                message = 'Dataset authentication failed; check token read permission.'
            elif status == 404:
                message = 'Dataset file/revision is missing or inaccessible to this token.'
            else:
                message = 'Dataset download failed; check access, revision and network.'
            raise DataError(message) from None
        extract_archive(archive, root, manifest)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Validate/extract an existing local ZIP; no network')
    parser.add_argument('--root', default=os.environ.get('GARAJ_DEMO_ROOT'))
    args = parser.parse_args()
    try:
        if not args.root:
            raise DataError('GARAJ_DEMO_ROOT or --root is required.')
        download(args.root, revision=os.environ.get('GARAJ_DEMO_DATA_REVISION', 'main'),
                 token=os.environ.get('GARAJ_DEMO_DATA_TOKEN'), archive=args.archive)
    except (DataError, OSError):
        # OSError paths and SDK exception bodies can contain sensitive information.
        exc = sys.exc_info()[1]
        print(str(exc) if isinstance(exc, DataError) else 'Data filesystem operation failed.', file=sys.stderr)
        return 2
    print('Inference data verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

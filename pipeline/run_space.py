"""Spaces-only bootstrap. Secrets are consumed at runtime, never at build time."""
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from download_demo_data import DataError, download


def main():
    from phase8_demo.app import settings_from_env, ConfigError
    from phase8_demo.mailer import SmtpMailer

    try:
        root = os.environ.get('GARAJ_DEMO_ROOT', '')
        store = os.environ.get('GARAJ_DEMO_STORE', '')
        hosts = os.environ.get('GARAJ_DEMO_ALLOWED_HOSTS', '')
        if not root or not Path(root).is_absolute() or not store or not Path(store).is_absolute():
            raise DataError('GARAJ_DEMO_ROOT and GARAJ_DEMO_STORE must be absolute paths.')
        if not hosts or any(not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', h.strip())
                            for h in hosts.split(',')):
            raise DataError('GARAJ_DEMO_ALLOWED_HOSTS must contain explicit hostnames without schemes or wildcards.')
        if Path(store).resolve().is_relative_to(Path(root).resolve()):
            raise DataError('GARAJ_DEMO_STORE must be outside the immutable data bundle.')
        # Validate credentials before any network access; the bundle need not exist yet.
        settings_from_env(os.environ, root=Path(__file__).resolve().parents[1])
        SmtpMailer.from_env(os.environ)
        download(root, revision=os.environ.get('GARAJ_DEMO_DATA_REVISION', 'main'),
                 token=os.environ.get('GARAJ_DEMO_DATA_TOKEN'))
    except DataError as exc:
        print(f'Space startup failed: {exc}', file=sys.stderr)
        return 2
    except (ConfigError, ValueError, OSError):
        print('Space startup failed: check credentials, hosts, data revision/integrity and writable runtime paths.', file=sys.stderr)
        return 2
    # Do not pass the dataset token to the serving process after the download.
    os.environ.pop('GARAJ_DEMO_DATA_TOKEN', None)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('run_demo.py')),
                            'serve', '--host', '0.0.0.0', '--port', '8000', '--behind-https'])


if __name__ == '__main__':
    sys.exit(main())

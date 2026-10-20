"""Prepare an explicit code-only upload folder. Does not contact any service."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('common', 'phase1_preparation', 'phase2_labels', 'phase3_firm_universe',
            'phase4_modeling', 'phase5_pricing', 'phase7_inference', 'phase8_demo')


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Destination must not exist; use a new directory.')
    files = {p.relative_to(ROOT).as_posix(): p.relative_to(ROOT).as_posix()
             for package in PACKAGES for p in (ROOT / 'src' / package).glob('*.py')}
    for name in ('index.html', 'app.js', 'style.css'):
        rel = 'src/phase8_demo/static/' + name
        files[rel] = rel
    for rel in ('requirements-demo.txt', 'requirements-space.txt', '.dockerignore',
                'pipeline/run_demo.py', 'pipeline/run_space.py', 'pipeline/download_demo_data.py',
                'deploy/huggingface/data-manifest.json'):
        files[rel] = rel
    files['Dockerfile.huggingface'] = 'Dockerfile'
    files['deploy/huggingface/README.md'] = 'README.md'
    for source, target in files.items():
        path = ROOT / source
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError('Unexpected deployment source file.')
        dest = destination / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    prepare(parser.parse_args().destination)

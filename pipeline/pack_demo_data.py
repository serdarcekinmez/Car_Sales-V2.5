#!/usr/bin/env python3
"""
pack_demo_data.py  —  zip exactly the data and model files the demo reads.

  python pipeline/pack_demo_data.py [--root DIR] [--out garaj_demo_data.zip]

The Streamlit Cloud demo (streamlit_app.py) cannot see the git-ignored Datasets/ and
workflow/runs/ folders. This script resolves every Phase-7 bundle input exactly as the app
does (BundleConfig defaults, the current Katılım / Üye / FX file found by role) and writes
them into one zip with their repository-relative paths. Upload the zip to PRIVATE storage
(e.g. a private Hugging Face dataset) and give its address to the app as
GARAJ_DEMO_DATA_URL (see docs/STREAMLIT_YAYIN.md). Never commit the zip to a public repo.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUTS = ("clean_sales", "member_directory", "clean_reel_db", "labeled_candidates",
          "firm_registry", "internal_lookup", "aa_katilim", "fx_file", "model_a", "model_b")


def main(argv=None) -> int:
    from phase7_inference import bundle as BD

    parser = argparse.ArgumentParser(description="Zip the demo's data and model files")
    parser.add_argument("--root", default=str(ROOT), help="data / model root (default: repo)")
    parser.add_argument("--out", default="garaj_demo_data.zip", help="output zip file")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    config = BD.BundleConfig(root=root)
    files = []
    for name in INPUTS:
        try:
            path = BD._resolve(config, name)
        except BD.BundleError as exc:
            print(f"pack_demo_data: {exc}", file=sys.stderr)
            return 2
        files.append(path.resolve().relative_to(root))

    out = Path(args.out).resolve()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(root / rel, rel.as_posix())
            print(f"  + {rel.as_posix()}")
    print(f"{len(files)} dosya -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

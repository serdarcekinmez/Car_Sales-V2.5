# FastAPI Docker Space deployment

Prepared locally on 2026-10-20. Nothing has been committed, pushed, uploaded or deployed.
Target: `CSerdar014191/Car_Sales_V2.5`. Dataset: `SerdarCZ/Garaj-demo-data` (private).

## Architecture and data contract

`pipeline/run_space.py` validates settings, invokes the runtime downloader, removes the
dataset token from its environment and execs the existing `pipeline/run_demo.py serve`
with `--host 0.0.0.0 --port 8000 --behind-https`. One Uvicorn worker serves the existing
FastAPI UI and API. `DemoRuntime` loads `BundleConfig` in a background thread. Model A
and Model B load independently; `/api/status` must show BOTH available before acceptance.
No algorithms, interface, models or training code were changed. `Dockerfile.demo` and
the existing local runner remain available.

The inspected local ZIP has 10 files, 33,371,094 uncompressed bytes. The reviewed paths,
sizes and SHA-256 hashes are in `deploy/huggingface/data-manifest.json`:

- Five fixed data files: clean sales, clean reel DB, labeled candidates, firm registry,
  internal firm lookup.
- Two role-discovered raw workbooks: member directory and AA participation.
- One FX snapshot ending 2026-10-09.
- Model A: `workflow/runs/phase4_experiment_build/results/final_model.json`.
- Model B: `workflow/runs/phase5_experiment_build/results/models/lgbm_noR.json`.

The downloader uses explicit `repo_type="dataset"`, token and revision arguments to
`hf_hub_download`. It rejects unexpected entries, missing files, duplicates, unsafe
paths, links, special files, encrypted ZIPs, bad CRCs and hash mismatches. Limits are
64 MiB compressed, 256 MiB total expanded, 128 MiB per member, 100 entries and 1000:1
compression ratio. Only the exact manifest files are written in a temporary sibling
directory, then published by rename. Existing invalid directories are never overwritten.

The serving runtime separately refreshes USD/TRY from Yahoo Finance on startup and once
per UTC day. An hourly worker retries failed checks and detects the next day. Immutable
FX snapshots and an atomic checkpoint live in `fx-cache/` beside `GARAJ_DEMO_STORE`,
outside the verified data bundle. The price model adopts a new snapshot without a server
restart; active predictions retain their original bundle identity. Network/write failures
keep the previous snapshot and show a warning. The existing seven-day FX limit still
applies. No extra HF secret or scheduled agent is needed; allow outbound HTTPS to Yahoo
Finance. After changing GitHub code, run the manual deployment workflow to update Space.
Verified existing data is reused offline. ZIP caches and staging directories are temporary.
The compressed limit is checked after the SDK download; token-scoped repository access
and the reviewed manifest are the trust boundary, not a streaming download quota.

Pin `GARAJ_DEMO_DATA_REVISION` to a full dataset commit SHA. `main` is a convenience
default, but content hashes are enforced even then. Updating the dataset requires reviewing
and updating the manifest, and selecting a new empty runtime root (or rebuilding ephemeral
storage). A new revision with identical manifest content can reuse the verified local data.
Never upload this local ZIP, extracted files or model binaries with the deployment code.

## Configuration

Space Secrets (names only):

```
GARAJ_DEMO_DATA_TOKEN
GARAJ_DEMO_USERNAME
GARAJ_DEMO_PASSWORD_HASH
GARAJ_DEMO_EMAIL
```

Space Variables (names only):

```
GARAJ_DEMO_ALLOWED_HOSTS
GARAJ_DEMO_ROOT
GARAJ_DEMO_STORE
GARAJ_DEMO_DATA_REVISION
```

GitHub Actions secret (names only):

```
HF_SPACE_WRITE_TOKEN
```

The dataset token must belong to an account authorized to read `SerdarCZ/Garaj-demo-data`,
with fine-grained read-only permission for that repository. Space ownership under a
different account does not grant dataset access. Use a separate fine-grained write token
restricted to the target Space for Actions; never give it to the application.

Generate the scrypt hash interactively with `python pipeline/run_demo.py hash-password`.
Paste the entire hash value after the equals sign into the Space secret, without quotes.
Never put the password itself in settings or scripts. Email is optional; password reset
also requires the existing SMTP settings. SMTP password belongs in a Secret if enabled.

Use the exact direct app hostname shown by Hugging Face for `GARAJ_DEMO_ALLOWED_HOSTS`,
without scheme, path or wildcard. Do not guess its underscore/dot normalization.
Defaults in the Space image are `/home/demo/runtime/bundle` for root and
`/home/demo/runtime/demo.sqlite3` for the store. The store must be outside the bundle.
The container runs as UID 1000 with a writable home/runtime directory; application code
is root-owned. Mounting any custom directory requires UID 1000 write permission.

## Local verification and later deployment

Do not commit, push or run the deployment workflow until the owner approves.

1. Use Python 3.10 for final compatibility checks. Install and run:

   ```powershell
   python -m pip install -r requirements-space.txt httpx==0.28.1
   python -m unittest discover -s tests -v
   python pipeline/download_demo_data.py --archive garaj_demo_data.zip --root .deployment-check/bundle
   $env:GARAJ_TEST_DATA_ROOT = (Resolve-Path .deployment-check/bundle).Path
   python -m unittest discover -s tests -v
   ```

   The optional real-data test only loads models locally; it does not serve customer data.
   If that root already holds different data, use a new empty path; do not delete data.

2. With Docker running, build the code-only image:

   ```powershell
   docker build -f Dockerfile.huggingface -t garaj-space .
   docker build -f Dockerfile.demo -t garaj-demo .
   ```

   For a local-only data check without downloading, supply the already verified bundle:

   ```powershell
   docker run --rm -p 127.0.0.1:8000:8000 --env-file .env --mount "type=bind,source=$((Resolve-Path .deployment-check/bundle).Path),target=/home/demo/runtime/bundle,readonly" garaj-space
   ```

   Secure cookies require an HTTPS reverse proxy even locally. A plain HTTP request to
   `/` is only a liveness check; full login acceptance uses HTTPS. Keep real data local.
   For a real Hub download test, obtain explicit owner approval first, then run without
   the bundle mount with the authorized dataset token and pinned revision.

3. After approval, commit/push the reviewed code. Configure the existing Space's Secrets
   and Variables without changing repository visibility. Add `HF_SPACE_WRITE_TOKEN` in
   GitHub. Optionally protect the `huggingface-space` Actions environment with reviewers.

4. In GitHub Actions, select **Deploy FastAPI Space (manual)**, choose the approved ref
   and **Run workflow**. The workflow only has `workflow_dispatch`; pushes do not deploy.
   It runs offline tests and a Docker build before upload. `prepare_space.py` creates a
   fresh code-only folder, maps `Dockerfile.huggingface` to `Dockerfile` and the Space
   README to root `README.md`; it does not upload the Git history or root README.
   The upload updates files in the existing Space and does not delete pre-existing files.
   Inspect any old Space files separately before customer access; this workflow cannot
   establish that the existing remote repository contains no previous confidential files.

5. Watch Space startup, then open its direct HTTPS app URL in a new tab. The existing
   `frame-ancestors 'none'`, `X-Frame-Options: DENY` and SameSite=Strict protections mean
   the embedded Hugging Face page is not the supported entry point. Do not relax these.
   Sign in and verify `/api/status` shows both model components ready. Exercise a synthetic
   vehicle, filtering, CSV download, feedback and logout with the authorized customer.
   Check bad credentials, no-session API requests, foreign Origin and missing CSRF rejection.

## Security and operational limits

Only the three explicit UI assets are served. Data/model paths have no file-serving route.
Predictions, history, status, CSV and feedback require authentication and owner checks.
Sensitive data is excluded from the Docker context and deployment folder. No secrets are
build arguments or Docker layers. Production entrypoint never enables mail-to-console or
remote shutdown. Download exceptions are sanitized; no SDK response bodies are logged.
Existing app logs still include operational usernames/IP addresses and bundle diagnostics;
restrict access to Space logs. This preparation does not audit remote settings or log ACLs.

HTTPS terminates at the HF reverse proxy. Secure cookies/HSTS and exact Host/Origin
checks remain enabled. Uvicorn proxy headers remain disabled. Leave trusted proxies unset
unless HF provides a verified fixed proxy IP set; never blindly trust all forwarded IPs.
Login throttling may be shared by users arriving through the same proxy, which can cause
temporary lockouts. One pilot account per deployment remains the supported design.

`/` is the public liveness page; `/api/status` is authenticated readiness information.
There is no new public diagnostic or model-health endpoint. A running process does not
prove both models loaded; partial model failure is an existing supported app state.

SQLite stores feedback, predictions and password resets on ephemeral storage by default.
Container replacement may lose all of them; sessions are in memory and restart invalidates
them. Before collecting important customer feedback, arrange a controlled SQLite backup
using SQLite's backup API, or later mount persistent storage and point GARAJ_DEMO_STORE
there. No external database, paid storage or automatic data upload has been introduced.

The inspected FX snapshot ends 2026-10-09. The existing seven-day lag rule means current
2026-10-20 pricing may be unavailable even when Model B loads correctly. Refresh approved
FX data and review a new manifest before present-day customer pricing tests, or explicitly
use a supported historical date. Historical customer data is also stale; no retraining or
data refresh was performed here.

References: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[Hub downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download),
[Hub uploads](https://huggingface.co/docs/huggingface_hub/en/guides/upload).

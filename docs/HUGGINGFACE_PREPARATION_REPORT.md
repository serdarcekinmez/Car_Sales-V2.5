# Local preparation report — 2026-10-20

Status: prepared locally; deployment acceptance is still blocked. No commits, pushes,
remote uploads, visibility changes, dataset downloads or predictive model changes occurred.
Only public Python packages were downloaded after approval, into `.venv-deploy`.

## Architecture

Existing FastAPI Phase 8 UI/API -> Phase 7 inference bundle -> frozen LightGBM 4.6.0
Model A ranking and Model B pricing. Background model loading, single worker, SQLite
prediction/feedback store, scrypt login, secure sessions, CSRF and exact host checks remain.
Space startup adds a runtime-only private data fetch before the existing runner starts.
Python 3.10, UID 1000, libgomp1 and port 8000 are configured in the separate Space image.

## Files

Created:

- `pipeline/download_demo_data.py`: authenticated download, bounded safe ZIP extraction,
  manifest validation, staged publication and verified data reuse.
- `pipeline/run_space.py`: settings validation, download and secure existing runner exec.
- `pipeline/prepare_space.py`: code-only deployment folder assembly.
- `Dockerfile.huggingface`, `requirements-space.txt`: separate Space runtime.
- `deploy/huggingface/README.md`: Space YAML metadata and direct-URL instructions.
- `deploy/huggingface/data-manifest.json`: ten reviewed paths, sizes and hashes only.
- `.github/workflows/deploy-space.yml`: manual-only test/build/upload workflow.
- `tests/test_download_demo_data.py`, `tests/test_space_app.py`: deployment tests.
- `docs/HUGGINGFACE_DEPLOYMENT.md`: configuration, security analysis and exact runbook.
- This report.

Modified: `.dockerignore`, `.gitignore`, `.env.example`.
Existing application source, models, requirements-demo pins and Dockerfile.demo are unchanged.
Ignored local test outputs: `.venv-deploy/`, `.deployment-check/`, `.space-package/`.
The extracted private test bundle remains local in `.deployment-check/bundle`; it was
not exposed through a server or uploaded, and is excluded from Git/Docker/upload packaging.

## Verification

Passed:

- Local ZIP CRC validation and SHA-256 manifest generation for all ten inference files.
- Safe extraction of that ZIP and code-only deployment package assembly.
- 15 offline tests in the final suite: token/revision validation and explicit SDK arguments,
  redacted auth/404/network errors, unsafe paths/symlinks/duplicates/limits, HTML rejection,
  missing/extra/tampered data, clean staging failure, data reuse, non-overwrite behavior,
  packaging, configuration and password-hash compatibility, FastAPI lifespan, login/logout,
  secure cookies, host/Origin/CSRF checks, protected status/prediction/CSV routes, unavailable
  data/model file routes, and the Space bootstrap/exec contract.
- The separate opt-in real-model test passed with `GARAJ_TEST_DATA_ROOT` pointed to the
  extracted local bundle: both Model A and Model B loaded. This test was intentionally
  skipped in the final offline rerun; its earlier passing result remains valid.
- `pip check`, Python syntax parsing, manual-only workflow YAML and `git diff --check`.

The suite contains 16 tests total. Runtime tests used Windows Python 3.13.1 with the
existing pinned model libraries, including LightGBM 4.6.0. No unresolved test failures.
Duplicate-ZIP fixture and Starlette/AnyIO deprecation warnings are non-failing.

BLOCKED / not claimed passed:

- Real private Hub authentication/download: no token used; no dataset download authorized.
- Python 3.10 Linux image build/startup and original Dockerfile.demo build: Docker daemon
  is unavailable and Python 3.10 is not installed locally.
- Live HF host/proxy/cookies, resource/startup-time limits and browser/customer acceptance:
  no deployment authorized. Synthetic HTTP tests are in-process, not public endpoints.
- GitHub Actions execution and target Space settings/files audit: workflow not dispatched.

## Customer-testing blockers and security concerns

1. Obtain separately scoped dataset read token and Space write token, set the account
   secrets, confirm exact allowed hostname, and pin the private dataset commit SHA.
2. Complete Docker/Python 3.10 verification and authorized real-download testing.
3. The FX snapshot ends 2026-10-09; current-date pricing can be unavailable under the
   unchanged seven-day rule. Approve refreshed FX data/manifest or use a supported date.
4. Use the direct HTTPS app URL in a separate tab. Existing anti-framing headers and
   SameSite=Strict cookies are preserved; HF's embedded frame is not supported.
5. SQLite feedback is ephemeral; plan a controlled backup before collecting important
   feedback. No external database or paid persistence was added.
6. Leave forwarded-IP trust unset until actual proxy addresses are verified. Shared proxy
   throttling can cause temporary account lockouts. Existing operational logs contain
   usernames/IPs; remote log access and pre-existing Space files still need review.

Required names, values guidance and exact deployment steps are in
[HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md). Stop here pending explicit owner
approval before any commit, push or deployment, as required by the task's strict rules.

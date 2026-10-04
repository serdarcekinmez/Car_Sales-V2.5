"""
Phase 8: the private customer web demo over the Phase 7 inference layer.

Modules (this package imports nothing at load):
  phase8_demo.security  password hashing, sessions, CSRF, login throttling, ASGI guard
  phase8_demo.store     SQLite store of immutable predictions and their feedback
  phase8_demo.app       DemoSettings, create_app (FastAPI) and the browser projection
Start it with pipeline/run_demo.py; see docs/DEMO_KURULUM.md.
"""

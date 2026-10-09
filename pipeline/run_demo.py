#!/usr/bin/env python3
"""
run_demo.py  —  Phase 8: start the private customer demo, or create its password hash.

  python pipeline/run_demo.py hash-password
      Asks twice for the pilot password (not echoed; at least 12 characters with upper-
      and lower-case letters, a digit and a symbol: security.password_problems) and prints
      the GARAJ_DEMO_PASSWORD_HASH=... line to paste into .env. Nothing else happens.

  python pipeline/run_demo.py serve [--root DIR] [--store FILE] [--host H] [--port P]
                                   [--insecure-local-http | --behind-https]
                                   [--allowed-host NAME ...] [--allow-shutdown]
                                   [--open-browser]
      Starts ONE uvicorn worker serving phase8_demo.app. It trains, evaluates, refreshes
      and fetches nothing; the Phase-7 bundle is read from --root (default: GARAJ_DEMO_ROOT
      or this repository). Configuration: GARAJ_DEMO_* environment variables (.env.example).

HOST RULES
  * Default host 127.0.0.1 (this machine only).
  * --insecure-local-http drops the cookie Secure flag so http://127.0.0.1:PORT works;
    it is refused on any non-loopback host.
  * Any non-loopback host (e.g. 0.0.0.0 in Docker) requires --behind-https: TLS is
    terminated by a reverse proxy in front of the app.
  * --behind-https requires explicit allowed host names (--allowed-host or
    GARAJ_DEMO_ALLOWED_HOSTS): the Host / Origin checks must know the public name.
  * --allow-shutdown shows the "Oturumu kapat" button: the signed-in user can stop this
    server from the page (the terminal / command window then returns or closes). It is
    refused on any non-loopback host, so a public deployment cannot be stopped remotely.
  * --open-browser opens http://127.0.0.1:PORT in the default browser once the server runs.
  * --mail-to-console writes password-reset e-mails (with the code) to this terminal
    instead of sending them, for trying the reset without an SMTP account. It is refused
    on any non-loopback host. Otherwise e-mail goes through GARAJ_DEMO_SMTP_* (mailer.py);
    without GARAJ_DEMO_SMTP_HOST or GARAJ_DEMO_EMAIL the password reset is off.
"""
from __future__ import annotations

import argparse
import getpass
import ipaddress
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _fail(message: str) -> int:
    print(f"run_demo: {message}", file=sys.stderr)
    return 2


def cmd_hash_password(_args) -> int:
    from phase8_demo import security

    print(security.PASSWORD_RULES_TR)
    password = getpass.getpass("Pilot parolası: ")
    again = getpass.getpass("Parola tekrar: ")
    if password != again:
        return _fail("parolalar eşleşmiyor")
    try:
        encoded = security.hash_password(password)
    except ValueError as exc:
        return _fail(f"parola kabul edilmedi: {exc}")
    print(f"GARAJ_DEMO_PASSWORD_HASH={encoded}")
    return 0


def cmd_serve(args) -> int:
    loopback = _is_loopback(args.host)
    if args.insecure_local_http and args.behind_https:
        return _fail("--insecure-local-http and --behind-https exclude each other")
    if args.insecure_local_http and not loopback:
        return _fail("--insecure-local-http is only allowed on a loopback host "
                     "(127.0.0.1 / localhost / ::1)")
    if not loopback and not args.behind_https:
        return _fail(f"host {args.host!r} is reachable from other machines; serve it only "
                     "behind an HTTPS reverse proxy and pass --behind-https")
    allowed = args.allowed_host or None
    if args.allow_shutdown and not loopback:
        return _fail("--allow-shutdown is only allowed on a loopback host")
    if args.mail_to_console and not loopback:
        return _fail("--mail-to-console is only allowed on a loopback host")
    if args.behind_https and allowed is None \
            and not os.environ.get("GARAJ_DEMO_ALLOWED_HOSTS", "").strip():
        return _fail("--behind-https needs the public host name: --allowed-host NAME or "
                     "GARAJ_DEMO_ALLOWED_HOSTS")

    from phase8_demo.app import ConfigError, create_app, settings_from_env
    from phase8_demo.mailer import ConsoleMailer, SmtpMailer

    import uvicorn

    try:
        mailer = ConsoleMailer() if args.mail_to_console else SmtpMailer.from_env(os.environ)
    except ValueError as exc:
        return _fail(str(exc))

    server: uvicorn.Server | None = None

    def stop_server() -> None:
        if server is not None:
            server.should_exit = True

    try:
        settings = settings_from_env(os.environ, root=args.root, store_path=args.store,
                                     secure_cookies=not args.insecure_local_http,
                                     allowed_hosts=allowed)
        app = create_app(settings, on_shutdown=stop_server if args.allow_shutdown else None,
                         mailer=mailer)
    except ConfigError as exc:
        return _fail(str(exc))

    log = logging.getLogger("run_demo")
    log.info("serving on %s:%d (secure cookies: %s, allowed hosts: %s); data root %s",
             args.host, args.port, settings.secure_cookies,
             ", ".join(sorted(settings.allowed_hosts)), settings.root)
    if loopback and settings.secure_cookies and not args.behind_https:
        log.warning("cookies are Secure: open the app through HTTPS, or restart with "
                    "--insecure-local-http for plain http://127.0.0.1:%d", args.port)
    if mailer is None or not settings.email:
        log.info("password reset is off (needs GARAJ_DEMO_EMAIL and GARAJ_DEMO_SMTP_HOST, "
                 "or --mail-to-console)")
    elif args.mail_to_console:
        log.warning("--mail-to-console: password-reset codes are printed here, not e-mailed")
    config = uvicorn.Config(app, host=args.host, port=args.port, workers=1,
                            proxy_headers=False, server_header=False, access_log=True,
                            log_level="info", timeout_keep_alive=5, limit_concurrency=64)
    server = uvicorn.Server(config)
    if args.open_browser:
        import threading
        import webbrowser

        url = f"http://{'127.0.0.1' if loopback else args.host}:{args.port}/"
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    server.run()
    if args.allow_shutdown:
        log.info("server stopped from the page")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Garaj Phase 8 private customer demo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hash-password", help="print a GARAJ_DEMO_PASSWORD_HASH line")
    serve = sub.add_parser("serve", help="start the demo web server")
    serve.add_argument("--root", help="data / model root (default GARAJ_DEMO_ROOT or repo)")
    serve.add_argument("--store", help="SQLite file (default GARAJ_DEMO_STORE or "
                                       "<root>/Datasets/phase8_demo/demo.sqlite3)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--insecure-local-http", action="store_true",
                       help="cookies without Secure; loopback host only")
    serve.add_argument("--behind-https", action="store_true",
                       help="required for non-loopback hosts; TLS by a reverse proxy")
    serve.add_argument("--allowed-host", action="append",
                       help="public host name the browser uses (repeatable)")
    serve.add_argument("--allow-shutdown", action="store_true",
                       help="show the 'Oturumu kapat' button that stops this server; "
                            "loopback host only")
    serve.add_argument("--open-browser", action="store_true",
                       help="open the page in the default browser after start")
    serve.add_argument("--mail-to-console", action="store_true",
                       help="print password-reset e-mails here instead of sending them; "
                            "loopback host only")
    args = parser.parse_args(argv)

    # A Windows console may not encode Turkish letters (cp1252): never lose a log line.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if args.command == "hash-password":
        return cmd_hash_password(args)
    if not 1 <= args.port <= 65535:
        return _fail("--port must be in 1..65535")
    return cmd_serve(args)


if __name__ == "__main__":
    sys.exit(main())

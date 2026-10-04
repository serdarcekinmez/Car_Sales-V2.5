"""
security.py  —  Phase 8: access control for the private customer demo.

Standard library only. Nothing happens at import time.

PASSWORD. One configured pilot account. Its password is stored ONLY as a salted scrypt
hash in the environment (GARAJ_DEMO_PASSWORD_HASH), never in code and never as a shared
default. Format (no '$', so .env files and shells need no quoting):
    scrypt:<N>:<r>:<p>:<salt, base64url>:<digest, base64url>
pipeline/run_demo.py hash-password generates it. verify_password always runs scrypt, also
for a wrong user name, so the user name cannot be probed by timing.

SESSIONS. A login creates a high-entropy random token (cookie, HttpOnly, SameSite=Strict,
Secure unless local HTTP was explicitly allowed) and a separate CSRF token. Sessions live
in server memory with an absolute expiry; logout revokes the token; a restart logs every
user out. Only the SHA-256 of a token is kept as the dictionary key. The number of live
sessions is bounded (oldest dropped first).

CSRF. Every mutating authenticated request must carry X-CSRF-Token equal to the session's
token (constant-time compare), Content-Type application/json (a cross-site form cannot
send it without a CORS preflight, and no CORS is configured) and, when the browser sends
an Origin header, an Origin whose host is an allowed host.

LOGIN THROTTLING. Failed logins are counted per client within a sliding window; a client
over the limit is refused (429 + Retry-After) before scrypt runs. The client is the
direct peer address. X-Forwarded-For is used only when the peer is a configured trusted
proxy (GARAJ_DEMO_TRUSTED_PROXIES), taking the right-most untrusted hop; otherwise it is
ignored. The table is bounded.

GUARD (GuardMiddleware, the outermost ASGI layer, so error responses get it too):
  * Host header must name an allowed host (DNS-rebinding / host-header defence).
  * Mutating requests: Content-Length required and <= max_body_bytes;
    Transfer-Encoding bodies are refused.
  * Security headers on every response: strict CSP without inline code, nosniff,
    frame denial, no referrer, no-store caching, same-origin isolation, HSTS when secure.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlsplit

HASH_SCHEME = "scrypt"
DEFAULT_SCRYPT_N = 2 ** 14
DEFAULT_SCRYPT_R = 8
DEFAULT_SCRYPT_P = 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_SALT_BYTES = 16
_DIGEST_BYTES = 32
MIN_PASSWORD_CHARS = 12               # for newly generated hashes
MAX_PASSWORD_CHARS = 1024
MAX_USERNAME_CHARS = 64
MAX_TOKEN_CHARS = 128

COOKIE_SECURE = "__Host-garaj_demo"   # __Host-: Secure, Path=/, no Domain
COOKIE_LOCAL = "garaj_demo"           # explicit local-HTTP mode only
CSRF_HEADER = "x-csrf-token"

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_MAX_FORWARDED_HOPS = 20


# ----------------------------------------------------------------------------
# Password hashing.
# ----------------------------------------------------------------------------
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    if not text or any(c not in
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                       for c in text):
        raise ValueError("invalid base64url field")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                          maxmem=_SCRYPT_MAXMEM, dklen=_DIGEST_BYTES)


def hash_password(password: str, *, n: int = DEFAULT_SCRYPT_N, r: int = DEFAULT_SCRYPT_R,
                  p: int = DEFAULT_SCRYPT_P) -> str:
    """A new salted scrypt hash in the documented format."""
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    if not MIN_PASSWORD_CHARS <= len(password) <= MAX_PASSWORD_CHARS:
        raise ValueError(f"password must have {MIN_PASSWORD_CHARS}..{MAX_PASSWORD_CHARS} "
                         "characters")
    salt = secrets.token_bytes(_SALT_BYTES)
    encoded = f"{HASH_SCHEME}:{n}:{r}:{p}:{_b64(salt)}:{_b64(_scrypt(password, salt, n, r, p))}"
    parse_password_hash(encoded)                    # the parameters must be accepted back
    return encoded


def parse_password_hash(encoded: str) -> tuple[int, int, int, bytes, bytes]:
    """(n, r, p, salt, digest); raises ValueError on any malformed or weak field."""
    if not isinstance(encoded, str):
        raise ValueError("password hash must be a string")
    parts = encoded.strip().split(":")
    if len(parts) != 6 or parts[0] != HASH_SCHEME:
        raise ValueError("password hash must look like scrypt:N:r:p:salt:digest")
    try:
        n, r, p = (int(x) for x in parts[1:4])
    except ValueError:
        raise ValueError("scrypt N, r and p must be integers") from None
    if n < DEFAULT_SCRYPT_N or n > 2 ** 20 or n & (n - 1):
        raise ValueError(f"scrypt N must be a power of two in {DEFAULT_SCRYPT_N}..2^20")
    if not 1 <= r <= 32 or not 1 <= p <= 16 or 128 * n * r > _SCRYPT_MAXMEM // 2:
        raise ValueError("scrypt r / p out of range, or N * r needs more than 32 MB")
    salt, digest = _unb64(parts[4]), _unb64(parts[5])
    if len(salt) < _SALT_BYTES or len(digest) != _DIGEST_BYTES:
        raise ValueError("scrypt salt or digest has the wrong length")
    return n, r, p, salt, digest


def verify_password(password, encoded: str) -> bool:
    """Constant-time check; runs scrypt even for a non-string or over-long password."""
    n, r, p, salt, digest = parse_password_hash(encoded)
    valid = isinstance(password, str) and 0 < len(password) <= MAX_PASSWORD_CHARS
    candidate = _scrypt(password if valid else "\x00invalid", salt, n, r, p)
    return hmac.compare_digest(candidate, digest) and valid


def check_credentials(username, password, *, expected_username: str,
                      password_hash: str) -> bool:
    """Both must match. The password is always hashed, whatever the user name."""
    pw_ok = verify_password(password, password_hash)
    user = username if isinstance(username, str) and len(username) <= MAX_USERNAME_CHARS \
        else ""
    user_ok = hmac.compare_digest(user.encode("utf-8"), expected_username.encode("utf-8"))
    return pw_ok and user_ok


# ----------------------------------------------------------------------------
# Sessions.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    session_key: str                    # SHA-256 of the cookie token (never the token)
    username: str
    csrf_token: str
    expires_at: float                   # clock() value


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode("ascii", "replace")).hexdigest()


class SessionStore:
    """Bounded in-memory sessions with absolute expiry and explicit revocation."""

    def __init__(self, ttl_seconds: float, *, max_sessions: int = 32, clock=time.monotonic):
        if ttl_seconds <= 0 or max_sessions < 1:
            raise ValueError("ttl_seconds and max_sessions must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self._max = int(max_sessions)
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def _prune(self, now: float) -> None:
        for key in [k for k, s in self._sessions.items() if s.expires_at <= now]:
            del self._sessions[key]

    def create(self, username: str) -> tuple[str, Session]:
        """(cookie token, session). The oldest session is dropped when the store is full."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            self._prune(now)
            while len(self._sessions) >= self._max:
                del self._sessions[next(iter(self._sessions))]
            session = Session(session_key=_token_key(token), username=username,
                              csrf_token=secrets.token_urlsafe(32),
                              expires_at=now + self.ttl_seconds)
            self._sessions[session.session_key] = session
        return token, session

    def get(self, token) -> Session | None:
        if not isinstance(token, str) or not 0 < len(token) <= MAX_TOKEN_CHARS:
            return None
        key = _token_key(token)
        with self._lock:
            session = self._sessions.get(key)
            if session is not None and session.expires_at <= self._clock():
                del self._sessions[key]
                return None
            return session

    def revoke(self, token) -> None:
        if isinstance(token, str) and 0 < len(token) <= MAX_TOKEN_CHARS:
            with self._lock:
                self._sessions.pop(_token_key(token), None)


def csrf_ok(session: Session, header_value) -> bool:
    return isinstance(header_value, str) and 0 < len(header_value) <= MAX_TOKEN_CHARS \
        and hmac.compare_digest(header_value.encode("ascii", "replace"),
                                session.csrf_token.encode("ascii"))


# ----------------------------------------------------------------------------
# Login throttling and client identity.
# ----------------------------------------------------------------------------
class LoginThrottle:
    """At most max_failures failed logins per client per window; bounded memory."""

    def __init__(self, *, max_failures: int = 5, window_seconds: float = 900.0,
                 max_clients: int = 10_000, clock=time.monotonic):
        self._max_failures = int(max_failures)
        self._window = float(window_seconds)
        self._max_clients = int(max_clients)
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: dict[str, deque] = {}

    def _trim(self, q: deque, now: float) -> None:
        while q and q[0] <= now - self._window:
            q.popleft()

    def retry_after(self, client: str) -> int:
        """Seconds until this client may try again; 0 = allowed now."""
        with self._lock:
            q = self._failures.get(client)
            if not q:
                return 0
            now = self._clock()
            self._trim(q, now)
            if not q:
                del self._failures[client]
                return 0
            if len(q) < self._max_failures:
                return 0
            return max(1, int(q[0] + self._window - now) + 1)

    def failure(self, client: str) -> None:
        with self._lock:
            now = self._clock()
            if client not in self._failures and len(self._failures) >= self._max_clients:
                for key in list(self._failures):
                    self._trim(self._failures[key], now)
                    if not self._failures[key]:
                        del self._failures[key]
                while len(self._failures) >= self._max_clients:
                    del self._failures[next(iter(self._failures))]
            q = self._failures.setdefault(client, deque(maxlen=self._max_failures * 4))
            q.append(now)

    def success(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)


def normalize_ip(text: str) -> str | None:
    try:
        return str(ipaddress.ip_address(text.strip()))
    except ValueError:
        return None


def client_ip(peer: str | None, forwarded_for: str | None,
              trusted_proxies: frozenset[str]) -> str:
    """The direct peer, or the right-most untrusted X-Forwarded-For hop when (and only
    when) the peer is a configured trusted proxy."""
    peer_ip = normalize_ip(peer) if peer else None
    if peer_ip is None:
        return "unknown"
    if peer_ip not in trusted_proxies or not forwarded_for:
        return peer_ip
    hops = forwarded_for.split(",")[-_MAX_FORWARDED_HOPS:]
    for hop in reversed(hops):
        ip = normalize_ip(hop)
        if ip is None:
            return peer_ip
        if ip not in trusted_proxies:
            return ip
    return peer_ip


# ----------------------------------------------------------------------------
# Hosts and origins.
# ----------------------------------------------------------------------------
def host_name(host_header: str) -> str:
    """'example.com:443' -> 'example.com'; '[::1]:8000' -> '::1'; lower-case."""
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else ""
    return host.split(":", 1)[0]


def origin_allowed(origin: str | None, allowed_hosts: frozenset[str]) -> bool:
    """No Origin header (non-browser client) passes; CSRF token is still required."""
    if origin is None:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (parts.hostname or "") in allowed_hosts


# ----------------------------------------------------------------------------
# ASGI guard.
# ----------------------------------------------------------------------------
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
       "base-uri 'none'; object-src 'none'")
_OVERRIDDEN = {b"server", b"cache-control", b"pragma", b"content-security-policy",
               b"x-content-type-options", b"x-frame-options", b"referrer-policy",
               b"permissions-policy", b"cross-origin-opener-policy",
               b"cross-origin-resource-policy", b"strict-transport-security"}


class GuardMiddleware:
    """Outermost ASGI wrapper: host allow-list, body limits, security headers."""

    def __init__(self, app, *, allowed_hosts: frozenset[str], secure: bool,
                 max_body_bytes: int):
        self.app = app
        self.allowed_hosts = frozenset(allowed_hosts)
        self.max_body_bytes = int(max_body_bytes)
        headers = [
            (b"cache-control", b"no-store"),
            (b"pragma", b"no-cache"),
            (b"content-security-policy", CSP.encode("ascii")),
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
            (b"cross-origin-opener-policy", b"same-origin"),
            (b"cross-origin-resource-policy", b"same-origin"),
        ]
        if secure:
            headers.append((b"strict-transport-security", b"max-age=31536000"))
        self._headers = headers

    async def _reject(self, send, status: int, text: str) -> None:
        body = text.encode("utf-8")
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                (b"content-length", str(len(body)).encode("ascii"))]
                    + self._headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {}
        for k, v in scope.get("headers", []):
            headers.setdefault(k.decode("latin-1").lower(), v.decode("latin-1"))
        if host_name(headers.get("host", "")) not in self.allowed_hosts:
            await self._reject(send, 400, "Unknown host")
            return
        if "transfer-encoding" in headers:
            await self._reject(send, 411, "Content-Length required")
            return
        length = headers.get("content-length")
        if length is not None:
            if not length.isdigit():
                await self._reject(send, 400, "Invalid Content-Length")
                return
            if int(length) > self.max_body_bytes:
                await self._reject(send, 413, "Request body too large")
                return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                kept = [(k, v) for k, v in message.get("headers", [])
                        if k.lower() not in _OVERRIDDEN]
                message = {**message, "headers": kept + self._headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)

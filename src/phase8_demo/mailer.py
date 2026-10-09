"""
mailer.py  —  Phase 8: e-mail for the password reset (standard library smtplib only).

Nothing happens at import time. Configuration (GARAJ_DEMO_* environment variables):
    GARAJ_DEMO_SMTP_HOST       SMTP server; empty = no e-mail (password reset is then off)
    GARAJ_DEMO_SMTP_PORT       587 (STARTTLS, default) or 465 (TLS from the first byte)
    GARAJ_DEMO_SMTP_USER       login, e.g. the sending Gmail address
    GARAJ_DEMO_SMTP_PASSWORD   its password (for Gmail: an "app password")
    GARAJ_DEMO_SMTP_FROM       sender address (default: GARAJ_DEMO_SMTP_USER)
TLS is always used and the server certificate is verified; plain SMTP is not offered.

ConsoleMailer writes the message to the server log instead of sending it. It is for
trying the reset on this computer without an SMTP account (pipeline/run_demo.py
--mail-to-console, loopback host only).
"""
from __future__ import annotations

import logging
import re
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger(__name__)

SMTP_TIMEOUT_S = 20
_ADDRESS_RE = re.compile(r"[^@\s]{1,64}@[^@\s]+\.[^@\s]+")


def valid_address(text: str) -> bool:
    return isinstance(text, str) and len(text) <= 254 and bool(_ADDRESS_RE.fullmatch(text))


class SmtpMailer:
    def __init__(self, *, host: str, port: int, user: str, password: str, sender: str):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender = sender

    @classmethod
    def from_env(cls, environ) -> SmtpMailer | None:
        """The configured mailer, None when GARAJ_DEMO_SMTP_HOST is empty; ValueError on
        an incomplete or invalid configuration."""
        host = environ.get("GARAJ_DEMO_SMTP_HOST", "").strip()
        if not host:
            return None
        try:
            port = int(environ.get("GARAJ_DEMO_SMTP_PORT", "587").strip() or "587")
        except ValueError:
            raise ValueError("GARAJ_DEMO_SMTP_PORT must be a number") from None
        if not 1 <= port <= 65535:
            raise ValueError("GARAJ_DEMO_SMTP_PORT must be in 1..65535")
        user = environ.get("GARAJ_DEMO_SMTP_USER", "").strip()
        password = environ.get("GARAJ_DEMO_SMTP_PASSWORD", "")
        sender = environ.get("GARAJ_DEMO_SMTP_FROM", "").strip() or user
        if bool(user) != bool(password):
            raise ValueError("GARAJ_DEMO_SMTP_USER and GARAJ_DEMO_SMTP_PASSWORD go together")
        if not valid_address(sender):
            raise ValueError("GARAJ_DEMO_SMTP_FROM (or GARAJ_DEMO_SMTP_USER) must be an "
                             "e-mail address")
        return cls(host=host, port=port, user=user, password=password, sender=sender)

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        context = ssl.create_default_context()
        if self.port == 465:
            server = smtplib.SMTP_SSL(self.host, self.port, timeout=SMTP_TIMEOUT_S,
                                      context=context)
        else:
            server = smtplib.SMTP(self.host, self.port, timeout=SMTP_TIMEOUT_S)
        with server:
            if self.port != 465:
                server.starttls(context=context)
            if self.user:
                server.login(self.user, self.password)
            server.send_message(msg)


class ConsoleMailer:
    def send(self, to: str, subject: str, body: str) -> None:
        log.warning("phase8 demo: e-mail NOT sent (--mail-to-console) to %s: %s\n%s",
                    to, subject, body)

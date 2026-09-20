"""Portal plumbing: captcha, rate limiting, the n8n bridge, activity trail.

Why a captcha of our own: a hosted one needs a third-party account and leaks
every visitor to it. This is a signed arithmetic challenge - the answer is
never sent to the browser, the token is HMAC'd with the server secret and
expires, and one token can only be spent once. It stops scripted sign-up
floods, which is what it is for; it is not a defence against a human.
"""
from __future__ import annotations

import base64
import hmac
import json
import logging
import random
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Deque, Dict, Optional, Tuple

import httpx
from fastapi import Request

from ..config import get_settings

log = logging.getLogger("sentinel.portal")

CAPTCHA_TTL_SECONDS = 300
SIGNUP_LIMIT = (5, 3600)          # 5 sign-ups per hour per IP
LOGIN_LIMIT = (10, 900)           # 10 sign-in attempts per 15 min per IP
OTP_LIMIT = (5, 900)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def client_ip(request: Request) -> str:
    """The caller's address as seen behind Traefik."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


# --------------------------------------------------------------- captcha
def _sign(payload: str) -> str:
    key = get_settings().jwt_secret.encode()
    return hmac.new(key, payload.encode(), sha256).hexdigest()[:32]


_spent: Dict[str, float] = {}
_spent_lock = threading.Lock()


def make_captcha() -> Dict[str, str]:
    """A small arithmetic challenge plus a signed token carrying the answer."""
    a, b = random.randint(2, 9), random.randint(2, 9)
    style = random.choice(["+", "x"])
    answer = a + b if style == "+" else a * b
    question = f"What is {a} {'plus' if style == '+' else 'times'} {b}?"
    body = json.dumps({"a": answer, "n": secrets.token_hex(8),
                       "exp": int(time.time()) + CAPTCHA_TTL_SECONDS})
    blob = base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")
    return {"question": question, "token": f"{blob}.{_sign(blob)}"}


def check_captcha(token: str, answer: str) -> Tuple[bool, str]:
    if not token or answer is None:
        return False, "Answer the anti-robot question."
    try:
        blob, sig = str(token).split(".", 1)
        if not hmac.compare_digest(sig, _sign(blob)):
            return False, "Anti-robot check failed. Try again."
        padded = blob + "=" * (-len(blob) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return False, "Anti-robot check failed. Try again."
    if int(data.get("exp", 0)) < time.time():
        return False, "The anti-robot question expired. Try again."
    with _spent_lock:
        now = time.time()
        for k, v in list(_spent.items()):          # keep the set small
            if v < now:
                _spent.pop(k, None)
        nonce = str(data.get("n"))
        if nonce in _spent:
            return False, "That anti-robot answer was already used."
        _spent[nonce] = now + CAPTCHA_TTL_SECONDS
    try:
        given = int(str(answer).strip())
    except (TypeError, ValueError):
        return False, "Answer with a number."
    if given != int(data.get("a")):
        return False, "That is not the right answer."
    return True, ""


# ----------------------------------------------------------- rate limits
_hits: Dict[str, Deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()


def rate_ok(bucket: str, ip: str, limit: Tuple[int, int]) -> bool:
    count, window = limit
    key = f"{bucket}:{ip}"
    now = time.time()
    with _hits_lock:
        q = _hits[key]
        while q and q[0] < now - window:
            q.popleft()
        if len(q) >= count:
            return False
        q.append(now)
        return True


# -------------------------------------------------------------- n8n bridge
class AuthBridge:
    """Calls the Sentinel_Auth workflow in n8n.

    The workflow owns passwords and one-time codes; this never sees a hash and
    never stores a code. If n8n is unreachable the portal says so plainly
    rather than pretending an account was created.
    """

    def __init__(self) -> None:
        s = get_settings()
        self.base = (s.portal_auth_base_url or "").rstrip("/")
        self.timeout = 30.0

    @property
    def configured(self) -> bool:
        return bool(self.base)

    def call(self, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        if not self.configured:
            return 503, {"ok": False, "error": "Sign-in service is not configured."}
        url = f"{self.base}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(url, json=payload)
        except httpx.HTTPError as exc:
            log.warning("auth bridge %s failed: %s", path, exc)
            return 503, {"ok": False, "error": "Sign-in service is unavailable. Try again shortly."}
        try:
            data = r.json()
        except ValueError:
            data = {"ok": False, "error": "Sign-in service returned an unreadable answer."}
        if not isinstance(data, dict):
            data = {"ok": False, "error": "Sign-in service returned an unreadable answer."}
        status = int(data.get("status") or (200 if data.get("ok") else r.status_code))
        return status, data


_bridge: Optional[AuthBridge] = None


def get_bridge() -> AuthBridge:
    global _bridge
    if _bridge is None:
        _bridge = AuthBridge()
    return _bridge

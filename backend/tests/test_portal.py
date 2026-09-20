"""Portal: captcha, sign-up gates, tenant isolation and the admin panel.

The n8n auth service is stubbed here - these tests pin what THIS side must
enforce before anything reaches it, and what it does with the answer.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="sentinel_portal_test_"))
os.environ.setdefault("SENTINEL_DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("SENTINEL_STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("SENTINEL_CONFIG_DIR", str(_TMP / "config"))
os.environ.setdefault("SENTINEL_VECTOR_DIR", str(_TMP / "vectors"))
os.environ.setdefault("SENTINEL_ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("SENTINEL_ADMIN_USERNAME", "admin")
os.environ.setdefault("SENTINEL_VECTOR_BACKEND", "memory")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.portal import service  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_limits():
    """Every test starts with empty rate-limit buckets.

    They all arrive from the same TestClient address, so without this the
    5-sign-ups-per-hour rule (which test_rate_limit_blocks_a_flood checks
    deliberately) would fail every later test instead.
    """
    service._hits.clear()
    service._spent.clear()
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin(client):
    r = client.post("/api/auth/login", json={"username": os.environ["SENTINEL_ADMIN_USERNAME"],
                                             "password": os.environ["SENTINEL_ADMIN_PASSWORD"]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def stub_auth(monkeypatch):
    """Stand in for the n8n auth workflow."""
    calls = []

    class Bridge:
        configured = True

        def call(self, path, payload):
            calls.append((path, payload))
            if path == "register":
                return 200, {"ok": True, "message": "code sent"}
            if path in ("verify", "otp-login"):
                if payload.get("code") == "123456":
                    return 200, {"ok": True, "email": payload["email"]}
                return 400, {"ok": False, "status": 400, "error": "That code is not valid."}
            if path == "login":
                if payload.get("password") == "Str0ngPass!2026":
                    return 200, {"ok": True}
                return 401, {"ok": False, "status": 401, "error": "Those details did not match."}
            return 200, {"ok": True}

    bridge = Bridge()
    monkeypatch.setattr(service, "get_bridge", lambda: bridge)
    import app.api.routes.portal as portal_routes
    monkeypatch.setattr(portal_routes, "get_bridge", lambda: bridge)
    return calls


def solve(client):
    """Fetch a captcha and work out the answer the way a person would."""
    c = client.get("/api/portal/captcha").json()
    q = c["question"]
    a, b = [int(w) for w in q.replace("?", "").split() if w.isdigit()]
    return c["token"], str(a + b if "plus" in q else a * b)


def signup_body(client, **over):
    token, answer = solve(client)
    body = {"name": "Test Operator", "email": "tester@gmail.com", "phone": "9876543210",
            "password": "Str0ngPass!2026", "account_type": "airport",
            "organisation": "Test International Airport", "site_label": "T2",
            "terms_accepted": True, "captcha_token": token, "captcha_answer": answer}
    body.update(over)
    return body


# ------------------------------------------------------------------ captcha
def test_captcha_is_required_and_single_use(client, stub_auth):
    r = client.post("/api/portal/signup", json=signup_body(client, captcha_answer="0"))
    assert r.status_code == 400 and "right answer" in r.json()["detail"]

    body = signup_body(client)
    assert client.post("/api/portal/signup", json=body).status_code == 200
    again = client.post("/api/portal/signup", json=body)          # replayed token
    assert again.status_code == 400 and "already used" in again.json()["detail"]


def test_captcha_token_cannot_be_forged(client):
    ok, why = service.check_captcha("bm90aGluZw.deadbeefdeadbeefdeadbeefdeadbeef", "4")
    assert ok is False and "failed" in why


# ------------------------------------------------------------------- signup
def test_terms_must_be_accepted(client, stub_auth):
    r = client.post("/api/portal/signup", json=signup_body(client, terms_accepted=False,
                                                           email="noterms@gmail.com"))
    assert r.status_code == 400 and "Terms" in r.json()["detail"]


def test_account_type_must_be_known(client, stub_auth):
    r = client.post("/api/portal/signup", json=signup_body(client, account_type="spaceport",
                                                           email="wrongtype@gmail.com"))
    assert r.status_code == 400


def test_signup_records_terms_and_organisation(client, stub_auth):
    from sqlalchemy import select

    from app.db import session_scope
    from app.portal.models import Organisation, PortalAccount

    body = signup_body(client, email="recorded@gmail.com", organisation="Delhi Airport")
    assert client.post("/api/portal/signup", json=body).status_code == 200
    with session_scope() as db:
        acc = db.scalar(select(PortalAccount).where(PortalAccount.email == "recorded@gmail.com"))
        assert acc is not None and acc.status == "pending"
        assert acc.terms_version and acc.terms_accepted_at is not None
        org = db.get(Organisation, acc.organisation_id)
        assert org.name == "Delhi Airport" and org.account_type == "airport"


def test_verification_activates_the_account(client, stub_auth):
    from sqlalchemy import select

    from app.db import session_scope
    from app.portal.models import PortalAccount

    client.post("/api/portal/signup", json=signup_body(client, email="verify@gmail.com"))
    bad = client.post("/api/portal/verify", json={"email": "verify@gmail.com", "code": "000000"})
    assert bad.status_code == 400
    ok = client.post("/api/portal/verify", json={"email": "verify@gmail.com", "code": "123456"})
    assert ok.status_code == 200
    with session_scope() as db:
        acc = db.scalar(select(PortalAccount).where(PortalAccount.email == "verify@gmail.com"))
        assert acc.status == "active" and acc.verified_at is not None


# -------------------------------------------------------------------- login
def test_password_alone_never_signs_you_in(client, stub_auth):
    client.post("/api/portal/signup", json=signup_body(client, email="two@gmail.com"))
    client.post("/api/portal/verify", json={"email": "two@gmail.com", "code": "123456"})
    token, answer = solve(client)
    r = client.post("/api/portal/login", json={"email": "two@gmail.com",
                                               "password": "Str0ngPass!2026",
                                               "captcha_token": token, "captcha_answer": answer})
    assert r.status_code == 200
    data = r.json()
    assert data["otp_required"] is True and "access_token" not in data


def test_wrong_password_is_refused_and_recorded(client, stub_auth):
    from sqlalchemy import select

    from app.db import session_scope
    from app.portal.models import PortalActivity

    token, answer = solve(client)
    r = client.post("/api/portal/login", json={"email": "two@gmail.com", "password": "nope-nope-1",
                                               "captcha_token": token, "captcha_answer": answer})
    assert r.status_code == 401
    with session_scope() as db:
        rows = db.scalars(select(PortalActivity).where(
            PortalActivity.event == "login_failed")).all()
        assert rows, "a failed sign-in must leave a trace"


def test_otp_completes_the_sign_in(client, stub_auth):
    client.post("/api/portal/signup", json=signup_body(client, email="full@gmail.com"))
    client.post("/api/portal/verify", json={"email": "full@gmail.com", "code": "123456"})
    r = client.post("/api/portal/otp-login", json={"email": "full@gmail.com", "code": "123456"})
    assert r.status_code == 200, r.text
    session = r.json()
    assert session["access_token"] and session["account"]["organisation"]["account_type"] == "airport"
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {session['access_token']}"})
    assert me.status_code == 200


def test_suspended_account_cannot_sign_in(client, stub_auth, admin):
    client.post("/api/portal/signup", json=signup_body(client, email="susp@gmail.com"))
    client.post("/api/portal/verify", json={"email": "susp@gmail.com", "code": "123456"})
    accounts = client.get("/api/portal/admin/accounts", headers=admin).json()["items"]
    acc = next(a for a in accounts if a["email"] == "susp@gmail.com")
    assert client.post(f"/api/portal/admin/accounts/{acc['id']}/status", headers=admin,
                       json={"status": "suspended"}).status_code == 200
    token, answer = solve(client)
    r = client.post("/api/portal/login", json={"email": "susp@gmail.com",
                                               "password": "Str0ngPass!2026",
                                               "captcha_token": token, "captcha_answer": answer})
    assert r.status_code == 403 and "suspended" in r.json()["detail"]


# --------------------------------------------------------- tenant isolation
def _session_for(client, email, organisation, account_type="airport"):
    client.post("/api/portal/signup", json=signup_body(client, email=email,
                                                       organisation=organisation,
                                                       account_type=account_type))
    client.post("/api/portal/verify", json={"email": email, "code": "123456"})
    r = client.post("/api/portal/otp-login", json={"email": email, "code": "123456"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_one_organisation_cannot_see_anothers_cameras(client, stub_auth, admin):
    airport = _session_for(client, "air@gmail.com", "Airport One", "airport")
    railway = _session_for(client, "rail@gmail.com", "Railway One", "railway_station")

    cam = client.post("/api/cameras", headers=airport, json={
        "name": "Airport Gate 4", "source_type": "synthetic", "source_uri": "",
        "width": 640, "height": 360, "fps": 10}).json()
    assert cam["meta"]["organisation_id"]

    mine = client.get("/api/cameras", headers=airport).json()["items"]
    assert [c["id"] for c in mine] == [cam["id"]]

    theirs = client.get("/api/cameras", headers=railway).json()["items"]
    assert cam["id"] not in [c["id"] for c in theirs], "railway can see an airport camera"
    assert client.get(f"/api/cameras/{cam['id']}", headers=railway).status_code == 404

    # the administrator sees across organisations
    assert cam["id"] in [c["id"] for c in client.get("/api/cameras", headers=admin).json()["items"]]


# ------------------------------------------------------------- admin panel
def test_admin_panel_is_invisible_to_operators(client, stub_auth):
    op = _session_for(client, "operator@gmail.com", "Some Airport")
    for path in ("/api/portal/admin/overview", "/api/portal/admin/accounts",
                 "/api/portal/admin/activity", "/api/portal/admin/organisations"):
        r = client.get(path, headers=op)
        assert r.status_code == 404, f"{path} leaked to an operator ({r.status_code})"


def test_admin_overview_counts_and_activity(client, stub_auth, admin):
    data = client.get("/api/portal/admin/overview", headers=admin).json()
    assert data["accounts"]["total"] >= 1
    assert data["organisations"]["total"] >= 1
    assert "signups" in data["last_24h"]
    assert isinstance(data["recent_activity"], list)


# --------------------------------------------------------------- the portal
def test_portal_pages_are_served(client):
    assert client.get("/portal/").status_code == 200
    for page in ("sign-in.html", "sign-up.html", "reset.html", "terms.html", "privacy.html"):
        r = client.get(f"/portal/{page}")
        assert r.status_code == 200, page
    terms = client.get("/portal/terms.html").text
    assert "educational research platform" in terms
    assert "not responsible" in terms.lower() or "no responsibility" in terms.lower()


def test_signup_page_has_no_google_button(client):
    page = client.get("/portal/sign-in.html").text.lower()
    assert "google" not in page and "sign in with" not in page


def test_only_vansh_is_listed_on_the_team(client):
    home = client.get("/portal/").text
    assert "Vansh Thakur" in home
    assert home.count('class="member"') == 1


def test_rate_limit_blocks_a_flood(client, stub_auth):
    """The sign-up limiter is what stops a script from creating accounts all day."""
    last = None
    for i in range(8):
        last = client.post("/api/portal/signup", json=signup_body(client, email=f"flood{i}@gmail.com"))
        if last.status_code == 429:
            break
    assert last.status_code == 429, "a sign-up flood was never rate-limited"
    assert "Too many" in last.json()["detail"]

#!/usr/bin/env python
"""Clone the VisionCoders_Auth n8n workflow into a Sentinel one.

The existing chain already does the hard parts properly (scrypt password
hashing, hashed one-time codes with TTL and attempt limits, email + SMS
delivery, Mongo storage). Rather than build a parallel auth service, this
copies it and changes only what must differ:

  * its own webhook namespace   /webhook/sentinel/*   (VisionCoders keeps auth/*)
  * its own Mongo collections   sentinel_*            (no shared user table)
  * Sentinel branding and site origin
  * signup additionally requires: account type (airport, railway station, ...),
    organisation name, and explicit acceptance of the terms
  * registration is limited to Gmail addresses (throwaway-inbox rule)

No credential values are touched: nodes keep referencing the same SMTP,
Twilio and MongoDB credentials by name, which stay inside n8n.
"""
from __future__ import annotations

import json
import re
import secrets
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "visioncoders_auth.json"
DST = HERE / "sentinel_auth.json"

BRAND = "Sentinel AI"
SITE = "eagleseye.microeagle.online"
ORIGIN = f"https://{SITE}"
WORKFLOW_NAME = "Sentinel_Auth"
WORKFLOW_ID = "SentinelAuth0001"

ACCOUNT_TYPES = ["airport", "railway_station", "metro_station", "bus_terminal",
                 "stadium", "mall", "campus", "police", "event", "other"]

SIGNUP_EXTRA = """
// --- Sentinel ----------------------------------------------------------
// An account belongs to ONE organisation of a known kind; operators only
// ever see their own organisation's cameras and incidents. The terms must
// be accepted before any record is created.
const ACCOUNT_TYPES = %s;
const accountType = String(b.account_type || '').trim().toLowerCase();
const organisation = String(b.organisation || '').trim().slice(0, 120);
const siteLabel = String(b.site_label || '').trim().slice(0, 120);
if (!ACCOUNT_TYPES.includes(accountType)) {
  return [{ json: { ok: false, status: 400,
    error: 'Choose the kind of site this account is for.' } }];
}
if (organisation.length < 2) {
  return [{ json: { ok: false, status: 400,
    error: 'Organisation name is required.' } }];
}
if (b.terms_accepted !== true && String(b.terms_accepted) !== 'true') {
  return [{ json: { ok: false, status: 400,
    error: 'You must accept the Terms and Conditions to register.' } }];
}
""" % json.dumps(ACCOUNT_TYPES)

GMAIL_RULE = """
// Throwaway inboxes defeat the point of verifying an address at all, so
// registration is limited to the domains in CFG.allowed_email_domains.
const allowedDomains = CFG.allowed_email_domains || [];
if (allowedDomains.length &&
    !allowedDomains.some(function (d) { return email.endsWith('@' + d); })) {
  return [{ json: { ok: false, status: 400,
    error: 'Please register with a Gmail address (' +
           allowedDomains.join(', ') + ').' } }];
}
"""


def patch_code(code: str, notify_key: str) -> str:
    code = code.replace('"brand": "VisionCoders"', f'"brand": "{BRAND}"')
    code = code.replace('"site": "visioncoders.in"', f'"site": "{SITE}"')
    code = code.replace('"site_origin": "https://visioncoders.in"',
                        f'"site_origin": "{ORIGIN}"')
    code = re.sub(r'"notify_key": "[^"]*"', f'"notify_key": "{notify_key}"', code)
    # config gains the domain allow-list, so it can be widened without a code edit
    code = code.replace('"min_password_length": 8',
                        '"min_password_length": 8, "allowed_email_domains": ["gmail.com"], '
                        '"terms_version": "v1.0"')
    code = code.replace("'/auth/verify-link?token='", "'/sentinel/verify-link?token='")
    for col in ("auth_users", "auth_verify", "auth_login", "auth_reset"):
        code = code.replace(col, col.replace("auth_", "sentinel_"))
    code = code.replace("'visioncoders'", "'sentinelai'")      # weak-password list
    return code


def main() -> int:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    wf = raw[0] if isinstance(raw, list) else raw
    notify_key = secrets.token_urlsafe(32)

    wf["name"] = WORKFLOW_NAME
    wf["id"] = WORKFLOW_ID
    wf["active"] = False                     # activated by the CLI after import
    wf.pop("versionId", None)
    wf.pop("meta", None)

    webhooks, mongos, codes = [], [], 0
    for node in wf["nodes"]:
        params = node.get("parameters", {})
        if "jsCode" in params:
            params["jsCode"] = patch_code(params["jsCode"], notify_key)
            codes += 1
        if node["type"].endswith("webhook"):
            path = params.get("path", "")
            if path.startswith("auth/"):
                params["path"] = "sentinel/" + path[len("auth/"):]
            node["webhookId"] = str(uuid.uuid4())       # must not clash with the original
            webhooks.append(f"{params.get('httpMethod', 'GET')} /webhook/{params['path']}")
        if node["type"].endswith("mongoDb"):
            col = params.get("collection", "")
            if col.startswith("auth_"):
                params["collection"] = "sentinel_" + col[len("auth_"):]
            mongos.append(params.get("collection"))
        # branding inside email bodies / responses
        for key in ("subject", "text", "html", "message", "toEmail", "fromEmail"):
            if isinstance(params.get(key), str):
                params[key] = (params[key]
                               .replace("VisionCoders", BRAND)
                               .replace("visioncoders.in", SITE))

    by = {n["name"]: n for n in wf["nodes"]}

    # 1. signup validation: account type, organisation, terms, Gmail only
    vs = by["Validate Signup"]["parameters"]
    code = vs["jsCode"]
    anchor = "const name = String(b.name || '').trim().slice(0, 80);"
    assert anchor in code, "signup anchor moved"
    code = code.replace(anchor, anchor + "\n" + SIGNUP_EXTRA, 1)
    mail_anchor = ("if (!validEmail(email)) {\n  return [{ json: { ok: false, status: 400, "
                   "error: 'A valid email is required.' } }];\n}")
    assert mail_anchor in code, "email anchor moved"
    code = code.replace(mail_anchor, mail_anchor + "\n" + GMAIL_RULE, 1)
    ret_anchor = "  ok: true, email, name, phone, phone_e164: e164(phone),"
    assert ret_anchor in code, "signup return moved"
    code = code.replace(ret_anchor, ret_anchor + """
  account_type: accountType, organisation, site_label: siteLabel,
  terms_version: CFG.terms_version, terms_accepted_at: new Date().toISOString(),""", 1)
    vs["jsCode"] = code

    # 2. the stored user row carries the organisation and the terms record
    sur = by["Shape User Row"]["parameters"]
    row_anchor = "  verified: false, failed_logins: 0, locked_until: '',"
    assert row_anchor in sur["jsCode"], "user row moved"
    sur["jsCode"] = sur["jsCode"].replace(row_anchor, """  verified: false, failed_logins: 0, locked_until: '',
  account_type: r.account_type || '', organisation: r.organisation || '',
  site_label: r.site_label || '', role: 'org_user', status: 'active',
  terms_version: r.terms_version || '', terms_accepted_at: r.terms_accepted_at || '',""", 1)

    uu = by["Upsert User"]["parameters"]
    uu["fields"] = uu["fields"] + (",account_type,organisation,site_label,role,status,"
                                   "terms_version,terms_accepted_at")

    DST.write_text(json.dumps([wf], indent=2), encoding="utf-8")
    print(f"wrote {DST.name}: {len(wf['nodes'])} nodes, {codes} code nodes patched")
    print("collections:", sorted(set(mongos)))
    print("endpoints:")
    for w in webhooks:
        print("  ", w)
    (HERE / "notify_key.txt").write_text(notify_key, encoding="utf-8")
    print("notify key written to deploy/n8n/notify_key.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

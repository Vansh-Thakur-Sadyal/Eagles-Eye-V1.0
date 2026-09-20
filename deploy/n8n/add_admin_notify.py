#!/usr/bin/env python
"""Add an admin-notify endpoint to the Sentinel auth workflow.

The inherited /notify hook only sends a fixed link mail. The platform also has
to tell the administrator, in words, that someone signed up and what they
accepted. This adds:

    POST /webhook/sentinel/admin-notify
    header  x-auth-key: <notify key>
    body    { "to": "...", "subject": "...", "text": "..." }

It reuses the SMTP credential already in n8n by name; no secret is read here.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
WF = HERE / "sentinel_auth.json"
KEY_FILE = HERE / "notify_key.txt"

GUARD_CODE = """
const crypto = require('crypto');
const KEY = %s;

function body() {
  const b = $input.first().json;
  return (b && b.body) || b || {};
}
function headers() {
  const b = $input.first().json;
  return (b && b.headers) || {};
}

// Compare digests, not strings: timingSafeEqual throws on a length mismatch,
// and that throw would itself leak the secret's length.
const given = crypto.createHash('sha256')
  .update(String(headers()['x-auth-key'] || '')).digest();
const want = crypto.createHash('sha256').update(KEY).digest();
if (!crypto.timingSafeEqual(given, want)) {
  return [{ json: { ok: false, status: 401, error: 'Not authorised.' } }];
}

const b = body();
const to = String(b.to || '').trim().toLowerCase();
if (!/^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$/.test(to)) {
  return [{ json: { ok: false, status: 400, error: 'A valid recipient is required.' } }];
}
const subject = String(b.subject || 'Sentinel AI notification').slice(0, 200);
const text = String(b.text || '').slice(0, 5000);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const html = '<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;'
  + 'font-size:15px;line-height:1.6;color:#222;max-width:640px">'
  + '<h2 style="margin:0 0 12px">' + esc(subject) + '</h2>'
  + '<pre style="white-space:pre-wrap;font:inherit;background:#f6f8f7;'
  + 'border:1px solid #e3e8e6;border-radius:8px;padding:12px">' + esc(text) + '</pre>'
  + '<p style="color:#667;font-size:13px">Sentinel AI - automated notice</p></div>';
return [{ json: { ok: true, to, subject, html } }];
"""


def main() -> int:
    data = json.loads(WF.read_text(encoding="utf-8"))
    wf = data[0] if isinstance(data, list) else data
    if any(n["name"] == "Admin Notify In" for n in wf["nodes"]):
        print("already present")
        return 0
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    smtp = next(n for n in wf["nodes"] if n["type"].endswith("emailSend"))
    y = 6200

    nodes = [
        {"parameters": {"httpMethod": "POST", "path": "sentinel/admin-notify",
                        "responseMode": "responseNode", "options": {}},
         "id": str(uuid.uuid4()), "name": "Admin Notify In",
         "type": "n8n-nodes-base.webhook", "typeVersion": 2,
         "position": [-1200, y], "webhookId": str(uuid.uuid4())},
        {"parameters": {"jsCode": GUARD_CODE % json.dumps(key)},
         "id": str(uuid.uuid4()), "name": "Check Admin Key",
         "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [-980, y]},
        {"parameters": {"conditions": {"options": {"caseSensitive": True, "version": 2},
                                       "conditions": [{"id": str(uuid.uuid4()),
                                                       "operator": {"type": "boolean",
                                                                    "operation": "true",
                                                                    "singleValue": True},
                                                       "leftValue": "={{ $json.ok }}",
                                                       "rightValue": ""}],
                                       "combinator": "and"},
                        "options": {}},
         "id": str(uuid.uuid4()), "name": "Admin Key Ok?",
         "type": "n8n-nodes-base.if", "typeVersion": 2.2, "position": [-760, y]},
        {"parameters": {"fromEmail": "support@visioncoders.online",
                        "toEmail": "={{ $json.to }}", "subject": "={{ $json.subject }}",
                        "emailFormat": "html", "html": "={{ $json.html }}",
                        "options": {"appendAttribution": False}},
         "id": str(uuid.uuid4()), "name": "Send Admin Notice",
         "type": smtp["type"], "typeVersion": smtp["typeVersion"],
         "position": [-540, y - 60], "credentials": smtp["credentials"]},
        {"parameters": {"respondWith": "json",
                        "responseBody": "={{ JSON.stringify({ ok: true, sent: true }) }}",
                        "options": {"responseCode": 200}},
         "id": str(uuid.uuid4()), "name": "Admin Notice Done",
         "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1,
         "position": [-320, y - 60]},
        {"parameters": {"respondWith": "json",
                        "responseBody": "={{ JSON.stringify({ ok: false, error: $json.error }) }}",
                        "options": {"responseCode": 401}},
         "id": str(uuid.uuid4()), "name": "Admin Notice Refused",
         "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1,
         "position": [-540, y + 80]},
    ]
    wf["nodes"].extend(nodes)
    wf["connections"].update({
        "Admin Notify In": {"main": [[{"node": "Check Admin Key", "type": "main", "index": 0}]]},
        "Check Admin Key": {"main": [[{"node": "Admin Key Ok?", "type": "main", "index": 0}]]},
        "Admin Key Ok?": {"main": [
            [{"node": "Send Admin Notice", "type": "main", "index": 0}],
            [{"node": "Admin Notice Refused", "type": "main", "index": 0}]]},
        "Send Admin Notice": {"main": [[{"node": "Admin Notice Done", "type": "main", "index": 0}]]},
    })
    WF.write_text(json.dumps([wf], indent=2), encoding="utf-8")
    print(f"added admin-notify ({len(wf['nodes'])} nodes total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

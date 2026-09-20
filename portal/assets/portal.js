/* Shared helpers for the portal forms.
 *
 * Everything talks to /api/portal on this same origin; the platform adds the
 * captcha check, the rate limit and the audit trail, then forwards to the n8n
 * auth service which owns the passwords and the one-time codes. No secret and
 * no code is ever held in the browser.
 */
const API = '/api/portal';

async function api(path, body) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await r.json(); } catch (e) { /* non-JSON error page */ }
  if (!r.ok) {
    throw new Error(data.detail || data.error || 'Something went wrong. Try again.');
  }
  return data;
}

function show(el, text, kind) {
  el.textContent = text;
  el.className = 'msg ' + (kind || 'err');
}

function clearMsg(el) { el.textContent = ''; el.className = 'msg'; }

/** Fetch a fresh anti-robot challenge into the given question/token elements. */
async function loadCaptcha(qEl, tokenEl, answerEl) {
  try {
    const r = await fetch(API + '/captcha');
    const c = await r.json();
    qEl.textContent = c.question;
    tokenEl.value = c.token;
    if (answerEl) answerEl.value = '';
  } catch (e) {
    qEl.textContent = 'Could not load the check. Reload the page.';
  }
}

function busy(btn, on, label) {
  btn.disabled = on;
  if (on) { btn.dataset.label = btn.textContent; btn.textContent = label || 'Working…'; }
  else if (btn.dataset.label) { btn.textContent = btn.dataset.label; }
}

/** Store the session the dashboard expects, then hand over to it. */
function enterDashboard(session) {
  try {
    localStorage.setItem('sentinel.token', session.access_token);
    localStorage.setItem('sentinel.account', JSON.stringify(session.account || {}));
  } catch (e) { /* private window: the token below still carries the session */ }
  window.location.href = '/app/#token=' + encodeURIComponent(session.access_token);
}

function stepTo(id) {
  document.querySelectorAll('.step').forEach((s) => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

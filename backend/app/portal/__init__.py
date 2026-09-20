"""Public sign-up / sign-in portal, organisations and activity tracking.

The identity store itself lives in the n8n auth workflow (Mongo, scrypt
passwords, e-mail + SMS one-time codes). This package is the platform side of
it: it proxies those calls so every attempt is rate-limited, captcha-gated and
audited, mirrors verified accounts into the local database as Organisation +
User rows, and scopes what each organisation is allowed to see.
"""

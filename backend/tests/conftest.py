"""Test-wide isolation.

Loaded by pytest before any test module imports the app, so these land before
Settings is built and cached.

The suite must not depend on a network service: with a real provider
configured in backend/.env, the commander's narration would call it on every
incident - slow, flaky, and billed. The deterministic-template path is what
the assertions are written against, and it is also the path that must keep
working when a provider is down.
"""
from __future__ import annotations

import os

os.environ["SENTINEL_LLM_PROVIDER"] = "none"
os.environ["SENTINEL_LLM_API_KEY"] = ""
# The 3B VLM is an optional, licence-restricted extra; the suite exercises the
# detector paths that always exist. tests/test_edge.py covers its parser.
os.environ.setdefault("SENTINEL_DETECTOR", "yolo")
os.environ.setdefault("SENTINEL_LOCATEANYTHING_ENABLED", "false")

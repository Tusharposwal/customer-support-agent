"""Behavioural checks of the chat page script (static/app.js).

Runs tests/js/app_smoke.js with Node.js: a small fake DOM that fails on any
innerHTML-style insertion, and a fake fetch (no server, no network). Covers
restoring a conversation (with HTML-looking text shown as text), sending the
conversation ID, 404/409/503 handling, and "New conversation". Skipped when
Node.js is not installed; nothing is downloaded or installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_chat_page_script_behaviour():
    result = subprocess.run(
        [NODE, str(ROOT / "tests" / "js" / "app_smoke.js"), str(ROOT / "src" / "support_agent" / "static" / "app.js")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL FRONTEND SMOKE CHECKS PASSED" in result.stdout

"""Runtime endpoint/model configuration, overridable via environment variables.

Defaults match the setup used for the published results: the target (Muse Glimmer) on :8001 and the
Gemma judge/author model on :8000. Override with COTI_TARGET_HOST, COTI_TARGET_BASE_URL, COTI_TARGET_MODEL,
COTI_JUDGE_HOST, COTI_JUDGE_BASE_URL, COTI_JUDGE_MODEL.
"""
import os
import sys

TARGET_HOST = os.environ.get("COTI_TARGET_HOST", "http://127.0.0.1:8001")  # Muse Glimmer server root (no /v1)
TARGET_BASE_URL = os.environ.get("COTI_TARGET_BASE_URL", TARGET_HOST + "/v1")
TARGET_MODEL = os.environ.get("COTI_TARGET_MODEL", "muse-glimmer")

JUDGE_HOST = os.environ.get("COTI_JUDGE_HOST", "http://127.0.0.1:8000")  # Gemma judge/author server root
JUDGE_BASE_URL = os.environ.get("COTI_JUDGE_BASE_URL", JUDGE_HOST + "/v1")
JUDGE_MODEL = os.environ.get("COTI_JUDGE_MODEL", "gemma-4-31b-it-qat-w4a16-ct")


def guard_base_url(url):
    """Refuse the :8000 (Gemma/author) server unless ALLOW_8000=1 is set."""
    if url and ("127.0.0.1:8000" in url or "localhost:8000" in url) and os.environ.get("ALLOW_8000") != "1":
        sys.exit("refusing to use :8000 (Gemma/author server) without ALLOW_8000=1")

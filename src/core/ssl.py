"""Utilities for relaxing SSL certificate verification in restricted environments.

This module applies a snippet that disables HTTPS certificate
verification for urllib3 / requests clients. It should only be used in
situations where the network perimeter breaks standard certificate validation.
"""

from __future__ import annotations

import os
import ssl
from typing import Optional


def disable_ssl_verification() -> None:
    """Disable SSL verification for urllib3 and requests (idempotent)."""
    try:
        import urllib3  # type: ignore
        import requests  # type: ignore
    except Exception:
        return

    # Set environment flags consumed by SSL libraries / huggingface_hub
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    os.environ["GIT_SSL_NO_VERIFY"] = "1"
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["HF_HUB_ENABLE_XET"] = "0"

    if getattr(disable_ssl_verification, "_patched", False):
        return

    try:
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
    except Exception:
        pass

    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    try:
        from huggingface_hub import configure_http_backend  # type: ignore
        from huggingface_hub import file_download as hf_file_download  # type: ignore
        from huggingface_hub.utils import _xet as hf_xet_utils  # type: ignore

        def backend_factory() -> requests.Session:  # type: ignore
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory)

        try:
            hf_file_download.is_xet_available = lambda *_, **__: False  # type: ignore
            if hasattr(hf_file_download, "XET_AVAILABLE"):
                hf_file_download.XET_AVAILABLE = False  # type: ignore[attr-defined]
            if hasattr(hf_file_download, "_XET_AVAILABLE"):
                hf_file_download._XET_AVAILABLE = False  # type: ignore[attr-defined]
            hf_xet_utils.is_available = lambda *_, **__: False  # type: ignore
        except Exception:
            pass
    except Exception:
        pass

    try:
        import httpx  # type: ignore

        if not hasattr(disable_ssl_verification, "_httpx_patched"):

            original_client = httpx.Client

            class PatchedClient(httpx.Client):
                def __init__(self, *args, **kwargs):
                    kwargs.setdefault("verify", False)
                    super().__init__(*args, **kwargs)

            httpx.Client = PatchedClient  # type: ignore
            setattr(disable_ssl_verification, "_httpx_original", original_client)
            setattr(disable_ssl_verification, "_httpx_patched", True)
    except Exception:
        pass

    original_request: Optional[requests.Session.request] = getattr(
        disable_ssl_verification, "_original_request", None
    )  # type: ignore
    if original_request is None:
        original_request = requests.Session.request
        setattr(disable_ssl_verification, "_original_request", original_request)

        def patched_request(self, *args, **kwargs):  # type: ignore
            kwargs.setdefault("verify", False)
            return original_request(self, *args, **kwargs)

        try:
            requests.Session.request = patched_request  # type: ignore
        except Exception:
            pass

    setattr(disable_ssl_verification, "_patched", True)

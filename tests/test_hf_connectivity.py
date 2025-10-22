"""Integration tests to validate Hugging Face access with SSL disabled."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from src.core.ssl import disable_ssl_verification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
MODEL_ID = "Qwen/Qwen2.5-0.5B"


def _load_hf_token() -> str | None:
    """Load HF token from environment or project .env and propagate to env vars."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token and ENV_PATH.exists():
        env_vars = dotenv_values(ENV_PATH)
        token = env_vars.get("HF_TOKEN") or env_vars.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return token


def _require_token() -> str:
    token = _load_hf_token()
    if not token:
        pytest.skip("HF_TOKEN not configured in environment or .env file")
    return token


@pytest.mark.network
@pytest.mark.integration
def test_hf_api_model_info_accessible():
    """Ensure model metadata can be retrieved with SSL verification disabled."""
    disable_ssl_verification()
    token = _require_token()
    api = HfApi()
    info = api.model_info(MODEL_ID, token=token, timeout=30)
    assert info.modelId == MODEL_ID
    assert info.sha is not None  # basic sanity check that metadata is returned


@pytest.mark.network
@pytest.mark.integration
def test_auto_tokenizer_downloads_with_ssl_disabled(tmp_path):
    """Ensure AutoTokenizer can download gated model assets when SSL is disabled."""
    disable_ssl_verification()
    token = _require_token()
    cache_dir = tmp_path / "hf_cache"
    config_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="config.json",
        token=token,
        cache_dir=str(cache_dir),
    )
    tokenizer_json = hf_hub_download(
        repo_id=MODEL_ID,
        filename="tokenizer.json",
        token=token,
        cache_dir=str(cache_dir),
    )
    assert Path(config_path).exists()
    assert Path(tokenizer_json).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=token,
        cache_dir=str(cache_dir),
    )
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
    assert vocab_size > 0
    assert tokenizer.eos_token is not None
    assert isinstance(getattr(tokenizer, "chat_template", None), str)

    encoded = tokenizer("quick connectivity check")
    assert encoded["input_ids"], "Tokenizer returned empty input_ids"
    decoded = tokenizer.decode(encoded["input_ids"])
    assert "quick" in decoded.lower()

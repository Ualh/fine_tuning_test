"""Minimal OpenAI-compatible client for local vLLM smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class VLLMClient:
    endpoint: str
    model: str
    timeout: float = 30.0

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        url = f"{self.endpoint.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data

    def healthcheck(self) -> bool:
        try:
            response = requests.get(f"{self.endpoint.rstrip('/')}/v1/models", timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

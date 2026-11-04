"""Narrow loopback-only Ollama JSON client."""

from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ModelInvocationError(RuntimeError):
    """Raised when the local model endpoint cannot return usable content."""


@dataclass(frozen=True, slots=True)
class OllamaJsonModel:
    base_url: str
    model_id: str
    timeout_seconds: int = 90

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama endpoint must be loopback HTTP")
        if not self.model_id or self.model_id.endswith(":cloud"):
            raise ValueError("MissionPay requires an explicit non-cloud Ollama model")
        if self.timeout_seconds < 1 or self.timeout_seconds > 300:
            raise ValueError("timeout_seconds must be between 1 and 300")

    def complete_json(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": 7},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                outer = json.loads(response.read())
            content = outer["message"]["content"]
        except Exception as exc:
            raise ModelInvocationError("LOCAL_MODEL_REQUEST_FAILED") from exc
        if not isinstance(content, str) or not content:
            raise ModelInvocationError("LOCAL_MODEL_RESPONSE_INVALID")
        return content

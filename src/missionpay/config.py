"""Explicit runtime configuration with safe defaults and sandbox-only enforcement.

Two independent switches are kept separate because they carry different risk:

* ``MISSIONPAY_PROVIDER_MODE`` selects how a payout is executed. The default is
  ``mock``: nothing leaves the process. ``paypal_sandbox`` is the only genuine
  mode and it refuses to start unless every precondition below is satisfied.
* ``MISSIONPAY_AI_MODE`` selects how evidence is interpreted. The default is
  ``fixture``: a deterministic, labelled stand-in. ``ollama`` talks to a
  loopback-only local model.

A configuration that cannot be honoured raises :class:`ConfigurationError` at
startup rather than degrading silently. Secrets are held with ``repr=False`` and
:meth:`RuntimeConfig.describe` never returns them, so a config object can be
logged, echoed in an error page, or printed in a traceback without disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Mapping
from urllib.parse import urlparse

PAYPAL_SANDBOX_ORIGIN = "https://api-m.sandbox.paypal.com"

#: The only hosted model origin this build will send a credential to.
HOSTED_ORIGIN = "https://api.openai.com"

#: Bounds that decide what one hosted request can cost. Validated at construction.
HOSTED_MAX_OUTPUT_TOKENS_CEILING = 4096
HOSTED_MAX_TOTAL_TOKENS_CEILING = 32768
HOSTED_MAX_TIMEOUT_SECONDS = 120
HOSTED_MAX_LATE_RESPONSE_SECONDS = 600

#: Origins a genuine run must never reach. The exact-origin comparison below is
#: the primary guard; this set makes the intent explicit and catches a caller that
#: swaps the URL after construction.
_PRODUCTION_HOSTS = frozenset({"api-m.paypal.com", "api.paypal.com"})

_ALLOWED_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class ConfigurationError(ValueError):
    """Raised when the requested runtime configuration cannot be honoured safely."""


class ProviderMode(str, Enum):
    MOCK = "mock"
    PAYPAL_SANDBOX = "paypal_sandbox"


class AiMode(str, Enum):
    FIXTURE = "fixture"
    OLLAMA = "ollama"
    HOSTED = "hosted"


def _flag(value: str | None) -> bool:
    return (value or "").strip() == "1"


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved configuration. Safe to log: secrets are redacted from repr."""

    provider_mode: ProviderMode = ProviderMode.MOCK
    ai_mode: AiMode = AiMode.FIXTURE
    database_path: str = "missionpay-demo.sqlite3"
    secure_cookies: bool = True
    paypal_base_url: str = PAYPAL_SANDBOX_ORIGIN
    paypal_client_id: str | None = field(default=None, repr=False)
    paypal_client_secret: str | None = field(default=None, repr=False)
    sandbox_recipient: str | None = None
    max_payout_cents: int | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    hosted_base_url: str = HOSTED_ORIGIN
    hosted_model: str | None = None
    hosted_api_key: str | None = field(default=None, repr=False)
    hosted_max_output_tokens: int | None = None
    #: Total ceiling for one request: prompt estimate plus the output ceiling. Bounded
    #: before sending, because a post-response check cannot undo a spend.
    hosted_max_total_tokens: int | None = None
    #: Bound on a single blocking socket operation.
    hosted_timeout_seconds: int = 45
    #: Threshold after which an answer that did arrive is refused for arriving late.
    #: This is late-response rejection, not a total deadline: the read is bounded by the
    #: socket timeout above, and nothing interrupts a call that is still running.
    hosted_late_response_seconds: int = 45
    hosted_confirmed: bool = False
    demo_bootstrap: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.provider_mode, str):
            object.__setattr__(self, "provider_mode", ProviderMode(self.provider_mode))
        if isinstance(self.ai_mode, str):
            object.__setattr__(self, "ai_mode", AiMode(self.ai_mode))

        if self.provider_mode is ProviderMode.PAYPAL_SANDBOX:
            self._validate_sandbox()

        if self.ai_mode is AiMode.OLLAMA:
            self._validate_ollama()

        if self.ai_mode is AiMode.HOSTED:
            self._validate_hosted()

    def _validate_sandbox(self) -> None:
        if not self.paypal_client_id or not self.paypal_client_secret:
            raise ConfigurationError("PAYPAL_CREDENTIALS_REQUIRED")
        self._validate_sandbox_origin(self.paypal_base_url)
        if not self.sandbox_recipient:
            raise ConfigurationError("SANDBOX_RECIPIENT_REQUIRED")
        if self.max_payout_cents is None:
            raise ConfigurationError("MAX_PAYOUT_CENTS_REQUIRED")

    @staticmethod
    def _validate_sandbox_origin(base_url: str) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or parsed.hostname in _PRODUCTION_HOSTS:
            raise ConfigurationError("PAYPAL_SANDBOX_ORIGIN_REQUIRED")
        if base_url.rstrip("/") != PAYPAL_SANDBOX_ORIGIN:
            raise ConfigurationError("PAYPAL_SANDBOX_ORIGIN_REQUIRED")

    def _validate_ollama(self) -> None:
        if not self.ollama_model:
            raise ConfigurationError("OLLAMA_MODEL_REQUIRED")
        if self.ollama_model.endswith(":cloud"):
            raise ConfigurationError("CLOUD_MODEL_REFUSED")
        parsed = urlparse(self.ollama_base_url)
        if parsed.scheme != "http" or parsed.hostname not in _ALLOWED_LOOPBACK:
            raise ConfigurationError("OLLAMA_LOOPBACK_REQUIRED")

    def _validate_hosted(self) -> None:
        """All-or-nothing, and unreachable without a separate confirmation.

        A remote paid path must never be reachable by setting one environment variable:
        a mode typo, an inherited shell or a copied .env would otherwise start spending
        money and transmitting a donor's receipt. ``MISSIONPAY_HOSTED_AI_CONFIRM=1`` is
        that second key, and it is checked first so the error names the real problem.
        """
        if not self.hosted_confirmed:
            raise ConfigurationError("HOSTED_AI_NOT_CONFIRMED")
        if not self.hosted_api_key:
            raise ConfigurationError("HOSTED_AI_KEY_REQUIRED")
        if not self.hosted_model:
            raise ConfigurationError("HOSTED_MODEL_REQUIRED")
        if self.hosted_base_url.rstrip("/") != HOSTED_ORIGIN:
            raise ConfigurationError("HOSTED_ORIGIN_REQUIRED")
        if self.hosted_max_output_tokens is None:
            raise ConfigurationError("HOSTED_MAX_OUTPUT_TOKENS_REQUIRED")
        if not 1 <= self.hosted_max_output_tokens <= HOSTED_MAX_OUTPUT_TOKENS_CEILING:
            raise ConfigurationError("HOSTED_MAX_OUTPUT_TOKENS_INVALID")
        if self.hosted_max_total_tokens is None:
            raise ConfigurationError("HOSTED_MAX_TOTAL_TOKENS_REQUIRED")
        if not (
            self.hosted_max_output_tokens
            < self.hosted_max_total_tokens
            <= HOSTED_MAX_TOTAL_TOKENS_CEILING
        ):
            # A total ceiling that cannot hold one full answer is a configuration
            # mistake, not a budget: it would refuse every request before sending.
            raise ConfigurationError("HOSTED_MAX_TOTAL_TOKENS_INVALID")
        if not 1 <= self.hosted_timeout_seconds <= HOSTED_MAX_TIMEOUT_SECONDS:
            raise ConfigurationError("HOSTED_TIMEOUT_INVALID")
        if not 1 <= self.hosted_late_response_seconds <= HOSTED_MAX_LATE_RESPONSE_SECONDS:
            raise ConfigurationError("HOSTED_LATE_RESPONSE_LIMIT_INVALID")

    # -- reporting -------------------------------------------------------

    @property
    def is_genuine(self) -> bool:
        return self.provider_mode is ProviderMode.PAYPAL_SANDBOX

    @property
    def mode_label(self) -> str:
        if self.provider_mode is ProviderMode.PAYPAL_SANDBOX:
            return "PAYPAL SANDBOX — GENUINE PROVIDER CALLS"
        return "MOCK — NO PROVIDER CALLS, NO MONEY MOVES"

    def describe(self) -> dict[str, object]:
        """Non-secret description, safe for logs, pages and evidence capture."""
        return {
            "provider_mode": self.provider_mode.value,
            "ai_mode": self.ai_mode.value,
            "paypal_base_url": self.paypal_base_url,
            "paypal_credentials_present": bool(
                self.paypal_client_id and self.paypal_client_secret
            ),
            "sandbox_recipient": self.sandbox_recipient,
            "max_payout_cents": self.max_payout_cents,
            "ollama_base_url": self.ollama_base_url if self.ai_mode is AiMode.OLLAMA else None,
            "ollama_model": self.ollama_model if self.ai_mode is AiMode.OLLAMA else None,
            "hosted_provider": "openai" if self.ai_mode is AiMode.HOSTED else None,
            "hosted_base_url": self.hosted_base_url if self.ai_mode is AiMode.HOSTED else None,
            "hosted_model": self.hosted_model if self.ai_mode is AiMode.HOSTED else None,
            "hosted_api_key_present": bool(self.hosted_api_key),
            "hosted_max_output_tokens": (
                self.hosted_max_output_tokens if self.ai_mode is AiMode.HOSTED else None
            ),
            "hosted_max_total_tokens": (
                self.hosted_max_total_tokens if self.ai_mode is AiMode.HOSTED else None
            ),
            "hosted_late_response_seconds": (
                self.hosted_late_response_seconds if self.ai_mode is AiMode.HOSTED else None
            ),
            "hosted_timeout_seconds": (
                self.hosted_timeout_seconds if self.ai_mode is AiMode.HOSTED else None
            ),
            "secure_cookies": self.secure_cookies,
        }


def from_environment(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Build a config from the process environment, failing closed."""
    env = os.environ if environ is None else environ

    raw_mode = (env.get("MISSIONPAY_PROVIDER_MODE") or "mock").strip().lower()
    try:
        provider_mode = ProviderMode(raw_mode)
    except ValueError as exc:
        raise ConfigurationError(f"UNKNOWN_PROVIDER_MODE:{raw_mode}") from exc

    # A production/live switch is a hard refusal, not a fallback.
    paypal_env = (env.get("PAYPAL_ENVIRONMENT") or env.get("PAYPAL_MODE") or "").strip().lower()
    if paypal_env in {"live", "production"}:
        raise ConfigurationError("PAYPAL_PRODUCTION_REFUSED")

    raw_ai = (env.get("MISSIONPAY_AI_MODE") or "fixture").strip().lower()
    try:
        ai_mode = AiMode(raw_ai)
    except ValueError as exc:
        raise ConfigurationError(f"UNKNOWN_AI_MODE:{raw_ai}") from exc

    max_cents_raw = (env.get("MISSIONPAY_MAX_PAYOUT_CENTS") or "").strip()

    # Credentials may come from the private file the owner fills in, or the
    # environment (which wins). A credential is never taken from an argument.
    #
    # The file is opened only when a credential could actually be used: a genuine
    # PayPal run, or a hosted model call. Mock mode and the fixture/Ollama AI modes
    # need no secret, so touching the file there would be an unnecessary read of a
    # secret-bearing path and would surface its parse errors in a mode that cannot
    # use the result.
    needs_credentials = (
        provider_mode is ProviderMode.PAYPAL_SANDBOX or ai_mode is AiMode.HOSTED
    )
    supplied: dict[str, str] = {}
    if needs_credentials:
        from .credentials import resolve as resolve_credentials

        supplied = resolve_credentials(env)

    client_id = (supplied.get("PAYPAL_CLIENT_ID") or "").strip() or None
    client_secret = (supplied.get("PAYPAL_CLIENT_SECRET") or "").strip() or None
    recipient = (env.get("MISSIONPAY_SANDBOX_RECIPIENT") or supplied.get("MISSIONPAY_SANDBOX_RECIPIENT") or "").strip() or None
    if not max_cents_raw:
        max_cents_raw = (supplied.get("MISSIONPAY_MAX_PAYOUT_CENTS") or "").strip()
    base_url = (env.get("PAYPAL_BASE_URL") or supplied.get("PAYPAL_BASE_URL") or PAYPAL_SANDBOX_ORIGIN).strip()

    max_cents = _positive_int(max_cents_raw, "MISSIONPAY_MAX_PAYOUT_CENTS") if max_cents_raw else None

    hosted_tokens_raw = (env.get("MISSIONPAY_HOSTED_MAX_OUTPUT_TOKENS") or "").strip()
    hosted_tokens = (
        _positive_int(hosted_tokens_raw, "MISSIONPAY_HOSTED_MAX_OUTPUT_TOKENS")
        if hosted_tokens_raw
        else None
    )
    hosted_total_raw = (env.get("MISSIONPAY_HOSTED_MAX_TOTAL_TOKENS") or "").strip()
    hosted_total = (
        _positive_int(hosted_total_raw, "MISSIONPAY_HOSTED_MAX_TOTAL_TOKENS")
        if hosted_total_raw
        else None
    )
    hosted_timeout_raw = (env.get("MISSIONPAY_HOSTED_TIMEOUT_SECONDS") or "").strip()
    hosted_timeout = (
        _positive_int(hosted_timeout_raw, "MISSIONPAY_HOSTED_TIMEOUT_SECONDS")
        if hosted_timeout_raw
        else 45
    )
    hosted_late_raw = (env.get("MISSIONPAY_HOSTED_LATE_RESPONSE_SECONDS") or "").strip()
    hosted_late_response = (
        _positive_int(hosted_late_raw, "MISSIONPAY_HOSTED_LATE_RESPONSE_SECONDS")
        if hosted_late_raw
        else hosted_timeout
    )

    return RuntimeConfig(
        provider_mode=provider_mode,
        ai_mode=ai_mode,
        database_path=(env.get("MISSIONPAY_DB") or "missionpay-demo.sqlite3").strip(),
        secure_cookies=not _flag(env.get("MISSIONPAY_ALLOW_INSECURE_LOCALHOST")),
        paypal_base_url=base_url,
        paypal_client_id=client_id,
        paypal_client_secret=client_secret,
        sandbox_recipient=recipient,
        max_payout_cents=max_cents,
        ollama_base_url=(env.get("MISSIONPAY_OLLAMA_URL") or "http://127.0.0.1:11434").strip(),
        ollama_model=(env.get("MISSIONPAY_OLLAMA_MODEL") or "").strip() or None,
        hosted_base_url=(env.get("MISSIONPAY_HOSTED_BASE_URL") or supplied.get("MISSIONPAY_HOSTED_BASE_URL") or HOSTED_ORIGIN).strip(),
        hosted_model=(env.get("MISSIONPAY_HOSTED_MODEL") or supplied.get("MISSIONPAY_HOSTED_MODEL") or "").strip() or None,
        hosted_api_key=(env.get("MISSIONPAY_HOSTED_API_KEY") or supplied.get("MISSIONPAY_HOSTED_API_KEY") or "").strip() or None,
        hosted_max_output_tokens=hosted_tokens,
        hosted_timeout_seconds=hosted_timeout,
        hosted_max_total_tokens=hosted_total,
        hosted_late_response_seconds=hosted_late_response,
        hosted_confirmed=_flag(env.get("MISSIONPAY_HOSTED_AI_CONFIRM")),
        demo_bootstrap=_flag(env.get("MISSIONPAY_DEMO_BOOTSTRAP")),
    )

"""PayPal sandbox OAuth 2.0 client-credentials token acquisition.

This module exists so that :mod:`missionpay.paypal` can be handed a bearer token
by a component that is allowed to hold a client secret. It is deliberately the
only place a secret is used, and it is written so that no failure path can put
the secret into a message, an exception, a repr, or a serialised structure.

Security properties enforced here and covered by tests:

* the origin is pinned to ``https://api-m.sandbox.paypal.com`` and a production
  origin is refused at construction time, not at request time;
* the secret appears only inside one ``Authorization: Basic`` header value;
* :class:`PayPalOAuthError` carries a status code and an optional provider error
  code, never a response body and never credential material;
* a cached token is reused until shortly before expiry, so a single run does not
  mint a token per request.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
from typing import Protocol
from urllib.parse import urlparse

from .config import PAYPAL_SANDBOX_ORIGIN

#: Refresh this many seconds before the provider's stated expiry.
_EXPIRY_SKEW_SECONDS = 60

_TOKEN_PATH = "/v1/oauth2/token"


class PayPalOAuthError(RuntimeError):
    """Token acquisition failed. The message never contains credential material."""


class TokenTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, bytes]: ...


#: The exact OAuth scope token PayPal documents for Payouts. Compared as a whole
#: token, never as a substring: ``"openid notpayouts"`` must not satisfy this.
PAYOUTS_SCOPE = "https://uri.paypal.com/payments/payouts"


class PayoutsEntitlement(str, Enum):
    """What a token can and cannot tell us about the ability to pay.

    Authentication and scope are observable from the token response. Actual Payouts
    entitlement is a PayPal-side account decision that only a real payout attempt
    can confirm, so it is reported as untested rather than inferred.
    """

    NOT_TESTED = "not_tested"


@dataclass(frozen=True, slots=True)
class OAuthToken:
    """A bearer token whose value is excluded from repr and equality display."""

    access_token: str = field(repr=False)
    token_type: str
    expires_at: datetime
    scope_tokens: frozenset[str] = frozenset()
    scope_verified: bool = False

    def is_usable_at(self, moment: datetime) -> bool:
        return moment + timedelta(seconds=_EXPIRY_SKEW_SECONDS) < self.expires_at

    def describe(self) -> dict[str, object]:
        """Non-secret summary, safe to print. Never claims entitlement."""
        return {
            "authentication": "ok",
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat(),
            "scope_verified": self.scope_verified,
            "payouts_scope_present": PAYOUTS_SCOPE in self.scope_tokens,
            "payouts_entitlement": PayoutsEntitlement.NOT_TESTED.value,
        }


@dataclass(slots=True)
class PayPalClientCredentials:
    """Acquire and cache sandbox bearer tokens for the Payouts API."""

    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    base_url: str = PAYPAL_SANDBOX_ORIGIN
    transport: TokenTransport | None = None
    _cached: OAuthToken | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.client_id or not self.client_secret:
            raise PayPalOAuthError("PAYPAL_CREDENTIALS_REQUIRED")
        self._validate_origin(self.base_url)
        if self.transport is None:
            from .paypal import UrllibTransport  # local import keeps cycles out

            self.transport = UrllibTransport()

    @staticmethod
    def _validate_origin(base_url: str) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or base_url.rstrip("/") != PAYPAL_SANDBOX_ORIGIN:
            raise PayPalOAuthError("PAYPAL_SANDBOX_ORIGIN_REQUIRED")

    def _basic_authorization(self) -> str:
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        return "Basic " + b64encode(raw).decode("ascii")

    def access_token(self, *, now: datetime | None = None) -> str:
        """Return a usable bearer token, acquiring one only when necessary."""
        moment = now or datetime.now(timezone.utc)
        cached = self._cached
        if cached is not None and cached.is_usable_at(moment):
            return cached.access_token

        token = self._acquire(moment)
        self._cached = token
        return token.access_token

    def cached_token(self) -> OAuthToken | None:
        """The last acquired token, for non-secret reporting. Never a new request."""
        return self._cached

    def _acquire(self, moment: datetime) -> OAuthToken:
        headers = {
            "Authorization": self._basic_authorization(),
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = b"grant_type=client_credentials"
        assert self.transport is not None
        status, raw = self.transport.request(
            "POST", self.base_url.rstrip("/") + _TOKEN_PATH, headers, body
        )
        if status < 200 or status >= 300:
            raise PayPalOAuthError(f"PAYPAL_OAUTH_HTTP_{status}")

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise PayPalOAuthError("PAYPAL_OAUTH_RESPONSE_INVALID") from exc
        if not isinstance(payload, dict):
            raise PayPalOAuthError("PAYPAL_OAUTH_RESPONSE_INVALID")

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise PayPalOAuthError("PAYPAL_OAUTH_TOKEN_MISSING")

        token_type = payload.get("token_type")
        if not isinstance(token_type, str) or token_type.lower() != "bearer":
            raise PayPalOAuthError("PAYPAL_OAUTH_TOKEN_TYPE_INVALID")

        expires_in = payload.get("expires_in")
        if isinstance(expires_in, bool) or not isinstance(expires_in, int) or expires_in <= 0:
            raise PayPalOAuthError("PAYPAL_OAUTH_EXPIRY_INVALID")

        scope = payload.get("scope")
        # Exact token membership, never a substring test. A malformed or absent scope
        # must not be reported as verified, and an app that cannot pay must not reach
        # a payout POST hoping the account is entitled.
        if not isinstance(scope, str) or not scope.strip():
            raise PayPalOAuthError("PAYPAL_OAUTH_SCOPE_UNVERIFIED")
        scope_tokens = frozenset(scope.split())
        if PAYOUTS_SCOPE not in scope_tokens:
            raise PayPalOAuthError("PAYPAL_OAUTH_PAYOUTS_SCOPE_MISSING")

        return OAuthToken(
            access_token=access_token,
            token_type=token_type,
            expires_at=moment + timedelta(seconds=expires_in),
            scope_tokens=scope_tokens,
            scope_verified=True,
        )

    def redacted_identity(self) -> str:
        """A non-secret identifier for logs: the client id's last four characters."""
        tail = self.client_id[-4:] if len(self.client_id) >= 4 else ""
        return f"paypal-client:…{tail}"

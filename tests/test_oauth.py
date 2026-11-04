"""OAuth token acquisition: exact request shape, caching, and secret containment.

The credential material in this file is synthetic. Every test that can fail asserts
that the failing message does not contain it.
"""

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs
import json

import pytest

from missionpay.oauth import (
    PayPalClientCredentials,
    PayPalOAuthError,
)

CLIENT_ID = "synthetic-sandbox-client-id"
CLIENT_SECRET = "synthetic-sandbox-secret-9d2f"
NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)


class RecordingTokenTransport:
    def __init__(self, status: int = 200, payload: dict | None = None, body: bytes | None = None):
        self.status = status
        self.payload = (
            payload
            if payload is not None
            else {
                "access_token": "A21AA-synthetic-bearer",
                "token_type": "Bearer",
                "expires_in": 32400,
                "scope": "https://uri.paypal.com/payments/payouts openid",
                "app_id": "APP-80W284485P519543T",
            }
        )
        self.raw = body
        self.calls: list[dict[str, object]] = []

    def request(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if self.raw is not None:
            return self.status, self.raw
        return self.status, json.dumps(self.payload).encode("utf-8")


def _credentials(transport, **overrides) -> PayPalClientCredentials:
    return PayPalClientCredentials(
        client_id=overrides.get("client_id", CLIENT_ID),
        client_secret=overrides.get("client_secret", CLIENT_SECRET),
        base_url=overrides.get("base_url", "https://api-m.sandbox.paypal.com"),
        transport=transport,
    )


def test_token_request_has_the_documented_shape():
    transport = RecordingTokenTransport()
    token = _credentials(transport).access_token(now=NOW)

    assert token == "A21AA-synthetic-bearer"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api-m.sandbox.paypal.com/v1/oauth2/token"
    assert call["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert call["headers"]["Accept"] == "application/json"
    # The secret appears only inside the Basic header, never in the body.
    assert call["body"] == b"grant_type=client_credentials"
    assert CLIENT_SECRET.encode() not in call["body"]
    assert call["headers"]["Authorization"].startswith("Basic ")


def test_basic_header_decodes_to_client_id_and_secret():
    from base64 import b64decode

    transport = RecordingTokenTransport()
    _credentials(transport).access_token(now=NOW)

    encoded = transport.calls[0]["headers"]["Authorization"].removeprefix("Basic ")
    assert b64decode(encoded).decode() == f"{CLIENT_ID}:{CLIENT_SECRET}"


def test_token_is_cached_until_shortly_before_expiry():
    transport = RecordingTokenTransport()
    credentials = _credentials(transport)

    credentials.access_token(now=NOW)
    credentials.access_token(now=NOW + timedelta(hours=1))
    assert len(transport.calls) == 1, "a second token was minted inside the validity window"

    # 32400s lifetime; 60s skew means it must be refreshed just before the end.
    credentials.access_token(now=NOW + timedelta(seconds=32400 - 30))
    assert len(transport.calls) == 2, "an expiring token was reused"


@pytest.mark.parametrize("status", [400, 401, 403, 500])
def test_http_failure_is_typed_and_leaks_nothing(status):
    transport = RecordingTokenTransport(
        status=status,
        payload={"error": "invalid_client", "error_description": "Client Authentication failed"},
    )
    with pytest.raises(PayPalOAuthError) as excinfo:
        _credentials(transport).access_token(now=NOW)
    message = str(excinfo.value)
    assert message == f"PAYPAL_OAUTH_HTTP_{status}"
    assert CLIENT_SECRET not in message
    assert "invalid_client" not in message


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"token_type": "Bearer", "expires_in": 100}, "PAYPAL_OAUTH_TOKEN_MISSING"),
        ({"access_token": "", "token_type": "Bearer", "expires_in": 100}, "PAYPAL_OAUTH_TOKEN_MISSING"),
        (
            {"access_token": "t", "token_type": "MAC", "expires_in": 100},
            "PAYPAL_OAUTH_TOKEN_TYPE_INVALID",
        ),
        (
            {"access_token": "t", "token_type": "Bearer", "expires_in": "soon"},
            "PAYPAL_OAUTH_EXPIRY_INVALID",
        ),
        (
            {"access_token": "t", "token_type": "Bearer", "expires_in": 0},
            "PAYPAL_OAUTH_EXPIRY_INVALID",
        ),
        (
            {"access_token": "t", "token_type": "Bearer", "expires_in": 100, "scope": "openid"},
            "PAYPAL_OAUTH_PAYOUTS_SCOPE_MISSING",
        ),
    ],
)
def test_malformed_provider_responses_are_refused(payload, expected):
    transport = RecordingTokenTransport(payload=payload)
    with pytest.raises(PayPalOAuthError, match=expected):
        _credentials(transport).access_token(now=NOW)


def test_non_json_body_is_refused():
    transport = RecordingTokenTransport(body=b"<html>gateway</html>")
    with pytest.raises(PayPalOAuthError, match="PAYPAL_OAUTH_RESPONSE_INVALID"):
        _credentials(transport).access_token(now=NOW)


def test_non_sandbox_origin_is_refused_before_any_request():
    transport = RecordingTokenTransport()
    with pytest.raises(PayPalOAuthError, match="PAYPAL_SANDBOX_ORIGIN_REQUIRED"):
        _credentials(transport, base_url="https://api-m.paypal.com").access_token(now=NOW)
    assert transport.calls == []


def test_empty_credentials_are_refused():
    with pytest.raises(PayPalOAuthError, match="PAYPAL_CREDENTIALS_REQUIRED"):
        _credentials(RecordingTokenTransport(), client_secret="").access_token(now=NOW)


def test_redacted_identity_exposes_only_a_tail():
    identity = _credentials(RecordingTokenTransport()).redacted_identity()
    assert identity == "paypal-client:…t-id"
    assert CLIENT_SECRET not in identity


def test_repr_of_credentials_omits_both_secrets():
    rendered = repr(_credentials(RecordingTokenTransport()))
    assert CLIENT_SECRET not in rendered
    assert CLIENT_ID not in rendered

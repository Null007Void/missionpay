"""Narrow PayPal Payouts adapter restricted to the official sandbox origin."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .domain import PaymentIntent

_SANDBOX_ORIGIN = "https://api-m.sandbox.paypal.com"
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class SandboxEndpointRequired(ValueError):
    pass


class PayPalContractError(RuntimeError):
    pass


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, bytes]: ...


class UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, bytes]:
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            raise PayPalContractError("PAYPAL_NETWORK_OUTCOME_UNCERTAIN") from exc


@dataclass(frozen=True, slots=True)
class PayPalSandboxAdapter:
    base_url: str = _SANDBOX_ORIGIN
    transport: HttpTransport | None = None

    def __post_init__(self) -> None:
        if self.base_url != _SANDBOX_ORIGIN:
            raise SandboxEndpointRequired("PAYPAL_SANDBOX_ORIGIN_REQUIRED")
        if self.transport is None:
            object.__setattr__(self, "transport", UrllibTransport())

    @staticmethod
    def _identities(intent: PaymentIntent) -> tuple[str, str]:
        digest = intent.operation_key.removeprefix("missionpay:")
        if len(digest) != 64:
            raise PayPalContractError("OPERATION_KEY_INVALID")
        return f"mp-{digest[:35]}", f"mp-{digest[:27]}"

    @staticmethod
    def _money_value(cents: int) -> str:
        return f"{cents // 100}.{cents % 100:02d}"

    def create_payout(self, intent: PaymentIntent, *, access_token: str) -> dict[str, str]:
        request_id, sender_batch_id = self._identities(intent)
        payload = {
            "sender_batch_header": {
                "sender_batch_id": sender_batch_id,
                "recipient_type": "EMAIL",
                "email_subject": "Your approved MissionPay reimbursement",
            },
            "items": [
                {
                    "amount": {
                        "currency": intent.amount.currency,
                        "value": self._money_value(intent.amount.cents),
                    },
                    "note": "Approved MissionPay nonprofit reimbursement",
                    "receiver": intent.recipient,
                    "sender_item_id": f"{intent.claim_id}-r{intent.claim_revision}",
                }
            ],
        }
        result = self._request_json(
            "POST",
            "/v1/payments/payouts",
            access_token,
            payload,
            request_id=request_id,
        )
        try:
            header = result["batch_header"]
            return {
                "payout_batch_id": header["payout_batch_id"],
                "batch_status": header["batch_status"],
            }
        except (KeyError, TypeError) as exc:
            raise PayPalContractError("PAYPAL_CREATE_RESPONSE_INVALID") from exc

    def get_batch(self, payout_batch_id: str, *, access_token: str) -> dict[str, object]:
        self._validate_provider_id(payout_batch_id)
        return self._request_json(
            "GET",
            f"/v1/payments/payouts/{payout_batch_id}?page_size=1000&total_required=true",
            access_token,
            None,
        )

    def get_item(self, payout_item_id: str, *, access_token: str) -> dict[str, object]:
        self._validate_provider_id(payout_item_id)
        return self._request_json(
            "GET",
            f"/v1/payments/payouts-item/{payout_item_id}",
            access_token,
            None,
        )

    @staticmethod
    def _validate_provider_id(value: str) -> None:
        if not _PROVIDER_ID.fullmatch(value):
            raise PayPalContractError("PAYPAL_PROVIDER_ID_INVALID")

    def _request_json(
        self,
        method: str,
        path: str,
        access_token: str,
        payload: dict[str, object] | None,
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        if not access_token:
            raise PayPalContractError("PAYPAL_ACCESS_TOKEN_REQUIRED")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if request_id is not None:
            headers["PayPal-Request-Id"] = request_id
        body = None if payload is None else json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        assert self.transport is not None
        status, raw = self.transport.request(method, self.base_url + path, headers, body)
        if status < 200 or status >= 300:
            raise PayPalContractError(f"PAYPAL_HTTP_{status}")
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise PayPalContractError("PAYPAL_RESPONSE_JSON_INVALID") from exc
        if not isinstance(result, dict):
            raise PayPalContractError("PAYPAL_RESPONSE_SHAPE_INVALID")
        return result

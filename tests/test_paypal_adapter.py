import json

import pytest

from missionpay.domain import Money, PaymentIntent
from missionpay.paypal import PayPalSandboxAdapter, SandboxEndpointRequired


class RecordingTransport:
    def __init__(self):
        self.calls = []
        self.responses = []

    def request(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        return self.responses.pop(0)


def _intent() -> PaymentIntent:
    return PaymentIntent(
        operation_key="missionpay:" + "1" * 64,
        claim_id="claim-001",
        claim_revision=1,
        evidence_sha256="2" * 64,
        policy_version="2026-03",
        fund_id="river-cleanup-fund",
        recipient="volunteer@example.org",
        amount=Money("USD", 7899),
        authorization_id="auth-001",
    )


def test_adapter_allows_only_exact_sandbox_origin_and_builds_one_bound_item():
    transport = RecordingTransport()
    transport.responses.append(
        (201, json.dumps({"batch_header": {"payout_batch_id": "BATCH123", "batch_status": "PENDING"}}).encode())
    )
    adapter = PayPalSandboxAdapter(transport=transport)

    result = adapter.create_payout(_intent(), access_token="sandbox-token")

    method, url, headers, raw = transport.calls[0]
    body = json.loads(raw)
    assert method == "POST"
    assert url == "https://api-m.sandbox.paypal.com/v1/payments/payouts"
    assert headers["Authorization"] == "Bearer sandbox-token"
    assert headers["PayPal-Request-Id"] == "mp-" + "1" * 35
    assert body["sender_batch_header"]["sender_batch_id"] == "mp-" + "1" * 27
    assert body["sender_batch_header"]["recipient_type"] == "EMAIL"
    assert body["items"] == [
        {
            "amount": {"currency": "USD", "value": "78.99"},
            "note": "Approved MissionPay nonprofit reimbursement",
            "receiver": "volunteer@example.org",
            "sender_item_id": "claim-001-r1",
        }
    ]
    assert result == {"payout_batch_id": "BATCH123", "batch_status": "PENDING"}

    with pytest.raises(SandboxEndpointRequired):
        PayPalSandboxAdapter(base_url="https://api-m.paypal.com", transport=transport)


def test_batch_and_item_retrieval_are_exact_read_only_contracts():
    transport = RecordingTransport()
    transport.responses.extend(
        [
            (200, b'{"batch_header":{"payout_batch_id":"BATCH123"},"items":[]}'),
            (200, b'{"payout_item_id":"ITEM123","transaction_status":"SUCCESS"}'),
        ]
    )
    adapter = PayPalSandboxAdapter(transport=transport)

    batch = adapter.get_batch("BATCH123", access_token="sandbox-token")
    item = adapter.get_item("ITEM123", access_token="sandbox-token")

    assert batch["batch_header"]["payout_batch_id"] == "BATCH123"
    assert item["payout_item_id"] == "ITEM123"
    assert [call[:2] for call in transport.calls] == [
        (
            "GET",
            "https://api-m.sandbox.paypal.com/v1/payments/payouts/BATCH123?page_size=1000&total_required=true",
        ),
        ("GET", "https://api-m.sandbox.paypal.com/v1/payments/payouts-item/ITEM123"),
    ]

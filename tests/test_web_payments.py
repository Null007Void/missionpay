"""Route-level tests for the payment surface.

These drive the real FastAPI app over `TestClient`: real CSRF, real session cookies,
real SQLite. Only the PayPal transport and the token provider are doubled, because
a genuine sandbox call is owner-gated.

Mock mode is tested separately and must refuse every provider operation.
"""

from datetime import date
import json
import re

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from missionpay.auth import Role
from missionpay.config import ProviderMode, RuntimeConfig
from missionpay.domain import FundPolicy, Money
from missionpay.executor import PayoutExecutor
from missionpay.service import PaymentService
from missionpay.store import MissionPayStore
from missionpay.web import create_app

from conftest import approval_fields

SANDBOX_RECIPIENT = "sb-volunteer@business.example.com"
CAP_CENTS = 10_000


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _store(tmp_path) -> MissionPayStore:
    store = MissionPayStore(tmp_path / "pay.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    store.register_user("volunteer-1", "river.volunteer@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    store.register_user("reviewer-1", "reviewer@example.test", Role.REVIEWER, "reviewer-passphrase")
    for fund_id, activity, category in (
        ("GARDEN_FUND", "COMMUNITY_GARDEN", "SEEDS"),
        ("RIVER_CLEANUP_FUND", "RIVER_CLEANUP", "PROTECTIVE_SUPPLIES"),
    ):
        store.create_fund(
            FundPolicy(
                fund_id=fund_id,
                version="2026.1",
                allowed_activity_codes=frozenset({activity}),
                allowed_expense_categories=frozenset({category}),
                starts_on=date(2026, 1, 1),
                ends_on=date(2026, 12, 31),
                available=Money("USD", 50_000),
                receipt_required=True,
            )
        )
    return store


class FakePayPal:
    """Serves payout POST and readback GETs; records every call."""

    def __init__(self, *, post_raises: Exception | None = None, item_status: str = "SUCCESS"):
        self.post_raises = post_raises
        self.item_status = item_status
        self.posts: list[dict[str, object]] = []
        self.gets: list[str] = []

    def create_payout(self, intent, *, access_token):
        self.posts.append({"intent": intent, "access_token": access_token})
        if self.post_raises is not None:
            raise self.post_raises
        return {"payout_batch_id": "BATCH-SB-1", "batch_status": "PENDING"}

    def get_batch(self, payout_batch_id, *, access_token):
        self.gets.append(f"batch:{payout_batch_id}")
        intent = self.posts[0]["intent"]
        digest = intent.operation_key.removeprefix("missionpay:")
        return {
            "batch_header": {
                "payout_batch_id": payout_batch_id,
                "sender_batch_header": {"sender_batch_id": f"mp-{digest[:27]}"},
            },
            "items": [
                {
                    "payout_item_id": "ITEM-SB-1",
                    "transaction_status": self.item_status,
                    "payout_item": {
                        "sender_item_id": f"{intent.claim_id}-r{intent.claim_revision}",
                        "receiver": intent.recipient,
                        "amount": {
                            "currency": intent.amount.currency,
                            "value": f"{intent.amount.cents // 100}.{intent.amount.cents % 100:02d}",
                        },
                    },
                }
            ],
        }

    def get_item(self, payout_item_id, *, access_token):
        self.gets.append(f"item:{payout_item_id}")
        batch = self.get_batch("BATCH-SB-1", access_token=access_token)
        return batch["items"][0]


def _genuine_app(store, adapter: FakePayPal):
    config = RuntimeConfig(
        provider_mode=ProviderMode.PAYPAL_SANDBOX,
        paypal_client_id="sandbox-client-id",
        paypal_client_secret="sandbox-client-secret",
        sandbox_recipient=SANDBOX_RECIPIENT,
        max_payout_cents=CAP_CENTS,
        secure_cookies=False,
    )
    service = PaymentService(
        store,
        config,
        adapter=adapter,
        executor=PayoutExecutor(store, adapter, max_payout_cents=CAP_CENTS),
        token_provider=lambda: "bearer-synthetic",
    )
    return create_app(store, config=config, service=service)


def _login(client, email, password):
    page = client.get("/login")
    response = client.post(
        "/login",
        data={"email": email, "password": password, "csrf_token": _csrf(page.text)},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _submit_and_approve(client) -> None:
    _login(client, "river.volunteer@example.test", "volunteer-passphrase")
    submitted = client.post(
        "/claims/demo", data={"csrf_token": _csrf(client.get("/app").text)}, follow_redirects=False
    )
    assert submitted.status_code == 303
    interpreted = client.post(
        "/claims/river-safety-001/interpret",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    assert interpreted.status_code == 303
    client.post("/logout", data={"csrf_token": _csrf(client.get("/app").text)})

    _login(client, "reviewer@example.test", "reviewer-passphrase")
    page = client.get("/claims/river-safety-001").text
    approved = client.post(
        "/claims/river-safety-001/approve",
        data={
            "csrf_token": _csrf(client.get("/app").text),
            "revision": "1",
            "amount_cents": "7899",
            **approval_fields(page),
        },
        follow_redirects=False,
    )
    assert approved.status_code == 303


# ------------------------------------------------------------- mock mode


def test_mock_mode_refuses_execution_and_contacts_nothing(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal()
    client = TestClient(create_app(store), base_url="https://testserver")
    _submit_and_approve(client)

    detail = client.get("/claims/river-safety-001")
    assert "MODE: MOCK — NO PROVIDER CALLS, NO MONEY MOVES" in detail.text
    assert "SIMULATED — NO PAYPAL PAYMENT EXECUTED" in detail.text
    assert "Mock mode will not contact PayPal" in detail.text

    refused = client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
    )
    assert refused.status_code == 409
    assert "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS" in refused.text
    assert adapter.posts == []


def test_mock_mode_refuses_reconciliation(tmp_path):
    store = _store(tmp_path)
    client = TestClient(create_app(store), base_url="https://testserver")
    _submit_and_approve(client)

    refused = client.post(
        "/claims/river-safety-001/reconcile",
        data={"csrf_token": _csrf(client.get("/app").text)},
    )
    assert refused.status_code == 409
    assert "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS" in refused.text


# ---------------------------------------------------------- genuine mode


def test_genuine_execution_sends_exactly_one_payout(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal()
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)

    detail = client.get("/claims/river-safety-001")
    assert "MODE: PAYPAL SANDBOX — GENUINE PROVIDER CALLS" in detail.text
    assert "SANDBOX FUNDS, NOT REAL MONEY MOVED" in detail.text
    assert "Execute one sandbox payout" in detail.text

    paid = client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    assert paid.status_code == 303
    assert len(adapter.posts) == 1

    intent = adapter.posts[0]["intent"]
    # The payee is the nominated sandbox account, not the volunteer's real address.
    assert intent.recipient == SANDBOX_RECIPIENT

    after = client.get("/claims/river-safety-001")
    assert "Acknowledged" in after.text
    assert "PAYOUT_ACKNOWLEDGED" in after.text


def test_duplicate_click_cannot_trigger_a_second_payout(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal()
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)

    first = client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    second = client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
    )

    assert first.status_code == 303
    assert second.status_code == 409
    assert "DISPATCH_STATE_NOT_PREPARED" in second.text
    assert len(adapter.posts) == 1


def test_missing_csrf_is_refused_before_any_dispatch(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal()
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)

    refused = client.post("/claims/river-safety-001/pay", data={"csrf_token": "forged"})
    assert refused.status_code == 403
    assert adapter.posts == []


def test_volunteer_cannot_execute(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal()
    app = _genuine_app(store, adapter)
    _submit_and_approve(
        TestClient(app, base_url="http://testserver")
    )

    volunteer = TestClient(app, base_url="http://testserver")
    _login(volunteer, "river.volunteer@example.test", "volunteer-passphrase")
    refused = volunteer.post(
        "/claims/river-safety-001/pay", data={"csrf_token": _csrf(volunteer.get("/app").text)}
    )
    assert refused.status_code == 403
    assert "REVIEWER_ROLE_REQUIRED" in refused.text
    assert adapter.posts == []


def test_reconciliation_reads_the_item_and_reports_completion(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal(item_status="SUCCESS")
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)

    client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    reconciled = client.post(
        "/claims/river-safety-001/reconcile",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    assert reconciled.status_code == 303

    after = client.get("/claims/river-safety-001")
    assert "Independently reconciled" in after.text
    assert "matched_completed" in after.text
    assert any(get.startswith("batch:") for get in adapter.gets)


def test_pending_provider_status_is_not_reported_as_paid(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal(item_status="PENDING")
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)

    client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    client.post(
        "/claims/river-safety-001/reconcile",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )

    after = client.get("/claims/river-safety-001")
    assert "matched_completed" not in after.text
    assert "PAYPAL_ITEM_PENDING" in after.text


# -------------------------------------------------- uncertain + recovery


def test_unknown_outcome_stops_and_offers_recovery(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal(post_raises=TimeoutError("synthetic timeout after send"))
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)

    first = client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )
    assert first.status_code == 303
    assert len(adapter.posts) == 1

    detail = client.get("/claims/river-safety-001")
    assert "Outcome unresolved" in detail.text
    assert "will not send another one" in detail.text
    assert "Bind an authoritative PayPal batch id" in detail.text

    second = client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
    )
    assert second.status_code == 409
    assert "UNCERTAIN_REQUIRES_RECONCILIATION" in second.text
    assert len(adapter.posts) == 1, "an uncertain operation was re-dispatched"


def test_attaching_a_batch_id_resolves_the_uncertain_operation(tmp_path):
    store = _store(tmp_path)
    adapter = FakePayPal(post_raises=TimeoutError("synthetic timeout after send"))
    client = TestClient(_genuine_app(store, adapter), base_url="http://testserver")
    _submit_and_approve(client)
    client.post(
        "/claims/river-safety-001/pay",
        data={"csrf_token": _csrf(client.get("/app").text)},
        follow_redirects=False,
    )

    blank = client.post(
        "/claims/river-safety-001/attach-batch",
        data={"csrf_token": _csrf(client.get("/app").text), "payout_batch_id": "  "},
    )
    assert blank.status_code == 400
    assert "PAYOUT_BATCH_ID_REQUIRED" in blank.text

    bound = client.post(
        "/claims/river-safety-001/attach-batch",
        data={"csrf_token": _csrf(client.get("/app").text), "payout_batch_id": "BATCH-SB-1"},
        follow_redirects=False,
    )
    assert bound.status_code == 303

    after = client.get("/claims/river-safety-001")
    assert "matched_completed" in after.text
    assert len(adapter.posts) == 1, "recovery must not send another payout"

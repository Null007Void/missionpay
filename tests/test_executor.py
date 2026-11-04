"""Executor: one intent, at most one payout POST, and an honest ambiguity state.

Fixtures are synthetic. No socket is opened: the adapter is either a recording
double or the real :class:`PayPalSandboxAdapter` driven by a recording transport.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import sqlite3

import pytest
from argon2 import PasswordHasher

from missionpay.auth import Role
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.executor import DispatchRefused, PayoutExecutor
from missionpay.paypal import PayPalSandboxAdapter
from missionpay.policy import evaluate_claim
from missionpay.store import MissionPayStore
from conftest import accept_interpretation, authorize

FUND_ID = "river-cleanup-fund"
POLICY_VERSION = "2026-03"


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def store(tmp_path):
    result = MissionPayStore(
        tmp_path / "missionpay.db",
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    result.initialize()
    result.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    result.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    result.create_fund(
        FundPolicy(
            fund_id=FUND_ID,
            version=POLICY_VERSION,
            allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
            allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        )
    )
    return result


def _intent(store, claim_id: str = "claim-river-001", amount: int = 7_899):
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
    claim = Claim(
        claim_id=claim_id,
        volunteer_id="volunteer-alex",
        activity=Activity("activity-river-001", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("a" * 64, "image/jpeg", "receipt.jpg"),
        requested=Money("USD", amount),
        recipient="sb-volunteer@business.example.com",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(
            claim, store.load_fund_policy(FUND_ID, POLICY_VERSION), prior_evidence_digests=frozenset()
        ),
        revision=1,
    )
    accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    authorization = authorize(store, reviewer.token, claim.claim_id, revision=1,
              record=accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256))
    return store.create_payment_intent(authorization.authorization_id)


class RecordingPayoutAdapter:
    """Returns a fixed batch header and records each invocation."""

    def __init__(self, *, raises: Exception | None = None, response: dict | None = None):
        self.raises = raises
        self.response = response if response is not None else {"payout_batch_id": "BATCH-SANDBOX-1"}
        self.calls: list[dict[str, object]] = []

    def create_payout(self, intent, *, access_token):
        self.calls.append({"intent": intent, "access_token": access_token})
        if self.raises is not None:
            raise self.raises
        return self.response


def _events(store: MissionPayStore, operation_key: str) -> list[str]:
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute(
            "SELECT event_type FROM audit_events WHERE aggregate_id=? ORDER BY sequence",
            (operation_key,),
        ).fetchall()
    return [row[0] for row in rows]


# ------------------------------------------------------------- happy path


def test_successful_dispatch_is_acknowledged_once(store):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)

    outcome = executor.execute(
        intent.operation_key, token_provider=lambda: "bearer-synthetic", owner_id="worker-A"
    )

    assert outcome.state == "ACKNOWLEDGED"
    assert outcome.dispatched is True
    assert outcome.payout_batch_id == "BATCH-SANDBOX-1"
    assert store.operation_state(intent.operation_key) == "ACKNOWLEDGED"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["access_token"] == "bearer-synthetic"

    events = _events(store, intent.operation_key)
    assert events.index("PAYOUT_DISPATCH_STARTED") < events.index("PAYOUT_ACKNOWLEDGED")


def test_the_real_adapter_builds_the_exact_authorized_payload(store):
    intent = _intent(store)
    recorded: list[dict[str, object]] = []

    class Transport:
        def request(self, method, url, headers, body):
            recorded.append({"method": method, "url": url, "headers": headers, "body": body})
            return 201, json.dumps(
                {"batch_header": {"payout_batch_id": "BATCH-REAL-9", "batch_status": "PENDING"}}
            ).encode()

    executor = PayoutExecutor(
        store, PayPalSandboxAdapter(transport=Transport()), max_payout_cents=10_000
    )
    outcome = executor.execute(
        intent.operation_key, token_provider=lambda: "bearer-synthetic", owner_id="worker-A"
    )

    assert outcome.state == "ACKNOWLEDGED"
    assert len(recorded) == 1
    call = recorded[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api-m.sandbox.paypal.com/v1/payments/payouts"
    digest = intent.operation_key.removeprefix("missionpay:")
    assert call["headers"]["PayPal-Request-Id"] == f"mp-{digest[:35]}"

    payload = json.loads(call["body"])
    assert payload["sender_batch_header"]["sender_batch_id"] == f"mp-{digest[:27]}"
    item = payload["items"][0]
    assert item["receiver"] == intent.recipient
    assert item["amount"] == {"currency": "USD", "value": "78.99"}
    assert item["sender_item_id"] == f"{intent.claim_id}-r{intent.claim_revision}"


# --------------------------------------------------- pre-dispatch refusals


def test_token_failure_leaves_the_operation_prepared_and_sends_nothing(store):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)

    def broken_token():
        raise RuntimeError("synthetic credential failure")

    with pytest.raises(DispatchRefused, match="PAYPAL_TOKEN_UNAVAILABLE"):
        executor.execute(intent.operation_key, token_provider=broken_token, owner_id="worker-A")

    assert store.operation_state(intent.operation_key) == "PREPARED"
    assert adapter.calls == []
    assert "PAYOUT_DISPATCH_STARTED" not in _events(store, intent.operation_key)

    # A later, healthy attempt is legitimate precisely because nothing was sent.
    outcome = executor.execute(
        intent.operation_key, token_provider=lambda: "bearer", owner_id="worker-A"
    )
    assert outcome.state == "ACKNOWLEDGED"


def test_empty_token_is_a_pre_dispatch_refusal(store):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter)

    with pytest.raises(DispatchRefused, match="EMPTY"):
        executor.execute(intent.operation_key, token_provider=lambda: "", owner_id="worker-A")

    assert store.operation_state(intent.operation_key) == "PREPARED"
    assert adapter.calls == []


def test_cap_is_enforced_before_dispatch(store):
    intent = _intent(store, amount=7_899)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=5_000)

    with pytest.raises(DispatchRefused, match="PAYOUT_CAP_EXCEEDED"):
        executor.execute(intent.operation_key, token_provider=lambda: "b", owner_id="worker-A")

    assert store.operation_state(intent.operation_key) == "PREPARED"
    assert adapter.calls == []


# ------------------------------------------------ post-dispatch ambiguity


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("socket timeout"),
        ConnectionError("reset by peer"),
    ],
)
def test_post_dispatch_failure_is_uncertain_and_never_retried(store, error):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter(raises=error)
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)

    outcome = executor.execute(
        intent.operation_key, token_provider=lambda: "bearer", owner_id="worker-A"
    )

    assert outcome.state == "UNCERTAIN"
    assert outcome.dispatched is True
    assert outcome.payout_batch_id is None
    assert store.operation_state(intent.operation_key) == "UNCERTAIN"

    # A second attempt must be refused by the executor, not merely by the adapter.
    with pytest.raises(DispatchRefused, match="UNCERTAIN_REQUIRES_RECONCILIATION"):
        executor.execute(intent.operation_key, token_provider=lambda: "bearer", owner_id="worker-A")

    assert len(adapter.calls) == 1, "a second payout POST was attempted"
    assert "PAYOUT_DISPATCH_UNCERTAIN" in _events(store, intent.operation_key)


def test_response_without_a_batch_id_is_uncertain(store):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter(response={"batch_status": "PENDING"})
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)

    outcome = executor.execute(
        intent.operation_key, token_provider=lambda: "bearer", owner_id="worker-A"
    )

    assert outcome.state == "UNCERTAIN"
    assert outcome.detail == "PROVIDER_RESPONSE_WITHOUT_BATCH_ID"
    assert store.operation_state(intent.operation_key) == "UNCERTAIN"


def test_acknowledged_operation_cannot_be_dispatched_again(store):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)

    executor.execute(intent.operation_key, token_provider=lambda: "b", owner_id="worker-A")
    with pytest.raises(DispatchRefused, match="DISPATCH_STATE_NOT_PREPARED"):
        executor.execute(intent.operation_key, token_provider=lambda: "b", owner_id="worker-B")

    assert len(adapter.calls) == 1


# --------------------------------------------------------- restart safety


def test_crash_after_dispatch_claim_becomes_uncertain_and_blocks_a_second_post(store):
    intent = _intent(store)
    # Simulate the process dying after SENDING was committed but before the POST.
    store.begin_dispatch(intent.operation_key, "worker-A")

    recovered = store.recover_interrupted_dispatches()
    assert recovered == 1
    assert store.operation_state(intent.operation_key) == "UNCERTAIN"

    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)
    with pytest.raises(DispatchRefused, match="UNCERTAIN_REQUIRES_RECONCILIATION"):
        executor.execute(intent.operation_key, token_provider=lambda: "b", owner_id="worker-B")
    assert adapter.calls == []


def test_concurrent_executors_permit_exactly_one_payout(store):
    intent = _intent(store)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000)

    def attempt(worker: str):
        try:
            return executor.execute(
                intent.operation_key, token_provider=lambda: "b", owner_id=worker
            ).state
        except DispatchRefused as exc:
            return f"refused:{exc}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, ["w1", "w2", "w3", "w4"]))

    assert results.count("ACKNOWLEDGED") == 1
    assert len(adapter.calls) == 1, "concurrency produced more than one payout POST"
    assert sum(1 for r in results if r.startswith("refused:")) == 3

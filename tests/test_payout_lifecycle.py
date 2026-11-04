"""Payout observation lifecycle: append-only history, ordered derivation, zero balance.

Covers MP-1 (later observations of the same provider item), MP-2 (a completed
payout superseded by REFUNDED/REVERSED/RETURNED) and MP-3 (an exactly exhausted
fund). Every provider-facing fixture counts request constructions: no path may
build a payout POST.
"""

from datetime import date, datetime, timedelta, timezone
import json
import sqlite3

import pytest
from argon2 import PasswordHasher

from missionpay.auth import Role
from missionpay.domain import (
    Activity,
    Claim,
    EvidenceIdentity,
    FundBalance,
    FundPolicy,
    Money,
    PolicyOutcome,
    ProviderItemStatus,
    ProviderObservation,
)
from missionpay.paypal import PayPalSandboxAdapter
from missionpay.policy import evaluate_claim
from missionpay.reconciliation import reconcile_payment
from missionpay.recovery import PayPalRecovery, RecoveryDisposition
from missionpay.store import InsufficientFunds, MissionPayStore, PersistenceConflict
from conftest import accept_interpretation, authorize

FUND_ID = "river-cleanup-fund"
POLICY_VERSION = "2026-03"
BATCH_ID = "BATCH123"
ITEM_ID = "ITEM123"
T0 = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _fast_hasher() -> PasswordHasher:
    return PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)


def _policy(fund_cents: int = 50_000) -> FundPolicy:
    return FundPolicy(
        fund_id=FUND_ID,
        version=POLICY_VERSION,
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money("USD", fund_cents),
        receipt_required=True,
    )


def _store(tmp_path, *, fund_cents: int = 50_000) -> MissionPayStore:
    store = MissionPayStore(tmp_path / "missionpay.db", password_hasher=_fast_hasher())
    store.initialize()
    store.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    store.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    store.create_fund(_policy(fund_cents))
    return store


def _claim(claim_id: str, amount: int, digest: str, recipient: str | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        volunteer_id="volunteer-alex",
        activity=Activity(f"activity-{claim_id}", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity(digest, "image/jpeg", f"receipt-{claim_id}.jpg"),
        requested=Money("USD", amount),
        recipient=recipient or "volunteer@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )


def _submit_and_authorize(store: MissionPayStore, claim: Claim):
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
    store.submit_claim(volunteer.token, claim, revision=1)
    policy = store.load_fund_policy(FUND_ID, POLICY_VERSION)
    store.record_policy_decision(
        evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
    )
    accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    return authorize(store, reviewer.token, claim.claim_id, revision=1,
              record=accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256))


def _acknowledged_operation(tmp_path, *, fund_cents: int = 50_000):
    store = _store(tmp_path, fund_cents=fund_cents)
    authorization = _submit_and_authorize(store, _claim("claim-river-001", 7_899, "a" * 64))
    intent = store.create_payment_intent(authorization.authorization_id)
    store.begin_dispatch(intent.operation_key, "worker-A")
    store.record_payout_acknowledgement(intent.operation_key, BATCH_ID)
    return store, intent


def _observation(
    intent,
    status: ProviderItemStatus,
    *,
    observed_at: datetime = T0,
    payout_item_id: str = ITEM_ID,
    recipient: str | None = None,
    amount: Money | None = None,
    raw_sha256: str = "1" * 64,
    observation_id: str | None = None,
    payout_batch_id: str = BATCH_ID,
) -> ProviderObservation:
    return ProviderObservation(
        observation_id=observation_id or f"observation-{status.value.lower()}-{raw_sha256[:6]}",
        attempt_id=intent.operation_key,
        provider="PAYPAL_SANDBOX",
        payout_batch_id=payout_batch_id,
        payout_item_id=payout_item_id,
        status=status,
        recipient=recipient if recipient is not None else intent.recipient,
        amount=amount if amount is not None else intent.amount,
        observed_at=observed_at,
        raw_sha256=raw_sha256,
    )


def _persist(store: MissionPayStore, intent, observation: ProviderObservation) -> bool:
    return store.persist_reconciliation(
        intent.operation_key, observation, reconcile_payment(intent, observation)
    )


def _sender_batch_id(intent) -> str:
    return f"mp-{intent.operation_key.removeprefix('missionpay:')[:27]}"


class SyntheticPayPalTransport:
    """Deterministic synthetic PayPal sandbox transport.

    Records every request it is asked to build. A POST is an immediate failure:
    observation paths are read-only by construction.
    """

    def __init__(
        self,
        intent,
        *,
        status: str = "SUCCESS",
        payout_item_id: str = ITEM_ID,
        receiver: str | None = None,
        amount_value: str = "78.99",
        sender_item_id: str | None = None,
        payout_batch_id: str = BATCH_ID,
        sender_batch_id: str | None = None,
    ):
        self.intent = intent
        self.status = status
        self.payout_item_id = payout_item_id
        self.receiver = receiver or intent.recipient
        self.amount_value = amount_value
        self.sender_item_id = sender_item_id or f"{intent.claim_id}-r{intent.claim_revision}"
        self.payout_batch_id = payout_batch_id
        self.sender_batch_id = sender_batch_id or _sender_batch_id(intent)
        self.requests: list[dict[str, object]] = []

    def _item_document(self) -> dict[str, object]:
        return {
            "payout_item_id": self.payout_item_id,
            "transaction_status": self.status,
            "payout_item": {
                "sender_item_id": self.sender_item_id,
                "receiver": self.receiver,
                "amount": {"currency": "USD", "value": self.amount_value},
            },
        }

    def request(self, method, url, headers, body):
        self.requests.append({"method": method, "url": url, "body": body})
        if method == "POST":
            raise AssertionError("a payout POST was constructed on an observation path")
        if "/payouts-item/" in url:
            return 200, json.dumps(self._item_document()).encode("utf-8")
        return 200, json.dumps(
            {
                "batch_header": {
                    "payout_batch_id": self.payout_batch_id,
                    "sender_batch_header": {"sender_batch_id": self.sender_batch_id},
                },
                "items": [self._item_document()],
            }
        ).encode("utf-8")


def _readback(store: MissionPayStore, intent, transport: SyntheticPayPalTransport):
    adapter = PayPalSandboxAdapter(transport=transport)
    return PayPalRecovery(store, adapter).observe_known_payout(
        intent.operation_key, access_token="synthetic-sandbox-token"
    )


def _assert_no_payout_post(transport: SyntheticPayPalTransport) -> None:
    assert transport.requests, "expected at least one provider read"
    assert [request["method"] for request in transport.requests] == ["GET"] * len(
        transport.requests
    )
    for request in transport.requests:
        body = request["body"]
        assert body is None or b"sender_batch_header" not in body


def _audit_events(store: MissionPayStore, operation_key: str) -> list[dict[str, object]]:
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute(
            "SELECT event_type,payload_json FROM audit_events WHERE aggregate_id=? ORDER BY sequence",
            (operation_key,),
        ).fetchall()
    return [{"type": row[0], "payload": json.loads(row[1])} for row in rows]


def _authorization_row(store: MissionPayStore, authorization_id: str) -> tuple:
    with sqlite3.connect(store.path) as conn:
        return conn.execute(
            """SELECT authorization_id,claim_id,recipient,currency,amount_cents,reviewer_id
               FROM authorizations WHERE authorization_id=?""",
            (authorization_id,),
        ).fetchone()


# --------------------------------------------------------------------------- MP-1


def test_pending_observation_then_success_appends_and_completes(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)

    assert _persist(
        store,
        intent,
        _observation(intent, ProviderItemStatus.PENDING, raw_sha256="a" * 64),
    ) is True
    assert store.operation_state(intent.operation_key) == "ACKNOWLEDGED"
    assert store.reconciliation_outcome(intent.operation_key) == "pending"
    assert store.reconciliation_count(intent.operation_key) == 1

    assert _persist(
        store,
        intent,
        _observation(
            intent,
            ProviderItemStatus.SUCCESS,
            observed_at=T0 + timedelta(minutes=1),
            raw_sha256="b" * 64,
        ),
    ) is True
    assert store.operation_state(intent.operation_key) == "VERIFIED"
    assert store.reconciliation_outcome(intent.operation_key) == "matched_completed"
    assert store.reconciliation_count(intent.operation_key) == 2
    history = store.reconciliation_history(intent.operation_key)
    assert [entry["provider_status"] for entry in history] == ["PENDING", "SUCCESS"]


def test_repeated_identical_readback_is_idempotent(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    first = _observation(
        intent, ProviderItemStatus.SUCCESS, observation_id="observation-first", raw_sha256="c" * 64
    )
    assert _persist(store, intent, first) is True

    replay_same_id = _observation(
        intent, ProviderItemStatus.SUCCESS, observation_id="observation-first", raw_sha256="c" * 64
    )
    replay_new_id = _observation(
        intent, ProviderItemStatus.SUCCESS, observation_id="observation-second", raw_sha256="c" * 64
    )

    assert _persist(store, intent, replay_same_id) is False
    assert _persist(store, intent, replay_new_id) is False
    assert store.reconciliation_count(intent.operation_key) == 1
    assert store.latest_financial_outcome(intent.operation_key)["observation_count"] == 1
    assert store.operation_state(intent.operation_key) == "VERIFIED"
    # first write wins: the retained evidence is the original observation
    assert store.reconciliation_history(intent.operation_key)[0]["observation_id"] == "observation-first"


def test_conflicting_recipient_observation_is_persisted_as_conflict(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    _persist(
        store,
        intent,
        _observation(intent, ProviderItemStatus.SUCCESS, raw_sha256="d" * 64),
    )
    assert store.operation_state(intent.operation_key) == "VERIFIED"

    conflicting = _observation(
        intent,
        ProviderItemStatus.SUCCESS,
        observed_at=T0 + timedelta(minutes=2),
        recipient="attacker@example.org",
        raw_sha256="e" * 64,
        observation_id="observation-attacker",
    )
    assert _persist(store, intent, conflicting) is True

    assert store.operation_state(intent.operation_key) == "CONFLICT"
    assert store.reconciliation_outcome(intent.operation_key) == "mismatch"
    history = store.reconciliation_history(intent.operation_key)
    assert len(history) == 2
    assert "RECIPIENT_MISMATCH" in history[-1]["reason_codes"]
    assert history[-1]["raw_sha256"] == "e" * 64
    assert history[-1]["recipient"] == "attacker@example.org"
    # the frozen intent still names the authorized recipient and amount
    assert store.load_intent(intent.operation_key).recipient == intent.recipient


def test_conflicting_payout_item_observation_keeps_original_binding(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    _persist(
        store,
        intent,
        _observation(intent, ProviderItemStatus.SUCCESS, raw_sha256="f" * 64),
    )
    assert store.operation_binding(intent.operation_key)["payout_item_id"] == ITEM_ID

    second_item = _observation(
        intent,
        ProviderItemStatus.SUCCESS,
        observed_at=T0 + timedelta(minutes=3),
        payout_item_id="ITEM-OTHER",
        raw_sha256="9" * 64,
        observation_id="observation-second-item",
    )
    assert _persist(store, intent, second_item) is True

    assert store.operation_state(intent.operation_key) == "CONFLICT"
    assert store.reconciliation_outcome(intent.operation_key) == "mismatch"
    history = store.reconciliation_history(intent.operation_key)
    assert len(history) == 2
    assert "PAYOUT_ITEM_ID_MISMATCH" in history[-1]["reason_codes"]
    assert history[-1]["payout_item_id"] == "ITEM-OTHER"
    assert store.operation_binding(intent.operation_key)["payout_item_id"] == ITEM_ID


@pytest.mark.parametrize("batch_id", ["BATCH-OTHER"])
def test_observation_of_another_batch_is_refused(tmp_path, batch_id):
    store, intent = _acknowledged_operation(tmp_path)
    foreign = _observation(
        intent,
        ProviderItemStatus.SUCCESS,
        payout_batch_id=batch_id,
        raw_sha256="8" * 64,
    )
    with pytest.raises(PersistenceConflict, match="PAYOUT_BATCH_MISMATCH"):
        _persist(store, intent, foreign)
    assert store.reconciliation_count(intent.operation_key) == 0


def test_observation_requires_an_acknowledged_payout(tmp_path):
    store = _store(tmp_path)
    authorization = _submit_and_authorize(store, _claim("claim-river-001", 7_899, "a" * 64))
    intent = store.create_payment_intent(authorization.authorization_id)

    observation = _observation(intent, ProviderItemStatus.SUCCESS, raw_sha256="7" * 64)
    with pytest.raises(PersistenceConflict, match="ACKNOWLEDGED_OPERATION_REQUIRED"):
        _persist(store, intent, observation)

    store.begin_dispatch(intent.operation_key, "worker-A")
    with pytest.raises(PersistenceConflict, match="ACKNOWLEDGED_OPERATION_REQUIRED"):
        _persist(store, intent, observation)
    assert store.reconciliation_count(intent.operation_key) == 0


# --------------------------------------------------------------------------- MP-2


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (ProviderItemStatus.REFUNDED, "PAYPAL_ITEM_REFUNDED"),
        (ProviderItemStatus.REVERSED, "PAYPAL_ITEM_REVERSED"),
        (ProviderItemStatus.RETURNED, "PAYPAL_ITEM_RETURNED"),
    ],
)
def test_completed_payout_is_superseded_by_terminal_provider_state(tmp_path, status, reason):
    store, intent = _acknowledged_operation(tmp_path)
    _persist(
        store,
        intent,
        _observation(intent, ProviderItemStatus.SUCCESS, raw_sha256="a" * 64),
    )
    assert store.operation_state(intent.operation_key) == "VERIFIED"
    assert store.reconciliation_outcome(intent.operation_key) == "matched_completed"

    terminal = _observation(
        intent,
        status,
        observed_at=T0 + timedelta(hours=1),
        raw_sha256="b" * 64,
    )
    assert _persist(store, intent, terminal) is True

    financial = store.latest_financial_outcome(intent.operation_key)
    assert financial["outcome"] == "failed"
    assert financial["provider_status"] == status.value
    assert reason in financial["reason_codes"]
    assert financial["operation_state"] == "VERIFIED"
    assert store.reconciliation_outcome(intent.operation_key) == "failed"
    history = store.reconciliation_history(intent.operation_key)
    assert [entry["provider_status"] for entry in history] == ["SUCCESS", status.value]
    # no second payout construction anywhere in the transition
    dispatch_events = [event for event in _audit_events(store, intent.operation_key)
                       if event["type"] == "PAYOUT_DISPATCH_STARTED"]
    assert len(dispatch_events) == 1


def test_late_readbacks_are_get_only_and_supersede_completed_evidence(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    expected = [
        ("SUCCESS", RecoveryDisposition.RECONCILED),
        ("REFUNDED", RecoveryDisposition.SUPERSEDED),
        ("REVERSED", RecoveryDisposition.SUPERSEDED),
        ("RETURNED", RecoveryDisposition.SUPERSEDED),
    ]
    for status, disposition in expected:
        transport = SyntheticPayPalTransport(intent, status=status)
        assert _readback(store, intent, transport) is disposition
        _assert_no_payout_post(transport)

    history = store.reconciliation_history(intent.operation_key)
    assert [entry["provider_status"] for entry in history] == [
        "SUCCESS",
        "REFUNDED",
        "REVERSED",
        "RETURNED",
    ]
    assert store.latest_financial_outcome(intent.operation_key)["provider_status"] == "RETURNED"
    assert store.reconciliation_count(intent.operation_key) == 4


def test_pending_then_success_through_authenticated_readback(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)

    pending_transport = SyntheticPayPalTransport(intent, status="PENDING")
    assert _readback(store, intent, pending_transport) is RecoveryDisposition.PENDING
    _assert_no_payout_post(pending_transport)
    assert store.operation_state(intent.operation_key) == "ACKNOWLEDGED"

    success_transport = SyntheticPayPalTransport(intent, status="SUCCESS")
    assert _readback(store, intent, success_transport) is RecoveryDisposition.RECONCILED
    _assert_no_payout_post(success_transport)
    assert store.operation_state(intent.operation_key) == "VERIFIED"
    assert store.reconciliation_count(intent.operation_key) == 2


def test_repeated_readback_of_same_item_is_idempotent_over_the_adapter(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    first_transport = SyntheticPayPalTransport(intent, status="SUCCESS")
    assert _readback(store, intent, first_transport) is RecoveryDisposition.RECONCILED
    second_transport = SyntheticPayPalTransport(intent, status="SUCCESS")
    assert _readback(store, intent, second_transport) is RecoveryDisposition.RECONCILED
    _assert_no_payout_post(first_transport)
    _assert_no_payout_post(second_transport)
    assert store.reconciliation_count(intent.operation_key) == 1


def test_no_payout_post_is_constructed_on_any_new_lifecycle_path(tmp_path, monkeypatch):
    store, intent = _acknowledged_operation(tmp_path)
    constructed: list[tuple] = []

    def forbidden(*args, **kwargs):
        constructed.append(args)
        raise AssertionError("a payout POST was constructed on a lifecycle path")

    monkeypatch.setattr(PayPalSandboxAdapter, "create_payout", forbidden)

    for status, minutes, raw in (
        (ProviderItemStatus.PENDING, 0, "a" * 64),
        (ProviderItemStatus.SUCCESS, 1, "b" * 64),
        (ProviderItemStatus.REFUNDED, 2, "c" * 64),
        (ProviderItemStatus.REVERSED, 3, "d" * 64),
        (ProviderItemStatus.RETURNED, 4, "e" * 64),
    ):
        _persist(
            store,
            intent,
            _observation(
                intent, status, observed_at=T0 + timedelta(minutes=minutes), raw_sha256=raw
            ),
        )
    for status in ("SUCCESS", "REFUNDED", "REVERSED", "RETURNED"):
        transport = SyntheticPayPalTransport(intent, status=status)
        _readback(store, intent, transport)
        _assert_no_payout_post(transport)

    assert constructed == []
    assert store.reconciliation_count(intent.operation_key) == 9


def test_out_of_order_ingestion_derives_by_authoritative_order(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)

    refunded = _observation(
        intent,
        ProviderItemStatus.REFUNDED,
        observed_at=T0 + timedelta(minutes=5),
        raw_sha256="c" * 64,
        observation_id="observation-refunded",
    )
    early_success = _observation(
        intent,
        ProviderItemStatus.SUCCESS,
        observed_at=T0,
        raw_sha256="d" * 64,
        observation_id="observation-early-success",
    )
    assert _persist(store, intent, refunded) is True
    assert _persist(store, intent, early_success) is True

    financial = store.latest_financial_outcome(intent.operation_key)
    assert financial["provider_status"] == "REFUNDED"
    assert financial["outcome"] == "failed"
    history = store.reconciliation_history(intent.operation_key)
    assert [entry["provider_status"] for entry in history] == ["SUCCESS", "REFUNDED"]
    # insertion order (reconciliation_id) is the reverse of authoritative order
    assert history[0]["observation_id"] == "observation-early-success"
    assert history[0]["reconciliation_id"] > history[1]["reconciliation_id"]


def test_restart_preserves_superseded_state_and_accepts_next_transition(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    _persist(
        store,
        intent,
        _observation(intent, ProviderItemStatus.SUCCESS, raw_sha256="a" * 64),
    )
    _persist(
        store,
        intent,
        _observation(
            intent,
            ProviderItemStatus.REFUNDED,
            observed_at=T0 + timedelta(hours=1),
            raw_sha256="b" * 64,
        ),
    )

    reopened = MissionPayStore(store.path, password_hasher=_fast_hasher())
    reopened.initialize()
    assert reopened.operation_state(intent.operation_key) == "VERIFIED"
    assert reopened.reconciliation_outcome(intent.operation_key) == "failed"
    assert reopened.latest_financial_outcome(intent.operation_key)["provider_status"] == "REFUNDED"
    assert reopened.reconciliation_count(intent.operation_key) == 2

    transport = SyntheticPayPalTransport(intent, status="REVERSED")
    assert _readback(reopened, intent, transport) is RecoveryDisposition.SUPERSEDED
    _assert_no_payout_post(transport)
    history = reopened.reconciliation_history(intent.operation_key)
    assert [entry["provider_status"] for entry in history] == ["SUCCESS", "REFUNDED", "REVERSED"]
    assert reopened.latest_financial_outcome(intent.operation_key)["provider_status"] == "REVERSED"


def test_late_observations_do_not_change_recipient_or_amount_binding(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    authorization_before = _authorization_row(store, intent.authorization_id)

    for status, minutes, raw in (
        (ProviderItemStatus.SUCCESS, 0, "a" * 64),
        (ProviderItemStatus.REFUNDED, 10, "b" * 64),
        (ProviderItemStatus.REVERSED, 20, "c" * 64),
    ):
        _persist(
            store,
            intent,
            _observation(
                intent, status, observed_at=T0 + timedelta(minutes=minutes), raw_sha256=raw
            ),
        )

    shifted = _observation(
        intent,
        ProviderItemStatus.SUCCESS,
        observed_at=T0 + timedelta(minutes=30),
        amount=Money("USD", 1),
        raw_sha256="e" * 64,
        observation_id="observation-shifted-amount",
    )
    _persist(store, intent, shifted)

    assert store.load_intent(intent.operation_key) == intent
    assert store.operation_binding(intent.operation_key)["payout_item_id"] == ITEM_ID
    assert _authorization_row(store, intent.authorization_id) == authorization_before
    history = store.reconciliation_history(intent.operation_key)
    assert "AMOUNT_MISMATCH" in history[-1]["reason_codes"]


# --------------------------------------------------------------------------- MP-3


def test_exhausted_fund_loads_with_zero_available_and_refuses_next_claim(tmp_path):
    store = _store(tmp_path, fund_cents=7_899)
    alpha = _submit_and_authorize(store, _claim("claim-alpha-001", 7_899, "a" * 64))
    bravo = _submit_and_authorize(store, _claim("claim-bravo-001", 7_899, "b" * 64))

    intent = store.create_payment_intent(alpha.authorization_id)
    assert store.fund_totals(FUND_ID, POLICY_VERSION) == (7_899, 7_899)

    policy = store.load_fund_policy(FUND_ID, POLICY_VERSION)  # must not raise
    assert isinstance(policy.available, FundBalance)
    assert policy.available.cents == 0
    assert policy.available.is_exhausted is True

    next_claim = _claim("claim-charlie-001", 100, "c" * 64)
    decision = evaluate_claim(next_claim, policy, prior_evidence_digests=frozenset())
    assert decision.outcome is PolicyOutcome.INELIGIBLE
    assert "INSUFFICIENT_FUND_BALANCE" in decision.reason_codes

    with pytest.raises(InsufficientFunds, match="INSUFFICIENT_UNRESERVED_FUNDS"):
        store.create_payment_intent(bravo.authorization_id)
    assert store.fund_totals(FUND_ID, POLICY_VERSION) == (7_899, 7_899)
    assert store.existing_operation_for_claim("claim-bravo-001", 1) is None
    assert store.operation_state(intent.operation_key) == "PREPARED"


def test_zero_available_policy_object_is_usable_and_amounts_stay_positive(tmp_path):
    zero_policy = _policy()
    zero_policy = FundPolicy(
        fund_id=zero_policy.fund_id,
        version=zero_policy.version,
        allowed_activity_codes=zero_policy.allowed_activity_codes,
        allowed_expense_categories=zero_policy.allowed_expense_categories,
        starts_on=zero_policy.starts_on,
        ends_on=zero_policy.ends_on,
        available=FundBalance("USD", 0),
        receipt_required=True,
    )
    claim = _claim("claim-zulu-001", 1, "f" * 64)
    decision = evaluate_claim(claim, zero_policy, prior_evidence_digests=frozenset())
    assert decision.outcome is PolicyOutcome.INELIGIBLE
    assert decision.reason_codes == ("INSUFFICIENT_FUND_BALANCE",)

    with pytest.raises(ValueError, match="positive"):
        Money("USD", 0)
    with pytest.raises(ValueError, match="negative"):
        FundBalance("USD", -1)

    store = MissionPayStore(tmp_path / "zero-fund.db", password_hasher=_fast_hasher())
    store.initialize()
    with pytest.raises(ValueError, match="FUND_ALLOCATION_MUST_BE_POSITIVE"):
        store.create_fund(zero_policy)


# --------------------------------------------------------------------------- schema


_LEGACY_OBSERVATIONS_DDL = """
CREATE TABLE provider_observations (
    observation_id TEXT PRIMARY KEY,
    operation_key TEXT NOT NULL REFERENCES payment_operations(operation_key),
    payout_batch_id TEXT NOT NULL,
    payout_item_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    recipient TEXT NOT NULL,
    currency TEXT NOT NULL CHECK (currency = 'USD'),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    observed_at TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL
)
"""

_LEGACY_RECONCILIATIONS_DDL = """
CREATE TABLE reconciliations (
    operation_key TEXT PRIMARY KEY REFERENCES payment_operations(operation_key),
    observation_id TEXT NOT NULL UNIQUE REFERENCES provider_observations(observation_id),
    outcome TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    reconciled_at TEXT NOT NULL
)
"""


def test_legacy_database_migrates_in_place_and_accepts_a_second_observation(tmp_path):
    store, intent = _acknowledged_operation(tmp_path)
    legacy_observation_id = "observation-legacy-pending"
    with sqlite3.connect(store.path) as conn:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS observations_no_update;
            DROP TRIGGER IF EXISTS observations_no_delete;
            DROP TRIGGER IF EXISTS reconciliations_no_update;
            DROP TRIGGER IF EXISTS reconciliations_no_delete;
            DROP TABLE reconciliations;
            DROP TABLE provider_observations;
            """
        )
        conn.executescript(_LEGACY_OBSERVATIONS_DDL + ";" + _LEGACY_RECONCILIATIONS_DDL + ";")
        conn.execute(
            """INSERT INTO provider_observations(
                   observation_id,operation_key,payout_batch_id,payout_item_id,status,
                   recipient,currency,amount_cents,observed_at,raw_sha256
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                legacy_observation_id,
                intent.operation_key,
                BATCH_ID,
                ITEM_ID,
                "PENDING",
                intent.recipient,
                "USD",
                intent.amount.cents,
                (T0 - timedelta(minutes=1)).isoformat(),
                "a" * 64,
            ),
        )
        conn.execute(
            """INSERT INTO reconciliations(
                   operation_key,observation_id,outcome,reason_codes_json,reconciled_at
               ) VALUES(?,?,?,?,?)""",
            (
                intent.operation_key,
                legacy_observation_id,
                "pending",
                json.dumps(["PAYPAL_ITEM_PENDING"]),
                T0.isoformat(),
            ),
        )

    migrated = MissionPayStore(store.path, password_hasher=_fast_hasher())
    migrated.initialize()

    with sqlite3.connect(store.path) as conn:
        observations_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='provider_observations'"
        ).fetchone()[0]
        reconciliations_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='reconciliations'"
        ).fetchone()[0]
        events = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type='PAYOUT_LIFECYCLE_SCHEMA_MIGRATED'"
        ).fetchone()[0]
    assert "UNIQUE" not in observations_sql.split("payout_item_id TEXT NOT NULL")[1].split(",")[0]
    assert "operation_key TEXT PRIMARY KEY" not in reconciliations_sql
    assert events == 1

    assert migrated.reconciliation_count(intent.operation_key) == 1
    assert migrated.reconciliation_outcome(intent.operation_key) == "pending"
    preserved = migrated.reconciliation_history(intent.operation_key)[0]
    assert preserved["observation_id"] == legacy_observation_id
    assert preserved["provider_status"] == "PENDING"

    # the migrated database now accepts the routine asynchronous transition
    assert _persist(
        migrated,
        intent,
        _observation(
            intent,
            ProviderItemStatus.SUCCESS,
            observed_at=T0 + timedelta(minutes=1),
            raw_sha256="b" * 64,
        ),
    ) is True
    assert migrated.reconciliation_count(intent.operation_key) == 2
    assert migrated.reconciliation_outcome(intent.operation_key) == "matched_completed"
    assert migrated.operation_state(intent.operation_key) == "VERIFIED"

    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_PROVIDER_OBSERVATION"):
        with sqlite3.connect(store.path) as conn:
            conn.execute("UPDATE provider_observations SET status='SUCCESS'")

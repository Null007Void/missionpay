from datetime import datetime, timezone

from argon2 import PasswordHasher

from missionpay.auth import Role
from missionpay.domain import (
    Activity,
    Claim,
    EvidenceIdentity,
    FundPolicy,
    Money,
    ProviderItemStatus,
    ProviderObservation,
)
from missionpay.policy import evaluate_claim
from missionpay.reconciliation import reconcile_payment
from missionpay.store import MissionPayStore
from conftest import accept_interpretation, authorize


def _prepared_store(tmp_path):
    store = MissionPayStore(
        tmp_path / "missionpay.db",
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    store.initialize()
    store.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    store.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
    policy = FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=datetime(2026, 1, 1).date(),
        ends_on=datetime(2026, 12, 31).date(),
        available=Money("USD", 50_000),
        receipt_required=True,
    )
    claim = Claim(
        claim_id="claim-river-001",
        volunteer_id="volunteer-alex",
        activity=Activity("activity-river-001", "RIVER_CLEANUP", datetime(2026, 10, 18).date()),
        evidence=EvidenceIdentity("f" * 64, "image/jpeg", "receipt.jpg"),
        requested=Money("USD", 7899),
        recipient="volunteer@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.create_fund(policy)
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
    )
    accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    auth = authorize(store, reviewer.token, claim.claim_id, revision=1,
              record=accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256))
    return store, store.create_payment_intent(auth.authorization_id)


def test_batch_acknowledgement_is_not_completion_until_item_is_retrieved(tmp_path):
    store, intent = _prepared_store(tmp_path)
    store.begin_dispatch(intent.operation_key, "worker-A")
    store.record_payout_acknowledgement(intent.operation_key, "BATCH123")

    assert store.operation_state(intent.operation_key) == "ACKNOWLEDGED"

    observation = ProviderObservation(
        observation_id="observation-001",
        attempt_id=intent.operation_key,
        provider="PAYPAL_SANDBOX",
        payout_batch_id="BATCH123",
        payout_item_id="ITEM123",
        status=ProviderItemStatus.SUCCESS,
        recipient=intent.recipient,
        amount=intent.amount,
        observed_at=datetime.now(timezone.utc),
        raw_sha256="a" * 64,
    )
    reconciliation = reconcile_payment(intent, observation)
    store.persist_reconciliation(intent.operation_key, observation, reconciliation)

    assert store.operation_state(intent.operation_key) == "VERIFIED"
    assert store.reconciliation_outcome(intent.operation_key) == "matched_completed"


def test_uncertain_dispatch_can_recover_by_readback_but_not_new_dispatch(tmp_path):
    store, intent = _prepared_store(tmp_path)
    store.begin_dispatch(intent.operation_key, "worker-A")
    store.mark_dispatch_uncertain(intent.operation_key, "worker-A", "NETWORK_TIMEOUT_AFTER_SEND")

    store.record_payout_acknowledgement(intent.operation_key, "BATCH123")

    assert store.operation_state(intent.operation_key) == "ACKNOWLEDGED"
    assert store.fund_totals("river-cleanup-fund", "2026-03") == (50_000, 7899)


def test_success_status_with_changed_recipient_is_persisted_as_conflict(tmp_path):
    store, intent = _prepared_store(tmp_path)
    store.begin_dispatch(intent.operation_key, "worker-A")
    store.record_payout_acknowledgement(intent.operation_key, "BATCH123")
    observation = ProviderObservation(
        observation_id="observation-conflict",
        attempt_id=intent.operation_key,
        provider="PAYPAL_SANDBOX",
        payout_batch_id="BATCH123",
        payout_item_id="ITEM-CONFLICT",
        status=ProviderItemStatus.SUCCESS,
        recipient="attacker@example.org",
        amount=intent.amount,
        observed_at=datetime.now(timezone.utc),
        raw_sha256="b" * 64,
    )

    reconciliation = reconcile_payment(intent, observation)
    store.persist_reconciliation(intent.operation_key, observation, reconciliation)

    assert store.operation_state(intent.operation_key) == "CONFLICT"
    assert store.reconciliation_outcome(intent.operation_key) == "mismatch"

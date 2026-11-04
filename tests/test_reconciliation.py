from datetime import datetime, timezone

from missionpay.domain import (
    Money,
    PaymentIntent,
    ProviderItemStatus,
    ProviderObservation,
    ReconciliationOutcome,
)
from missionpay.reconciliation import reconcile_payment


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


def _observation(status: ProviderItemStatus = ProviderItemStatus.SUCCESS) -> ProviderObservation:
    return ProviderObservation(
        observation_id="observation-001",
        attempt_id="attempt-001",
        provider="PAYPAL_SANDBOX",
        payout_batch_id="batch-001",
        payout_item_id="item-001",
        status=status,
        recipient="volunteer@example.org",
        amount=Money("USD", 7899),
        observed_at=datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc),
        raw_sha256="3" * 64,
    )


def test_success_requires_exact_independent_item_match():
    matched = reconcile_payment(_intent(), _observation())
    mismatched = reconcile_payment(
        _intent(),
        ProviderObservation(
            observation_id="observation-002",
            attempt_id="attempt-001",
            provider="PAYPAL_SANDBOX",
            payout_batch_id="batch-001",
            payout_item_id="item-001",
            status=ProviderItemStatus.SUCCESS,
            recipient="attacker@example.org",
            amount=Money("USD", 7899),
            observed_at=datetime(2026, 10, 22, 22, 1, tzinfo=timezone.utc),
            raw_sha256="4" * 64,
        ),
    )

    assert matched.outcome is ReconciliationOutcome.MATCHED_COMPLETED
    assert mismatched.outcome is ReconciliationOutcome.MISMATCH
    assert "RECIPIENT_MISMATCH" in mismatched.reason_codes


def test_nonterminal_and_failed_provider_states_never_complete_reimbursement():
    pending = reconcile_payment(_intent(), _observation(ProviderItemStatus.PENDING))
    unclaimed = reconcile_payment(_intent(), _observation(ProviderItemStatus.UNCLAIMED))
    failed = reconcile_payment(_intent(), _observation(ProviderItemStatus.FAILED))
    returned = reconcile_payment(_intent(), _observation(ProviderItemStatus.RETURNED))

    assert pending.outcome is ReconciliationOutcome.PENDING
    assert unclaimed.outcome is ReconciliationOutcome.PENDING
    assert failed.outcome is ReconciliationOutcome.FAILED
    assert returned.outcome is ReconciliationOutcome.FAILED

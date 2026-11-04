from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from missionpay.domain import (
    AuditEvent,
    ExecutionAttempt,
    ExecutionState,
    Money,
    ProviderItemStatus,
    ProviderObservation,
    Reconciliation,
    ReconciliationOutcome,
)


def test_execution_and_provider_records_are_immutable_and_exact():
    now = datetime(2026, 10, 22, 21, 30, tzinfo=timezone.utc)
    attempt = ExecutionAttempt(
        attempt_id="attempt-001",
        operation_key="missionpay:" + "1" * 64,
        request_sha256="2" * 64,
        state=ExecutionState.PREPARED,
        created_at=now,
    )
    observation = ProviderObservation(
        observation_id="observation-001",
        attempt_id=attempt.attempt_id,
        provider="PAYPAL_SANDBOX",
        payout_batch_id="batch-001",
        payout_item_id="item-001",
        status=ProviderItemStatus.SUCCESS,
        recipient="volunteer.alex@example.org",
        amount=Money("USD", 7899),
        observed_at=now,
        raw_sha256="3" * 64,
    )
    reconciliation = Reconciliation(
        operation_key=attempt.operation_key,
        observation_id=observation.observation_id,
        outcome=ReconciliationOutcome.MATCHED_COMPLETED,
        reason_codes=("EXACT_ITEM_MATCH",),
    )
    event = AuditEvent(
        event_id="event-001",
        sequence=1,
        aggregate_id="claim-river-001",
        event_type="PAYMENT_RECONCILED",
        payload_sha256="4" * 64,
        occurred_at=now,
    )

    assert observation.status is ProviderItemStatus.SUCCESS
    assert reconciliation.outcome is ReconciliationOutcome.MATCHED_COMPLETED
    assert event.sequence == 1
    with pytest.raises(FrozenInstanceError):
        attempt.state = ExecutionState.SUBMITTED  # type: ignore[misc]


def test_provider_records_reject_production_or_naive_time():
    now = datetime(2026, 10, 22, 21, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="sandbox"):
        ProviderObservation(
            observation_id="observation-001",
            attempt_id="attempt-001",
            provider="PAYPAL_PRODUCTION",
            payout_batch_id="batch-001",
            payout_item_id="item-001",
            status=ProviderItemStatus.SUCCESS,
            recipient="volunteer.alex@example.org",
            amount=Money("USD", 7899),
            observed_at=now,
            raw_sha256="3" * 64,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        ExecutionAttempt(
            attempt_id="attempt-001",
            operation_key="missionpay:" + "1" * 64,
            request_sha256="2" * 64,
            state=ExecutionState.PREPARED,
            created_at=datetime(2026, 10, 22, 21, 30),
        )

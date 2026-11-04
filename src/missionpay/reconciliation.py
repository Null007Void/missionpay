"""Independent PayPal payout-item reconciliation."""

from __future__ import annotations

from .domain import (
    PaymentIntent,
    ProviderItemStatus,
    ProviderObservation,
    Reconciliation,
    ReconciliationOutcome,
)

_PENDING = {
    ProviderItemStatus.PENDING,
    ProviderItemStatus.UNCLAIMED,
    ProviderItemStatus.ONHOLD,
}
_FAILED = {
    ProviderItemStatus.FAILED,
    ProviderItemStatus.BLOCKED,
    ProviderItemStatus.RETURNED,
    ProviderItemStatus.REFUNDED,
    ProviderItemStatus.REVERSED,
}


def reconcile_payment(
    intent: PaymentIntent,
    observation: ProviderObservation,
) -> Reconciliation:
    mismatches: list[str] = []
    if observation.recipient != intent.recipient:
        mismatches.append("RECIPIENT_MISMATCH")
    if observation.amount.currency != intent.amount.currency:
        mismatches.append("CURRENCY_MISMATCH")
    if observation.amount.cents != intent.amount.cents:
        mismatches.append("AMOUNT_MISMATCH")
    if mismatches:
        return Reconciliation(
            operation_key=intent.operation_key,
            observation_id=observation.observation_id,
            outcome=ReconciliationOutcome.MISMATCH,
            reason_codes=tuple(mismatches),
        )

    if observation.status is ProviderItemStatus.SUCCESS:
        return Reconciliation(
            operation_key=intent.operation_key,
            observation_id=observation.observation_id,
            outcome=ReconciliationOutcome.MATCHED_COMPLETED,
            reason_codes=("EXACT_ITEM_MATCH",),
        )
    if observation.status in _PENDING:
        return Reconciliation(
            operation_key=intent.operation_key,
            observation_id=observation.observation_id,
            outcome=ReconciliationOutcome.PENDING,
            reason_codes=(f"PAYPAL_ITEM_{observation.status.value}",),
        )
    if observation.status in _FAILED:
        return Reconciliation(
            operation_key=intent.operation_key,
            observation_id=observation.observation_id,
            outcome=ReconciliationOutcome.FAILED,
            reason_codes=(f"PAYPAL_ITEM_{observation.status.value}",),
        )
    return Reconciliation(
        operation_key=intent.operation_key,
        observation_id=observation.observation_id,
        outcome=ReconciliationOutcome.UNCERTAIN,
        reason_codes=("PAYPAL_ITEM_STATUS_UNRECOGNIZED",),
    )

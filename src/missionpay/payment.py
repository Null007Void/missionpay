"""Fail-closed creation of immutable payment intents."""

from __future__ import annotations

from hashlib import sha256
import json

from .domain import (
    Claim,
    HumanAuthorization,
    PaymentIntent,
    PolicyDecision,
    PolicyOutcome,
)


class PaymentRefused(ValueError):
    """Raised before any provider operation when financial authority is absent."""


def create_payment_intent(
    claim: Claim,
    decision: PolicyDecision,
    authorization: HumanAuthorization,
) -> PaymentIntent:
    if not authorization.authenticated:
        raise PaymentRefused("AUTHENTICATION_REQUIRED")
    if decision.outcome is not PolicyOutcome.ELIGIBLE:
        raise PaymentRefused("POLICY_NOT_ELIGIBLE")

    exact_claim_binding = (
        authorization.claim_id == claim.claim_id
        and authorization.evidence_sha256 == claim.evidence.sha256
        and authorization.policy_version == decision.policy_version
        and authorization.fund_id == decision.fund_id
        and authorization.recipient == claim.recipient
        and authorization.amount == claim.requested
        and decision.claim_id == claim.claim_id
    )
    if not exact_claim_binding:
        raise PaymentRefused("AUTHORIZATION_BINDING_MISMATCH")

    canonical = json.dumps(
        {
            "authorization_id": authorization.authorization_id,
            "claim_id": claim.claim_id,
            "claim_revision": authorization.claim_revision,
            "evidence_sha256": claim.evidence.sha256,
            "policy_version": decision.policy_version,
            "fund_id": decision.fund_id,
            "recipient": claim.recipient,
            "currency": claim.requested.currency,
            "cents": claim.requested.cents,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    operation_key = f"missionpay:{sha256(canonical).hexdigest()}"

    return PaymentIntent(
        operation_key=operation_key,
        claim_id=claim.claim_id,
        claim_revision=authorization.claim_revision,
        evidence_sha256=claim.evidence.sha256,
        policy_version=decision.policy_version,
        fund_id=decision.fund_id,
        recipient=claim.recipient,
        amount=claim.requested,
        authorization_id=authorization.authorization_id,
    )
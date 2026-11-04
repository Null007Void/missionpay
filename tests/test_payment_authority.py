from dataclasses import replace
from datetime import date

import pytest

from missionpay.domain import (
    Activity,
    Claim,
    EvidenceIdentity,
    FundPolicy,
    HumanAuthorization,
    Money,
)
from missionpay.payment import PaymentRefused, create_payment_intent
from missionpay.policy import evaluate_claim


def _eligible_inputs():
    claim = Claim(
        claim_id="claim-river-001",
        volunteer_id="volunteer-alex",
        activity=Activity("activity-river-001", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("a" * 64, "image/jpeg", "receipt.jpg"),
        requested=Money("USD", 7899),
        recipient="volunteer.alex@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    policy = FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money("USD", 50_000),
        receipt_required=True,
    )
    decision = evaluate_claim(claim, policy, prior_evidence_digests=frozenset())
    authorization = HumanAuthorization(
        authorization_id="auth-river-001",
        reviewer_id="reviewer-morgan",
        reviewer_role="REIMBURSEMENT_REVIEWER",
        authenticated=True,
        claim_id=claim.claim_id,
        claim_revision=1,
        evidence_sha256=claim.evidence.sha256,
        policy_version=policy.version,
        fund_id=policy.fund_id,
        recipient=claim.recipient,
        amount=claim.requested,
    )
    return claim, decision, authorization


def test_unauthenticated_or_tampered_authorization_cannot_create_payment_intent():
    claim, decision, authorization = _eligible_inputs()

    with pytest.raises(PaymentRefused, match="AUTHENTICATION_REQUIRED"):
        create_payment_intent(claim, decision, replace(authorization, authenticated=False))

    with pytest.raises(PaymentRefused, match="AUTHORIZATION_BINDING_MISMATCH"):
        create_payment_intent(
            claim,
            decision,
            replace(authorization, amount=Money("USD", 7999)),
        )


def test_exact_authenticated_authorization_creates_immutable_intent():
    claim, decision, authorization = _eligible_inputs()

    intent = create_payment_intent(claim, decision, authorization)

    assert intent.claim_id == claim.claim_id
    assert intent.amount == Money("USD", 7899)
    assert intent.recipient == "volunteer.alex@example.org"
    assert intent.authorization_id == authorization.authorization_id
    assert intent.operation_key.startswith("missionpay:")

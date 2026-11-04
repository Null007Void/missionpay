"""Executable deterministic demonstration of MissionPay's authority split."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import json

from .domain import (
    Activity,
    Claim,
    EvidenceIdentity,
    FundPolicy,
    HumanAuthorization,
    Money,
)
from .payment import PaymentRefused, create_payment_intent
from .policy import evaluate_claim


def run_signature_demo() -> dict[str, object]:
    claim = Claim(
        claim_id="claim-river-001",
        volunteer_id="volunteer-alex",
        activity=Activity("activity-river-001", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("a" * 64, "image/jpeg", "protective-supplies-receipt.jpg"),
        requested=Money("USD", 7899),
        recipient="volunteer.alex@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    common = {
        "starts_on": date(2026, 1, 1),
        "ends_on": date(2026, 12, 31),
        "available": Money("USD", 50_000),
        "receipt_required": True,
    }
    garden = FundPolicy(
        fund_id="garden-fund",
        version="2026-01",
        allowed_activity_codes=frozenset({"COMMUNITY_GARDEN"}),
        allowed_expense_categories=frozenset({"SEEDS", "GARDEN_TOOLS"}),
        **common,
    )
    river = FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        **common,
    )
    garden_decision = evaluate_claim(claim, garden, prior_evidence_digests=frozenset())
    river_decision = evaluate_claim(claim, river, prior_evidence_digests=frozenset())
    authorization = HumanAuthorization(
        authorization_id="auth-river-001",
        reviewer_id="reviewer-morgan",
        reviewer_role="REIMBURSEMENT_REVIEWER",
        authenticated=True,
        claim_id=claim.claim_id,
        claim_revision=1,
        evidence_sha256=claim.evidence.sha256,
        policy_version=river.version,
        fund_id=river.fund_id,
        recipient=claim.recipient,
        amount=claim.requested,
    )

    unauthenticated: dict[str, object]
    try:
        create_payment_intent(claim, river_decision, replace(authorization, authenticated=False))
    except PaymentRefused as exc:
        unauthenticated = {"result": "refused", "reason": str(exc)}
    else:  # pragma: no cover - defensive invariant
        raise AssertionError("unauthenticated payment was not refused")

    intent = create_payment_intent(claim, river_decision, authorization)
    return {
        "garden_fund": {
            "outcome": garden_decision.outcome.value,
            "reasons": list(garden_decision.reason_codes),
        },
        "river_cleanup_fund": {
            "outcome": river_decision.outcome.value,
            "reasons": list(river_decision.reason_codes),
        },
        "unauthenticated_payment": unauthenticated,
        "exact_authorized_intent": {
            "created": True,
            "operation_key": intent.operation_key,
            "amount_cents": intent.amount.cents,
            "currency": intent.amount.currency,
        },
        "provider_execution": "not_attempted",
    }


def main() -> None:
    print(json.dumps(run_signature_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

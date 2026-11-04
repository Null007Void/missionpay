from datetime import date

from missionpay.domain import (
    Activity,
    Claim,
    EvidenceIdentity,
    FundPolicy,
    Money,
    PolicyOutcome,
)
from missionpay.policy import evaluate_claim


def _signature_claim() -> Claim:
    return Claim(
        claim_id="claim-river-001",
        volunteer_id="volunteer-alex",
        activity=Activity(
            activity_id="activity-river-001",
            activity_code="RIVER_CLEANUP",
            occurred_on=date(2026, 10, 18),
        ),
        evidence=EvidenceIdentity(
            sha256="a" * 64,
            media_type="image/jpeg",
            original_name="protective-supplies-receipt.jpg",
        ),
        requested=Money(currency="USD", cents=7899),
        recipient="volunteer.alex@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )


def test_signature_claim_is_refused_by_garden_fund_and_allowed_by_river_fund():
    claim = _signature_claim()
    garden = FundPolicy(
        fund_id="garden-fund",
        version="2026-01",
        allowed_activity_codes=frozenset({"COMMUNITY_GARDEN"}),
        allowed_expense_categories=frozenset({"SEEDS", "GARDEN_TOOLS"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money(currency="USD", cents=50_000),
        receipt_required=True,
    )
    river = FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money(currency="USD", cents=50_000),
        receipt_required=True,
    )

    garden_decision = evaluate_claim(claim, garden, prior_evidence_digests=frozenset())
    river_decision = evaluate_claim(claim, river, prior_evidence_digests=frozenset())

    assert garden_decision.outcome is PolicyOutcome.INELIGIBLE
    assert "FUND_ACTIVITY_RESTRICTED" in garden_decision.reason_codes
    assert river_decision.outcome is PolicyOutcome.ELIGIBLE
    assert river_decision.reason_codes == ("ELIGIBLE",)
    assert river_decision.claim_id == claim.claim_id
    assert river_decision.fund_id == river.fund_id
    assert river_decision.policy_version == river.version

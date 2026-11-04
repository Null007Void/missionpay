from dataclasses import replace
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


def _claim(*, category: str | None = "PROTECTIVE_SUPPLIES") -> Claim:
    return Claim(
        claim_id="claim-001",
        volunteer_id="volunteer-001",
        activity=Activity("activity-001", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("b" * 64, "image/jpeg", "receipt.jpg"),
        requested=Money("USD", 7899),
        recipient="volunteer@example.org",
        expense_category=category,
    )


def _policy() -> FundPolicy:
    return FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money("USD", 10_000),
        receipt_required=True,
    )


def test_unknown_category_needs_review_only_when_no_hard_refusal_exists():
    claim = _claim(category=None)

    undecidable = evaluate_claim(claim, _policy(), prior_evidence_digests=frozenset())
    duplicate = evaluate_claim(
        claim,
        _policy(),
        prior_evidence_digests=frozenset({claim.evidence.sha256}),
    )

    assert undecidable.outcome is PolicyOutcome.NEEDS_REVIEW
    assert undecidable.reason_codes == ("EXPENSE_CATEGORY_UNKNOWN",)
    assert duplicate.outcome is PolicyOutcome.INELIGIBLE
    assert "DUPLICATE_EVIDENCE" in duplicate.reason_codes


def test_time_window_and_budget_fail_closed_with_reproducible_reasons():
    claim = replace(
        _claim(),
        activity=Activity("activity-001", "RIVER_CLEANUP", date(2025, 12, 31)),
        requested=Money("USD", 10_001),
    )

    decision = evaluate_claim(claim, _policy(), prior_evidence_digests=frozenset())

    assert decision.outcome is PolicyOutcome.INELIGIBLE
    assert decision.reason_codes == (
        "OUTSIDE_POLICY_WINDOW",
        "INSUFFICIENT_FUND_BALANCE",
    )

"""Deterministic restricted-fund policy evaluation."""

from __future__ import annotations

from typing import AbstractSet

from .domain import Claim, FundPolicy, PolicyDecision, PolicyOutcome


def evaluate_claim(
    claim: Claim,
    policy: FundPolicy,
    *,
    prior_evidence_digests: AbstractSet[str],
) -> PolicyDecision:
    """Evaluate exact typed facts without consulting model prose."""
    refusal_reasons: list[str] = []

    if claim.evidence.sha256 in prior_evidence_digests:
        refusal_reasons.append("DUPLICATE_EVIDENCE")
    if claim.activity.activity_code not in policy.allowed_activity_codes:
        refusal_reasons.append("FUND_ACTIVITY_RESTRICTED")
    if not (policy.starts_on <= claim.activity.occurred_on <= policy.ends_on):
        refusal_reasons.append("OUTSIDE_POLICY_WINDOW")
    if claim.requested.currency != policy.available.currency:
        refusal_reasons.append("CURRENCY_MISMATCH")
    if claim.requested.cents > policy.available.cents:
        refusal_reasons.append("INSUFFICIENT_FUND_BALANCE")

    if claim.expense_category is None:
        if not refusal_reasons:
            return PolicyDecision(
                claim_id=claim.claim_id,
                fund_id=policy.fund_id,
                policy_version=policy.version,
                outcome=PolicyOutcome.NEEDS_REVIEW,
                reason_codes=("EXPENSE_CATEGORY_UNKNOWN",),
            )
    elif claim.expense_category not in policy.allowed_expense_categories:
        refusal_reasons.append("FUND_EXPENSE_RESTRICTED")

    if refusal_reasons:
        return PolicyDecision(
            claim_id=claim.claim_id,
            fund_id=policy.fund_id,
            policy_version=policy.version,
            outcome=PolicyOutcome.INELIGIBLE,
            reason_codes=tuple(refusal_reasons),
        )

    return PolicyDecision(
        claim_id=claim.claim_id,
        fund_id=policy.fund_id,
        policy_version=policy.version,
        outcome=PolicyOutcome.ELIGIBLE,
        reason_codes=("ELIGIBLE",),
    )

from dataclasses import replace
from datetime import date

import pytest
from argon2 import PasswordHasher

from missionpay.auth import AuthorizationDenied, Role
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.policy import evaluate_claim
from missionpay.store import MissionPayStore, StaleAuthorization
from conftest import accept_interpretation, authorize


@pytest.fixture
def store(tmp_path):
    result = MissionPayStore(
        tmp_path / "missionpay.db",
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    result.initialize()
    return result


def _claim(amount: int = 7899) -> Claim:
    return Claim(
        claim_id="claim-river-001",
        volunteer_id="volunteer-alex",
        activity=Activity("activity-river-001", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("e" * 64, "image/jpeg", "receipt.jpg"),
        requested=Money("USD", amount),
        recipient="volunteer.alex@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )


def _policy() -> FundPolicy:
    return FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money("USD", 50_000),
        receipt_required=True,
    )


def test_authenticated_reviewer_is_separate_and_authorization_uses_stored_values(store):
    store.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    store.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
    policy = _policy()
    claim = _claim()
    store.create_fund(policy)
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
    )

    record = accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    with pytest.raises(AuthorizationDenied, match="REVIEWER_ROLE_REQUIRED"):
        authorize(store, volunteer.token, claim.claim_id, revision=1, record=record)

    authorization = authorize(store, reviewer.token, claim.claim_id, revision=1, record=record)

    assert authorization.reviewer_id == "reviewer-morgan"
    assert authorization.claim_revision == 1
    assert authorization.evidence_sha256 == claim.evidence.sha256
    assert authorization.fund_id == policy.fund_id
    assert authorization.policy_version == policy.version
    assert authorization.recipient == claim.recipient
    assert authorization.amount == claim.requested


def test_changed_claim_revision_requires_fresh_authorization(store):
    store.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    store.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
    policy = _policy()
    claim = _claim()
    store.create_fund(policy)
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
    )
    accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    authorization = authorize(store, reviewer.token, claim.claim_id, revision=1,
              record=accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256))

    changed = replace(claim, requested=Money("USD", 8999))
    store.submit_claim(volunteer.token, changed, revision=2)
    store.record_policy_decision(
        evaluate_claim(changed, policy, prior_evidence_digests=frozenset()), revision=2
    )

    with pytest.raises(StaleAuthorization, match="CLAIM_REVISION_CHANGED"):
        store.create_payment_intent(authorization.authorization_id)

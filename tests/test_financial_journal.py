from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import sqlite3

import pytest
from argon2 import PasswordHasher

from missionpay.auth import Role
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.policy import evaluate_claim
from missionpay.store import DispatchBlocked, InsufficientFunds, MissionPayStore
from conftest import accept_interpretation, authorize


@pytest.fixture
def store(tmp_path):
    result = MissionPayStore(
        tmp_path / "missionpay.db",
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    result.initialize()
    result.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    result.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    return result


def _policy(available: int = 10_000) -> FundPolicy:
    return FundPolicy(
        fund_id="river-cleanup-fund",
        version="2026-03",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1),
        ends_on=date(2026, 12, 31),
        available=Money("USD", available),
        receipt_required=True,
    )


def _authorized(store, suffix: str, amount: int):
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
    claim = Claim(
        claim_id=f"claim-{suffix}",
        volunteer_id="volunteer-alex",
        activity=Activity(f"activity-{suffix}", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity((suffix[0] * 64), "image/jpeg", f"receipt-{suffix}.jpg"),
        requested=Money("USD", amount),
        recipient=f"{suffix}@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.submit_claim(volunteer.token, claim, revision=1)
    policy = _policy()
    store.record_policy_decision(
        evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
    )
    accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    return authorize(store, reviewer.token, claim.claim_id, revision=1,
              record=accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256))


def test_atomic_reservation_allows_only_one_competing_intent(store):
    store.create_fund(_policy())
    first = _authorized(store, "alpha", 7000)
    second = _authorized(store, "bravo", 7000)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.create_payment_intent, first.authorization_id),
            pool.submit(store.create_payment_intent, second.authorization_id),
        ]
    results = []
    failures = []
    for future in futures:
        try:
            results.append(future.result())
        except InsufficientFunds as exc:
            failures.append(str(exc))

    assert len(results) == 1
    assert failures == ["INSUFFICIENT_UNRESERVED_FUNDS"]
    assert store.fund_totals("river-cleanup-fund", "2026-03") == (10_000, 7_000)


def test_duplicate_action_and_stale_owner_never_create_another_operation(store):
    store.create_fund(_policy())
    authorization = _authorized(store, "charlie", 7899)
    first = store.create_payment_intent(authorization.authorization_id)
    duplicate = store.create_payment_intent(authorization.authorization_id)

    assert duplicate.operation_key == first.operation_key
    assert store.fund_totals("river-cleanup-fund", "2026-03") == (10_000, 7899)

    store.begin_dispatch(first.operation_key, "worker-A")
    with pytest.raises(DispatchBlocked, match="DISPATCH_ALREADY_CLAIMED"):
        store.begin_dispatch(first.operation_key, "worker-B")

    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE payment_operations SET updated_at=? WHERE operation_key=?",
            (old, first.operation_key),
        )
    with pytest.raises(DispatchBlocked, match="DISPATCH_ALREADY_CLAIMED"):
        store.begin_dispatch(first.operation_key, "worker-C")


def test_restart_converts_inflight_to_uncertain_and_blocks_redispatch(store):
    store.create_fund(_policy())
    authorization = _authorized(store, "delta", 7899)
    intent = store.create_payment_intent(authorization.authorization_id)
    store.begin_dispatch(intent.operation_key, "worker-A")

    reopened = MissionPayStore(
        store.path,
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    reopened.initialize()
    assert reopened.recover_interrupted_dispatches() == 1
    assert reopened.operation_state(intent.operation_key) == "UNCERTAIN"

    with pytest.raises(DispatchBlocked, match="UNCERTAIN_REQUIRES_RECONCILIATION"):
        reopened.begin_dispatch(intent.operation_key, "worker-B")


def test_concurrent_executors_allow_exactly_one_dispatch_owner(store):
    store.create_fund(_policy())
    authorization = _authorized(store, "echo", 7899)
    intent = store.create_payment_intent(authorization.authorization_id)

    def attempt(owner):
        try:
            store.begin_dispatch(intent.operation_key, owner)
            return "won"
        except DispatchBlocked:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["worker-A", "worker-B"]))

    assert sorted(results) == ["blocked", "won"]
    assert store.operation_state(intent.operation_key) == "SENDING"


def test_authority_intent_and_audit_records_are_database_immutable(store):
    store.create_fund(_policy())
    authorization = _authorized(store, "foxtrot", 7899)
    intent = store.create_payment_intent(authorization.authorization_id)

    with sqlite3.connect(store.path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_AUTHORIZATION"):
            conn.execute(
                "UPDATE authorizations SET amount_cents=1 WHERE authorization_id=?",
                (authorization.authorization_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_PAYMENT_INTENT"):
            conn.execute(
                "DELETE FROM payment_intents WHERE operation_key=?",
                (intent.operation_key,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY_AUDIT"):
            conn.execute("DELETE FROM audit_events")

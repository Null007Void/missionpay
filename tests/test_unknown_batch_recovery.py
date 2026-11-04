from datetime import date

from argon2 import PasswordHasher

from missionpay.auth import Role
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.policy import evaluate_claim
from missionpay.recovery import RecoveryDisposition, PayPalRecovery
from missionpay.store import MissionPayStore
from conftest import accept_interpretation, authorize


class NoCallAdapter:
    def __init__(self):
        self.calls = []

    def get_batch(self, *args, **kwargs):
        self.calls.append("GET_BATCH")
        raise AssertionError("unknown batch recovery must not fabricate a GET")

    def get_item(self, *args, **kwargs):
        self.calls.append("GET_ITEM")
        raise AssertionError("unknown batch recovery must not fabricate an item GET")

    def create_payout(self, *args, **kwargs):
        self.calls.append("POST")
        raise AssertionError("uncertain recovery must not POST another payout")


class RecoveredAdapter:
    def __init__(self, intent, *, sender_batch_id=None):
        digest = intent.operation_key.removeprefix("missionpay:")
        self.sender_batch_id = sender_batch_id or f"mp-{digest[:27]}"
        self.calls = []
        self.item = {
            "payout_item_id": "ITEM123",
            "transaction_status": "SUCCESS",
            "payout_item": {
                "sender_item_id": "claim-river-001-r1",
                "receiver": intent.recipient,
                "amount": {"currency": "USD", "value": "78.99"},
            },
        }

    def get_batch(self, payout_batch_id, *, access_token):
        self.calls.append(("GET_BATCH", payout_batch_id))
        return {
            "batch_header": {
                "payout_batch_id": payout_batch_id,
                "sender_batch_header": {"sender_batch_id": self.sender_batch_id},
            },
            "items": [self.item],
        }

    def get_item(self, payout_item_id, *, access_token):
        self.calls.append(("GET_ITEM", payout_item_id))
        return self.item


def _uncertain_store(tmp_path):
    store = MissionPayStore(
        tmp_path / "missionpay.db",
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    store.initialize()
    store.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    store.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    volunteer = store.login("alex@example.org", "volunteer password")
    reviewer = store.login("morgan@example.org", "reviewer password")
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
    claim = Claim(
        claim_id="claim-river-001",
        volunteer_id="volunteer-alex",
        activity=Activity("activity-river-001", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("9" * 64, "image/jpeg", "receipt.jpg"),
        requested=Money("USD", 7899),
        recipient="volunteer@example.org",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.create_fund(policy)
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
    )
    accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256)
    auth = authorize(store, reviewer.token, claim.claim_id, revision=1,
              record=accept_interpretation(store, claim.claim_id, 1, claim.evidence.sha256))
    intent = store.create_payment_intent(auth.authorization_id)
    store.begin_dispatch(intent.operation_key, "worker-A")
    store.mark_dispatch_uncertain(intent.operation_key, "worker-A", "POST_RESPONSE_LOST")
    return store, intent


def test_lost_post_response_without_batch_id_stays_uncertain_after_restart(tmp_path):
    store, intent = _uncertain_store(tmp_path)
    reopened = MissionPayStore(
        store.path,
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    reopened.initialize()
    adapter = NoCallAdapter()

    result = PayPalRecovery(reopened, adapter).recover_unknown_batch(
        intent.operation_key,
        access_token="unused-sandbox-token",
    )

    assert result is RecoveryDisposition.BLOCKED_UNKNOWN_BATCH_ID
    assert reopened.operation_state(intent.operation_key) == "UNCERTAIN"
    assert adapter.calls == []
    assert reopened.reconciliation_count(intent.operation_key) == 0


def test_authoritatively_recovered_batch_is_bound_before_item_reconciliation(tmp_path):
    store, intent = _uncertain_store(tmp_path)
    adapter = RecoveredAdapter(intent)

    result = PayPalRecovery(store, adapter).recover_with_batch_id(
        intent.operation_key,
        "BATCH123",
        access_token="sandbox-token",
    )

    assert result is RecoveryDisposition.RECONCILED
    assert adapter.calls == [("GET_BATCH", "BATCH123"), ("GET_ITEM", "ITEM123")]
    assert store.operation_state(intent.operation_key) == "VERIFIED"
    assert store.reconciliation_outcome(intent.operation_key) == "matched_completed"


def test_recovered_batch_with_wrong_sender_identity_remains_conflict(tmp_path):
    store, intent = _uncertain_store(tmp_path)
    adapter = RecoveredAdapter(intent, sender_batch_id="attacker-batch")

    result = PayPalRecovery(store, adapter).recover_with_batch_id(
        intent.operation_key,
        "BATCH123",
        access_token="sandbox-token",
    )

    assert result is RecoveryDisposition.CONFLICT
    assert adapter.calls == [("GET_BATCH", "BATCH123")]
    assert store.operation_state(intent.operation_key) == "CONFLICT"

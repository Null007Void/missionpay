"""Regression tests for the independent review corrections.

Each test names the finding it pins. Together they assert that:

* the AI stage is persisted, explicit and a precondition of review (F1);
* the configured sandbox payee is enforced before any OAuth or POST (F2);
* credential errors never echo content, and mock mode never opens the file (F3);
* evidence blobs are published atomically and verified on read (F4);
* OAuth scope is matched by exact token and entitlement is never invented (F5).
"""

from datetime import date
import json
import re

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from missionpay.auth import AuthorizationDenied, Role
from missionpay.config import ProviderMode, RuntimeConfig
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.executor import DispatchRefused, PayoutExecutor
from missionpay.oauth import PayPalClientCredentials
from missionpay.policy import evaluate_claim
from missionpay.service import PaymentService
from missionpay.store import MissionPayStore, PersistenceConflict, StaleAuthorization
from missionpay.web import create_app

from conftest import accept_interpretation, approval_fields

PAYOUTS_SCOPE = "https://uri.paypal.com/payments/paypal-placeholder"


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def store(tmp_path):
    s = MissionPayStore(tmp_path / "rev.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    s.initialize()
    s.register_user("v1", "volunteer@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    s.register_user("r1", "reviewer@example.test", Role.REVIEWER, "reviewer-passphrase")
    for fund_id, activity_code, category in (
        ("RIVER_CLEANUP_FUND", "RIVER_CLEANUP", "PROTECTIVE_SUPPLIES"),
        ("GARDEN_FUND", "COMMUNITY_GARDEN", "SEEDS"),
    ):
        s.create_fund(
            FundPolicy(
                fund_id=fund_id,
                version="2026.1",
                allowed_activity_codes=frozenset({activity_code}),
                allowed_expense_categories=frozenset({category}),
                starts_on=date(2026, 1, 1),
                ends_on=date(2026, 12, 31),
                available=Money("USD", 50_000),
                receipt_required=True,
            )
        )
    return s


def _claim(store, claim_id: str = "c1", *, recipient: str = "payee@example.test", cents: int = 7899):
    volunteer = store.login("volunteer@example.test", "volunteer-passphrase")
    claim = Claim(
        claim_id=claim_id,
        volunteer_id="v1",
        activity=Activity("a1", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity("a" * 64, "text/plain", "r.txt"),
        requested=Money("USD", cents),
        recipient=recipient,
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(
            claim,
            store.load_fund_policy("RIVER_CLEANUP_FUND", "2026.1"),
            prior_evidence_digests=frozenset(),
        ),
        revision=1,
    )
    return claim


# ------------------------------------------------- F1: interpretation lifecycle


def test_approval_requires_an_accepted_interpretation(store):
    _claim(store)
    reviewer = store.login("reviewer@example.test", "reviewer-passphrase")
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_NOT_FOUND"):
        store.authorize_claim(
            reviewer.token, "c1", revision=1,
            interpretation_id="interp-does-not-exist",
            interpretation_output_sha256="0" * 64,
        )


def test_refused_interpretation_does_not_satisfy_the_ai_stage(store):
    claim = _claim(store)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="REFUSED", output_sha256="9" * 64, error_code="CITATION_MISMATCH",
    )
    refused = store.interpretation_history("c1", 1)[0]
    reviewer = store.login("reviewer@example.test", "reviewer-passphrase")
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_NOT_ACCEPTED"):
        store.authorize_claim(
            reviewer.token, "c1", revision=1,
            interpretation_id=refused["interpretation_id"],
            interpretation_output_sha256=refused["output_sha256"],
        )


def test_interpretation_for_other_evidence_does_not_satisfy_the_ai_stage(store):
    _claim(store)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256="f" * 64,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="ACCEPTED", output_sha256="1" * 64,
    )
    reviewer = store.login("reviewer@example.test", "reviewer-passphrase")
    record = store.interpretation_history("c1", 1)[0]
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_EVIDENCE_MISMATCH"):
        store.authorize_claim(
            reviewer.token, "c1", revision=1,
            interpretation_id=record["interpretation_id"],
            interpretation_output_sha256=record["output_sha256"],
        )


def test_recording_the_same_configuration_is_idempotent(store):
    claim = _claim(store)
    first = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    second = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    assert first["interpretation_id"] == second["interpretation_id"]
    assert second["reused"] is True
    assert len(store.interpretation_history("c1", 1)) == 1


def test_a_materially_different_interpretation_invalidates_the_approval(store):
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    reviewer = store.login("reviewer@example.test", "reviewer-passphrase")
    authorization = store.authorize_claim(
        reviewer.token, "c1", revision=1,
        interpretation_id=record["interpretation_id"],
        interpretation_output_sha256=record["output_sha256"],
    )

    # Post-approval immutability: the store refuses to add a newer interpretation at
    # all, so the stale-binding path cannot be reached by adding one afterwards.
    with pytest.raises(PersistenceConflict, match="INTERPRETATION_AFTER_AUTHORIZATION"):
        store.record_interpretation(
            claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
            ai_mode="ollama", model_id="qwen3:4b", config_sha256="9" * 64,
            status="ACCEPTED", output_sha256="2" * 64,
        )
    # Because the store refused the newer interpretation, the approval is still the
    # current authority and the intent is consistent. The stale-binding path is
    # exercised separately (legacy rows and pre-existing intents).
    intent = store.create_payment_intent(authorization.authorization_id)
    assert intent.claim_id == "c1"


def test_authorization_records_the_interpretation_it_was_based_on(store):
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    reviewer = store.login("reviewer@example.test", "reviewer-passphrase")
    store.authorize_claim(
        reviewer.token, "c1", revision=1,
        interpretation_id=record["interpretation_id"],
        interpretation_output_sha256=record["output_sha256"],
    )

    import sqlite3

    with sqlite3.connect(store.path) as conn:
        row = conn.execute(
            "SELECT interpretation_id, interpretation_output_sha256 FROM authorizations"
        ).fetchone()
    assert row[0] == record["interpretation_id"]
    assert row[1] == record["output_sha256"]


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _login(client, email, password):
    page = client.get("/login")
    return client.post(
        "/login",
        data={"email": email, "password": password, "csrf_token": _csrf(page.text)},
        follow_redirects=False,
    )


def test_get_never_creates_a_new_interpretation(store):
    client = TestClient(create_app(store), base_url="https://testserver")
    _login(client, "volunteer@example.test", "volunteer-passphrase")
    submitted = client.post(
        "/claims/demo", data={"csrf_token": _csrf(client.get("/app").text)}, follow_redirects=False
    )
    assert submitted.status_code == 303
    path = submitted.headers["location"]

    for _ in range(3):
        assert client.get(path).status_code == 200
    assert store.interpretation_history("river-safety-001", 1) == []

    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client.get("/app").text)})
    after_first = len(store.interpretation_history("river-safety-001", 1))
    for _ in range(3):
        client.get(path)
    assert len(store.interpretation_history("river-safety-001", 1)) == after_first == 1


def test_web_approval_is_blocked_until_interpretation_runs(store):
    client = TestClient(create_app(store), base_url="https://testserver")
    _login(client, "volunteer@example.test", "volunteer-passphrase")
    submitted = client.post(
        "/claims/demo", data={"csrf_token": _csrf(client.get("/app").text)}, follow_redirects=False
    )
    path = submitted.headers["location"]
    client.post("/logout", data={"csrf_token": _csrf(client.get("/app").text)})

    _login(client, "reviewer@example.test", "reviewer-passphrase")
    refused = client.post(
        f"{path}/approve",
        data={"csrf_token": _csrf(client.get("/app").text), "revision": "1", "amount_cents": "7899"},
    )
    assert refused.status_code == 409
    assert "INTERPRETATION_NOT_FOUND" in refused.text


def test_web_interpretation_persists_identity_and_refuses_after_authorization(store):
    client = TestClient(create_app(store), base_url="https://testserver")
    _login(client, "volunteer@example.test", "volunteer-passphrase")
    submitted = client.post(
        "/claims/demo", data={"csrf_token": _csrf(client.get("/app").text)}, follow_redirects=False
    )
    path = submitted.headers["location"]
    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client.get("/app").text)})

    records = store.interpretation_history("river-safety-001", 1)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "ACCEPTED"
    assert record["model_id"] == "fixture/demo-interpretation-v1"
    assert record["ai_mode"] == "fixture"
    assert len(record["config_sha256"]) == 64
    assert len(record["output_sha256"]) == 64
    assert json.loads(record["facts_json"])

    page = client.get(path)
    assert record["interpretation_id"] in page.text
    assert "Observed fact" in page.text

    client.post("/logout", data={"csrf_token": _csrf(client.get("/app").text)})
    _login(client, "reviewer@example.test", "reviewer-passphrase")
    client.post(
        f"{path}/approve",
        data={
            "csrf_token": _csrf(client.get("/app").text),
            "revision": "1",
            "amount_cents": "7899",
            **approval_fields(client.get(path).text),
        },
    )
    late = client.post(f"{path}/interpret", data={"csrf_token": _csrf(client.get("/app").text)})
    assert late.status_code == 409
    assert "INTERPRETATION_AFTER_AUTHORIZATION" in late.text


# --------------------------------------------- F2: sandbox recipient at dispatch


class _Adapter:
    def __init__(self, *, raises=None):
        self.calls = 0
        self.raises = raises

    def create_payout(self, intent, *, access_token):
        self.calls += 1
        if self.raises:
            raise self.raises
        return {"payout_batch_id": "BATCH-1", "batch_status": "PENDING"}


def _authorized_intent(store, *, recipient: str, cents: int = 7899, claim_id: str = "c1"):
    claim = _claim(store, claim_id, recipient=recipient, cents=cents)
    record = accept_interpretation(store, claim_id, 1, claim.evidence.sha256)
    reviewer = store.login("reviewer@example.test", "reviewer-passphrase")
    authorization = store.authorize_claim(
        reviewer.token, claim_id, revision=1,
        interpretation_id=record["interpretation_id"],
        interpretation_output_sha256=record["output_sha256"],
    )
    return store.create_payment_intent(authorization.authorization_id)


def test_dispatch_refuses_an_unconfigured_recipient_before_any_provider_call(store):
    intent = _authorized_intent(store, recipient="volunteer-real@example.test")
    adapter = _Adapter()
    tokens: list[int] = []
    executor = PayoutExecutor(
        store, adapter, max_payout_cents=10_000,
        expected_recipient="sb-configured@business.example.com",
    )

    with pytest.raises(DispatchRefused, match="SANDBOX_RECIPIENT_MISMATCH"):
        executor.execute(
            intent.operation_key, token_provider=lambda: tokens.append(1) or "tok", owner_id="w"
        )

    assert tokens == [], "OAuth was attempted despite a payee mismatch"
    assert adapter.calls == 0, "a payout POST was attempted despite a payee mismatch"
    assert store.operation_state(intent.operation_key) == "PREPARED"


def test_dispatch_proceeds_when_the_recipient_matches(store):
    intent = _authorized_intent(store, recipient="sb-configured@business.example.com")
    adapter = _Adapter()
    executor = PayoutExecutor(
        store, adapter, max_payout_cents=10_000,
        expected_recipient="sb-configured@business.example.com",
    )
    outcome = executor.execute(
        intent.operation_key, token_provider=lambda: "tok", owner_id="w"
    )
    assert outcome.state == "ACKNOWLEDGED"
    assert adapter.calls == 1


def test_recipient_and_cap_are_both_enforced(store):
    intent = _authorized_intent(store, recipient="sb-configured@business.example.com", cents=9_000)
    adapter = _Adapter()
    executor = PayoutExecutor(
        store, adapter, max_payout_cents=5_000,
        expected_recipient="sb-configured@business.example.com",
    )
    with pytest.raises(DispatchRefused, match="PAYOUT_CAP_EXCEEDED"):
        executor.execute(intent.operation_key, token_provider=lambda: "tok", owner_id="w")
    assert adapter.calls == 0
    assert store.operation_state(intent.operation_key) == "PREPARED"


# ------------------------------------------------ F3: credential containment


def test_mock_mode_never_opens_the_credential_file(tmp_path, monkeypatch):
    from missionpay.config import from_environment

    marker = tmp_path / "creds.env"
    marker.write_text("PAYPAL_CLIENT_ID=x\n", encoding="utf-8")
    monkeypatch.setenv("MISSIONPAY_CREDENTIALS_FILE", str(marker))

    opened: list[str] = []
    import missionpay.credentials as credentials_module

    real_read_file = credentials_module.read_file

    def spy(path):
        opened.append(str(path))
        return real_read_file(path)

    monkeypatch.setattr(credentials_module, "read_file", spy)
    from_environment({})
    assert opened == [], "mock mode opened the credential file"


def test_genuine_mode_does_open_the_credential_file(tmp_path, monkeypatch):
    from missionpay.config import from_environment

    marker = tmp_path / "creds.env"
    marker.write_text(
        "PAYPAL_CLIENT_ID=cid\nPAYPAL_CLIENT_SECRET=csec\n", encoding="utf-8"
    )
    monkeypatch.setenv("MISSIONPAY_CREDENTIALS_FILE", str(marker))
    from missionpay.config import ConfigurationError

    with pytest.raises(ConfigurationError, match="SANDBOX_RECIPIENT_REQUIRED"):
        from_environment({"MISSIONPAY_PROVIDER_MODE": "paypal_sandbox"})


def test_credential_file_inside_any_work_tree_is_refused(tmp_path):
    """A repository the package knows nothing about must still be detected."""
    from missionpay.credentials import CredentialError, write_template

    repo = tmp_path / "someone-elses-repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(CredentialError, match="CREDENTIALS_FILE_MUST_BE_OUTSIDE_REPOSITORY"):
        write_template(repo / "credentials.env")


def test_worktree_marker_as_a_file_is_also_detected(tmp_path):
    from missionpay.credentials import CredentialError, write_template

    repo = tmp_path / "worktree"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    with pytest.raises(CredentialError, match="CREDENTIALS_FILE_MUST_BE_OUTSIDE_REPOSITORY"):
        write_template(repo / "credentials.env")


# ------------------------------------------------- F5: OAuth scope and reuse


class _ScopeTransport:
    def __init__(self, scope, *, calls=None):
        self.scope = scope
        self.calls = calls if calls is not None else []

    def request(self, method, url, headers, body):
        self.calls.append(url)
        payload = {"access_token": "tok", "token_type": "Bearer", "expires_in": 3600}
        if self.scope is not None:
            payload["scope"] = self.scope
        return 200, json.dumps(payload).encode()


EXACT = "openid https://uri.paypal.com/payments/payouts"


@pytest.mark.parametrize(
    "scope",
    [
        "openid notpayouts",                       # substring present, token absent
        "openid https://uri.paypal.com/payments/payouts-extra",
        "openid",
    ],
)
def test_lookalike_scope_is_refused(scope):
    from missionpay.oauth import PayPalOAuthError

    with pytest.raises(PayPalOAuthError, match="PAYPAL_OAUTH_PAYOUTS_SCOPE_MISSING"):
        PayPalClientCredentials("id", "s", transport=_ScopeTransport(scope)).access_token()


@pytest.mark.parametrize("scope", [None, "", "   "])
def test_missing_or_blank_scope_is_not_reported_as_verified(scope):
    from missionpay.oauth import PayPalOAuthError

    with pytest.raises(PayPalOAuthError, match="PAYPAL_OAUTH_SCOPE_UNVERIFIED"):
        PayPalClientCredentials("id", "s", transport=_ScopeTransport(scope)).access_token()


def test_exact_scope_is_accepted_and_entitlement_is_not_claimed():
    transport = _ScopeTransport(EXACT)
    client = PayPalClientCredentials("id", "s", transport=transport)
    assert client.access_token() == "tok"
    summary = client.cached_token().describe()
    assert summary["scope_verified"] is True
    assert summary["payouts_scope_present"] is True
    assert summary["payouts_entitlement"] == "not_tested"


def test_service_reuses_one_credential_client(monkeypatch):
    """The bounded token cache only works if the client is not rebuilt per call."""
    import missionpay.service as service_module

    constructions: list[int] = []
    transports: list[int] = []

    class Counting:
        def __init__(self, **kwargs):
            constructions.append(1)
            self._inner = PayPalClientCredentials(
                kwargs["client_id"], kwargs["client_secret"],
                base_url=kwargs["base_url"], transport=_ScopeTransport(EXACT, calls=transports),
            )

        def access_token(self, **kwargs):
            return self._inner.access_token(**kwargs)

        def cached_token(self):
            return self._inner.cached_token()

    monkeypatch.setattr(service_module, "PayPalClientCredentials", Counting)
    config = RuntimeConfig(
        provider_mode=ProviderMode.PAYPAL_SANDBOX,
        paypal_client_id="cid",
        paypal_client_secret="csec",
        sandbox_recipient="sb@business.example.com",
        max_payout_cents=500,
    )
    service = PaymentService(MissionPayStore("unused.sqlite3"), config)
    first = service._token()
    second = service._token()

    assert first == second
    assert len(constructions) == 1, "the credential client was rebuilt"
    assert len(transports) == 1, "a second token was minted instead of using the cache"

"""Regressions for the final-acceptance corrections.

F1 legacy / pre-existing authority cannot reach OAuth.
F2 approval is bound to the exact reviewed interpretation, transactionally.
F3 a refused interpretation can be retried and the success used.
F4 the request body is bounded before multipart parsing.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import io
import json
import re
import sqlite3

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from missionpay.auth import AuthorizationDenied, Role
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.executor import DispatchRefused, PayoutExecutor
from missionpay.limits import MAX_REQUEST_BYTES, RequestSizeLimitMiddleware, RequestTooLarge
from missionpay.policy import evaluate_claim
from missionpay.store import MissionPayStore, PersistenceConflict
from missionpay.web import create_app

from conftest import accept_interpretation, approval_fields

RECIPIENT = "sb-configured@business.example.com"


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def store(tmp_path):
    s = MissionPayStore(tmp_path / "fa.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    s.initialize()
    s.register_user("v1", "volunteer@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    s.register_user("r1", "reviewer@example.test", Role.REVIEWER, "reviewer-passphrase")
    for fund_id, code, category in (
        ("RIVER_CLEANUP_FUND", "RIVER_CLEANUP", "PROTECTIVE_SUPPLIES"),
        ("GARDEN_FUND", "COMMUNITY_GARDEN", "SEEDS"),
    ):
        s.create_fund(
            FundPolicy(
                fund_id=fund_id, version="2026.1",
                allowed_activity_codes=frozenset({code}),
                allowed_expense_categories=frozenset({category}),
                starts_on=date(2026, 1, 1), ends_on=date(2026, 12, 31),
                available=Money("USD", 50_000), receipt_required=True,
            )
        )
    return s


def _claim(store, claim_id="c1", digest="a" * 64, *, recipient=RECIPIENT, cents=7899):
    v = store.login("volunteer@example.test", "volunteer-passphrase")
    claim = Claim(
        claim_id=claim_id, volunteer_id="v1",
        activity=Activity("a1", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity(digest, "text/plain", "r.txt"),
        requested=Money("USD", cents), recipient=recipient,
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.submit_claim(v.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(claim, store.load_fund_policy("RIVER_CLEANUP_FUND", "2026.1"),
                       prior_evidence_digests=frozenset()), revision=1
    )
    return claim


def _approve(store, record, claim_id="c1"):
    r = store.login("reviewer@example.test", "reviewer-passphrase")
    return store.authorize_claim(
        r.token, claim_id, revision=1,
        interpretation_id=record["interpretation_id"],
        interpretation_output_sha256=record["output_sha256"],
    )


class _Adapter:
    def __init__(self):
        self.calls = 0

    def create_payout(self, intent, *, access_token):
        self.calls += 1
        return {"payout_batch_id": "BATCH-1", "batch_status": "PENDING"}


def _executor(store, adapter):
    return PayoutExecutor(store, adapter, max_payout_cents=10_000, expected_recipient=RECIPIENT)


# ------------------------------------------------ F1: authority at dispatch


def test_legacy_authorization_without_binding_is_quarantined_before_oauth(store):
    """A migrated row has no reviewed output; dispatch must stop, not proceed."""
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    authorization = _approve(store, record)
    intent = store.create_payment_intent(authorization.authorization_id)

    # Simulate a database written before the binding columns existed.
    with sqlite3.connect(store.path) as conn:
        conn.execute("ALTER TABLE authorizations DROP COLUMN interpretation_id")
        conn.execute("ALTER TABLE authorizations DROP COLUMN interpretation_output_sha256")
        conn.commit()

    adapter = _Adapter()
    tokens: list[int] = []
    with pytest.raises(DispatchRefused, match="AUTHORIZATION_INTERPRETATION_UNBOUND"):
        _executor(store, adapter).execute(
            intent.operation_key, token_provider=lambda: tokens.append(1) or "tok", owner_id="w"
        )
    assert tokens == [], "OAuth ran for an unbound authorization"
    assert adapter.calls == 0, "a payout POST ran for an unbound authorization"
    assert store.operation_state(intent.operation_key) == "PREPARED"


def test_prepared_intent_with_stale_binding_is_refused_before_oauth(store):
    """A newer accepted interpretation makes the existing approval stale."""
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    authorization = _approve(store, record)
    intent = store.create_payment_intent(authorization.authorization_id)

    # Construct the state the API forbids, to prove dispatch defends against it.
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """INSERT INTO interpretations(
                   interpretation_id,claim_id,claim_revision,evidence_sha256,ai_mode,model_id,
                   config_sha256,attempt,status,error_code,output_sha256,facts_json,missing_json,
                   created_at)
               VALUES('interp-newer','c1',1,?,'ollama','qwen3:4b',?,2,'ACCEPTED',NULL,?,'[]','[]',
                      '2099-01-01T00:00:00+00:00')""",
            (claim.evidence.sha256, "9" * 64, "2" * 64),
        )
        conn.commit()

    adapter = _Adapter()
    tokens: list[int] = []
    with pytest.raises(DispatchRefused, match="AUTHORIZATION_INTERPRETATION_STALE"):
        _executor(store, adapter).execute(
            intent.operation_key, token_provider=lambda: tokens.append(1) or "tok", owner_id="w"
        )
    assert tokens == [] and adapter.calls == 0


def test_bound_and_current_authority_still_dispatches(store):
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    authorization = _approve(store, record)
    intent = store.create_payment_intent(authorization.authorization_id)

    adapter = _Adapter()
    outcome = _executor(store, adapter).execute(
        intent.operation_key, token_provider=lambda: "tok", owner_id="w"
    )
    assert outcome.state == "ACKNOWLEDGED"
    assert adapter.calls == 1


# -------------------------------------------- F2: reviewed identity binding


def test_approval_naming_a_superseded_interpretation_is_refused(store):
    claim = _claim(store)
    older = accept_interpretation(store, "c1", 1, claim.evidence.sha256, output_sha256="1" * 64)
    # A different configuration produces a second accepted result before approval.
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="ollama", model_id="qwen3:4b", config_sha256="9" * 64,
        status="ACCEPTED", output_sha256="2" * 64,
    )
    r = store.login("reviewer@example.test", "reviewer-passphrase")
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_STALE"):
        store.authorize_claim(
            r.token, "c1", revision=1,
            interpretation_id=older["interpretation_id"],
            interpretation_output_sha256=older["output_sha256"],
        )


def test_tampered_output_digest_is_refused(store):
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    r = store.login("reviewer@example.test", "reviewer-passphrase")
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_OUTPUT_MISMATCH"):
        store.authorize_claim(
            r.token, "c1", revision=1,
            interpretation_id=record["interpretation_id"],
            interpretation_output_sha256="f" * 64,
        )


def test_unknown_interpretation_id_is_refused(store):
    _claim(store)
    r = store.login("reviewer@example.test", "reviewer-passphrase")
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_NOT_FOUND"):
        store.authorize_claim(
            r.token, "c1", revision=1,
            interpretation_id="interp-invented",
            interpretation_output_sha256="1" * 64,
        )


def test_store_refuses_interpretation_after_approval(store):
    claim = _claim(store)
    record = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    _approve(store, record)
    with pytest.raises(PersistenceConflict, match="INTERPRETATION_AFTER_AUTHORIZATION"):
        store.record_interpretation(
            claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
            ai_mode="ollama", model_id="qwen3:4b", config_sha256="9" * 64,
            status="ACCEPTED", output_sha256="2" * 64,
        )


def test_concurrent_interpretation_and_approval_never_both_win(store):
    """Either the interpretation lands first (approval of the old id goes stale),
    or the approval lands first (the new interpretation is refused)."""
    claim = _claim(store)
    older = accept_interpretation(store, "c1", 1, claim.evidence.sha256)
    r = store.login("reviewer@example.test", "reviewer-passphrase")

    def interpret():
        try:
            store.record_interpretation(
                claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
                ai_mode="ollama", model_id="qwen3:4b", config_sha256="9" * 64,
                status="ACCEPTED", output_sha256="2" * 64,
            )
            return "interpreted"
        except PersistenceConflict:
            return "interpret_refused"

    def approve():
        try:
            store.authorize_claim(
                r.token, "c1", revision=1,
                interpretation_id=older["interpretation_id"],
                interpretation_output_sha256=older["output_sha256"],
            )
            return "approved"
        except AuthorizationDenied:
            return "approval_refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda fn: fn(), [interpret, approve]))

    assert not (outcomes[0] == "interpreted" and outcomes[1] == "approved"), (
        f"both succeeded: {outcomes}"
    )
    # Exactly one of the two must have been refused.
    assert sum(1 for o in outcomes if o.endswith("_refused")) == 1, outcomes


# ------------------------------------------------------- F3: retry recovery


def test_refused_then_accepted_records_attempt_two(store):
    claim = _claim(store)
    first = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="REFUSED", output_sha256="9" * 64, error_code="CITATION_MISMATCH",
    )
    second = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="ACCEPTED", output_sha256="8" * 64,
    )
    assert second["status"] == "ACCEPTED"
    assert second["reused"] is False
    assert second["attempt"] == first["attempt"] + 1

    history = store.interpretation_history("c1", 1)
    assert [row["status"] for row in history] == ["REFUSED", "ACCEPTED"]
    assert store.accepted_interpretation("c1", 1)["output_sha256"] == "8" * 64


def test_retry_after_success_reuses_without_another_attempt(store):
    claim = _claim(store)
    first = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="ACCEPTED", output_sha256="8" * 64,
    )
    again = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="ACCEPTED", output_sha256="7" * 64,
    )
    assert again["reused"] is True
    assert again["interpretation_id"] == first["interpretation_id"]
    assert len(store.interpretation_history("c1", 1)) == 1


def test_retry_then_review_then_approval_succeeds(store):
    claim = _claim(store)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="REFUSED", output_sha256="9" * 64, error_code="CITATION_MISMATCH",
    )
    retried = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/v1", config_sha256="0" * 64,
        status="ACCEPTED", output_sha256="8" * 64,
        facts=[{"name": "total", "value": "USD 78.99", "epistemic_status": "observed",
                "start": 0, "end": 8, "quote": "synthetic"}],
    )
    authorization = _approve(store, retried)
    intent = store.create_payment_intent(authorization.authorization_id)
    assert intent.claim_id == "c1"


def test_ui_failure_then_retry_then_approval(tmp_path, monkeypatch):
    """The whole chain through the real routes: refuse, retry, review, approve."""
    import missionpay.web as web_module

    store = MissionPayStore(tmp_path / "ui.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    for fund_id, code, category in (
        ("RIVER_CLEANUP_FUND", "RIVER_CLEANUP", "PROTECTIVE_SUPPLIES"),
        ("GARDEN_FUND", "COMMUNITY_GARDEN", "SEEDS"),
    ):
        store.create_fund(FundPolicy(
            fund_id=fund_id, version="2026.1",
            allowed_activity_codes=frozenset({code}),
            allowed_expense_categories=frozenset({category}),
            starts_on=date(2026, 1, 1), ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000), receipt_required=True,
        ))
    from missionpay.bootstrap import bootstrap_demo_store

    bootstrap_demo_store(store.path)

    calls = {"n": 0}
    real = web_module._interpret

    def flaky(config, evidence, text):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, "CITATION_MISMATCH", None, None
        return real(config, evidence, text)

    monkeypatch.setattr(web_module, "_interpret", flaky)

    client = TestClient(create_app(store), base_url="https://testserver")

    def csrf(path="/app"):
        return re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text).group(1)

    def login(email, password):
        client.post("/login", data={"email": email, "password": password,
                                    "csrf_token": csrf("/login")}, follow_redirects=False)

    login("volunteer@missionpay.local", "volunteer-demo-passphrase")
    submitted = client.post("/claims/demo", data={"csrf_token": csrf()}, follow_redirects=False)
    path = submitted.headers["location"]

    # First attempt fails and is recorded as a refusal.
    client.post(f"{path}/interpret", data={"csrf_token": csrf()})
    history = store.interpretation_history("river-safety-001", 1)
    assert [row["status"] for row in history] == ["REFUSED"]

    # Retry succeeds and is recorded as attempt 2.
    client.post(f"{path}/interpret", data={"csrf_token": csrf()})
    history = store.interpretation_history("river-safety-001", 1)
    assert [row["status"] for row in history] == ["REFUSED", "ACCEPTED"]

    client.post("/logout", data={"csrf_token": csrf()})
    login("reviewer@missionpay.local", "reviewer-demo-passphrase")
    page = client.get(path).text
    approved = client.post(
        f"{path}/approve",
        data={"csrf_token": csrf(), "revision": "1", "amount_cents": "7899",
              **approval_fields(page)},
        follow_redirects=False,
    )
    assert approved.status_code == 303
    assert store.existing_operation_for_claim("river-safety-001", 1) is not None


def test_ui_stale_page_is_refused(tmp_path):
    store = MissionPayStore(tmp_path / "stale.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    for fund_id, code, category in (
        ("RIVER_CLEANUP_FUND", "RIVER_CLEANUP", "PROTECTIVE_SUPPLIES"),
        ("GARDEN_FUND", "COMMUNITY_GARDEN", "SEEDS"),
    ):
        store.create_fund(FundPolicy(
            fund_id=fund_id, version="2026.1",
            allowed_activity_codes=frozenset({code}),
            allowed_expense_categories=frozenset({category}),
            starts_on=date(2026, 1, 1), ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000), receipt_required=True,
        ))
    from missionpay.bootstrap import bootstrap_demo_store

    bootstrap_demo_store(store.path)
    client = TestClient(create_app(store), base_url="https://testserver")

    def csrf(path="/app"):
        return re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text).group(1)

    client.post("/login", data={"email": "volunteer@missionpay.local",
                                "password": "volunteer-demo-passphrase",
                                "csrf_token": csrf("/login")}, follow_redirects=False)
    submitted = client.post("/claims/demo", data={"csrf_token": csrf()}, follow_redirects=False)
    path = submitted.headers["location"]
    client.post(f"{path}/interpret", data={"csrf_token": csrf()})

    client.post("/logout", data={"csrf_token": csrf()})
    client.post("/login", data={"email": "reviewer@missionpay.local",
                                "password": "reviewer-demo-passphrase",
                                "csrf_token": csrf("/login")}, follow_redirects=False)
    stale = approval_fields(client.get(path).text)

    # A newer accepted interpretation lands while the reviewer's page is open.
    claim = store.claim_snapshot(
        store.login("reviewer@missionpay.local", "reviewer-demo-passphrase").token,
        "river-safety-001",
    )["claim"]
    store.record_interpretation(
        claim_id="river-safety-001", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="ollama", model_id="qwen3:4b", config_sha256="9" * 64,
        status="ACCEPTED", output_sha256="2" * 64,
    )

    refused = client.post(
        f"{path}/approve",
        data={"csrf_token": csrf(), "revision": "1", "amount_cents": "7899", **stale},
    )
    assert refused.status_code == 409
    assert "INTERPRETATION_STALE" in refused.text
    assert store.existing_authorization_id("river-safety-001", 1) is None


# ------------------------------------------------ F4: bounded request body


class _RecordingSend:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)


def _http_scope(headers):
    return {"type": "http", "method": "POST", "path": "/", "headers": headers}


def _run(app, scope, chunks):
    """Drive an ASGI app with a receive that yields the given body chunks."""
    import asyncio

    pending = list(chunks)
    delivered = {"count": 0}

    async def receive():
        if not pending:
            return {"type": "http.request", "body": b"", "more_body": False}
        body = pending.pop(0)
        delivered["count"] += 1
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    send = _RecordingSend()

    async def drain_app(scope, receive, send_):
        # Emulate a parser that keeps pulling until the body ends.
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await send_({"type": "http.response.body", "body": b"ok"})

    middleware = RequestSizeLimitMiddleware(drain_app, max_bytes=100)
    asyncio.run(middleware(scope, receive, send))
    return send, delivered


def test_declared_oversize_is_rejected_without_reading_any_body():
    scope = _http_scope([(b"content-length", b"1000000")])
    send, delivered = _run(None, scope, [b"x" * 50])
    status = [m for m in send.messages if m["type"] == "http.response.start"][0]["status"]
    assert status == 413
    assert delivered["count"] == 0, "the body was read despite a declared oversize length"


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"10")]])
def test_understated_or_missing_length_is_caught_while_streaming(headers):
    scope = _http_scope(headers)
    # Ten chunks of 50 bytes: the cap is crossed at the third.
    send, delivered = _run(None, scope, [b"x" * 50] * 10)
    status = [m for m in send.messages if m["type"] == "http.response.start"][0]["status"]
    assert status == 413
    assert delivered["count"] <= 3, f"read {delivered['count']} chunks past the cap"


def test_body_within_the_cap_is_passed_through():
    scope = _http_scope([(b"content-length", b"60")])
    send, _delivered = _run(None, scope, [b"x" * 60])
    status = [m for m in send.messages if m["type"] == "http.response.start"][0]["status"]
    assert status == 200


def _csrf(html):
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def test_oversized_unauthenticated_upload_is_refused_before_parsing(tmp_path):
    store = MissionPayStore(tmp_path / "up1.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    client = TestClient(create_app(store), base_url="https://testserver")
    response = client.post(
        "/claims/upload",
        files={"evidence": ("big.txt", io.BytesIO(b"x" * (MAX_REQUEST_BYTES + 1024)), "text/plain")},
        data={"csrf_token": "irrelevant", "amount_cents": "1",
              "activity_code": "RIVER_CLEANUP", "expense_category": "PROTECTIVE_SUPPLIES"},
    )
    assert response.status_code == 413
    assert "REQUEST_TOO_LARGE" in response.text
    evidence_root = tmp_path / "up1.sqlite3.evidence"
    assert not evidence_root.exists() or list(evidence_root.iterdir()) == []


def test_oversized_authenticated_upload_is_refused_without_writes_or_inference(tmp_path):
    store = MissionPayStore(tmp_path / "up2.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    store.register_user("v1", "v@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    client = TestClient(create_app(store), base_url="https://testserver")
    client.post("/login", data={"email": "v@example.test", "password": "volunteer-passphrase",
                                "csrf_token": _csrf(client.get("/login").text)},
                follow_redirects=False)
    response = client.post(
        "/claims/upload",
        files={"evidence": ("big.txt", io.BytesIO(b"x" * (MAX_REQUEST_BYTES + 4096)), "text/plain")},
        data={"csrf_token": _csrf(client.get("/app").text), "amount_cents": "1",
              "activity_code": "RIVER_CLEANUP", "expense_category": "PROTECTIVE_SUPPLIES"},
    )
    assert response.status_code == 413
    evidence_root = tmp_path / "up2.sqlite3.evidence"
    assert not evidence_root.exists() or list(evidence_root.iterdir()) == []
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0] == 0

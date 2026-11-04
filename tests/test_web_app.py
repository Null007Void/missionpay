import re
from datetime import date

from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from missionpay.auth import Role
from missionpay.bootstrap import bootstrap_demo_store
from missionpay.domain import FundPolicy, Money
from missionpay.store import MissionPayStore
from missionpay.web import create_app

from conftest import approval_fields


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _store(tmp_path):
    store = MissionPayStore(tmp_path / "web.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    store.register_user("volunteer-1", "river.volunteer@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    store.register_user("reviewer-1", "reviewer@example.test", Role.REVIEWER, "reviewer-passphrase")
    store.register_user("volunteer-2", "other@example.test", Role.VOLUNTEER, "other-volunteer-passphrase")
    store.create_fund(
        FundPolicy(
            fund_id="GARDEN_FUND",
            version="2026.1",
            allowed_activity_codes=frozenset({"COMMUNITY_GARDEN"}),
            allowed_expense_categories=frozenset({"SEEDS"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        )
    )
    store.create_fund(
        FundPolicy(
            fund_id="RIVER_CLEANUP_FUND",
            version="2026.1",
            allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
            allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        )
    )
    return store


def _login(client, email, password):
    page = client.get("/login")
    response = client.post(
        "/login",
        data={"email": email, "password": password, "csrf_token": _csrf(page.text)},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _current_csrf(client):
    return _csrf(client.get("/app").text)


def test_login_uses_secure_server_side_session_and_csrf(tmp_path):
    store = _store(tmp_path)
    client = TestClient(create_app(store), base_url="https://testserver")

    login = client.get("/login")
    response = client.post(
        "/login",
        data={
            "email": "river.volunteer@example.test",
            "password": "volunteer-passphrase",
            "csrf_token": _csrf(login.text),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    cookies = response.headers.get_list("set-cookie")
    session_cookie = next(value for value in cookies if value.startswith("mp_session="))
    assert "HttpOnly" in session_cookie
    assert "Secure" in session_cookie
    assert "SameSite=strict" in session_cookie

    dashboard = client.get("/app")
    assert dashboard.status_code == 200
    assert "Volunteer workspace" in dashboard.text

    refused = client.post("/claims/demo", data={"csrf_token": "wrong"})
    assert refused.status_code == 403


def test_demo_bootstrap_is_idempotent_and_creates_runnable_local_accounts(tmp_path):
    path = tmp_path / "demo.sqlite3"

    first = bootstrap_demo_store(path)
    second = bootstrap_demo_store(path)

    assert first.user_email("demo-volunteer") == "volunteer@missionpay.local"
    assert second.login("reviewer@missionpay.local", "reviewer-demo-passphrase").role is Role.REVIEWER


def test_authenticated_offline_vertical_slice_enforces_ownership_and_frozen_approval(tmp_path):
    store = _store(tmp_path)
    volunteer = TestClient(create_app(store), base_url="https://testserver")
    _login(volunteer, "river.volunteer@example.test", "volunteer-passphrase")

    submitted = volunteer.post(
        "/claims/demo",
        data={"csrf_token": _current_csrf(volunteer)},
        follow_redirects=False,
    )
    assert submitted.status_code == 303
    assert submitted.headers["location"] == "/claims/river-safety-001"

    # The AI stage is an explicit, persisted action, not a side effect of rendering.
    before = volunteer.get(submitted.headers["location"])
    assert "NOT RUN — THE AI STAGE IS INCOMPLETE" in before.text
    interpreted = volunteer.post(
        "/claims/river-safety-001/interpret",
        data={"csrf_token": _current_csrf(volunteer)},
        follow_redirects=False,
    )
    assert interpreted.status_code == 303

    detail = volunteer.get(submitted.headers["location"])
    assert detail.status_code == 200
    assert "Expense evidence" in detail.text
    assert "AI interpretation" in detail.text
    assert "Observed fact" in detail.text
    assert "Inference" in detail.text
    assert "Uncertainty" in detail.text
    assert "Citation" in detail.text
    assert "Garden Fund" in detail.text and "Refused" in detail.text
    assert "River Cleanup Fund" in detail.text and "Eligible" in detail.text
    assert "$78.99" in detail.text
    assert "SIMULATED INTERPRETATION FIXTURE — NO LIVE MODEL INVOKED" in detail.text

    other = TestClient(create_app(store), base_url="https://testserver")
    _login(other, "other@example.test", "other-volunteer-passphrase")
    assert other.get("/claims/river-safety-001").status_code == 403

    reviewer = TestClient(create_app(store), base_url="https://testserver")
    _login(reviewer, "reviewer@example.test", "reviewer-passphrase")
    queue = reviewer.get("/review")
    assert queue.status_code == 200
    assert "river-safety-001" in queue.text

    tampered = reviewer.post(
        "/claims/river-safety-001/approve",
        data={"csrf_token": _current_csrf(reviewer), "revision": "1", "amount_cents": "1"},
    )
    assert tampered.status_code == 409
    assert "FROZEN_AMOUNT_MISMATCH" in tampered.text

    approved = reviewer.post(
        "/claims/river-safety-001/approve",
        data={
            "csrf_token": _current_csrf(reviewer),
            "revision": "1",
            "amount_cents": "7899",
            **approval_fields(reviewer.get("/claims/river-safety-001").text),
        },
        follow_redirects=False,
    )
    assert approved.status_code == 303

    approved_detail = reviewer.get(approved.headers["location"])
    assert "Human authorization" in approved_detail.text
    assert "Not attempted" in approved_detail.text
    assert "Sending" in approved_detail.text
    assert "Uncertain" in approved_detail.text
    assert "Acknowledged" in approved_detail.text
    assert "Pending" in approved_detail.text
    assert "Independently reconciled" in approved_detail.text
    assert "SIMULATED — NO PAYPAL PAYMENT EXECUTED" in approved_detail.text

    repeated = reviewer.post(
        "/claims/river-safety-001/approve",
        data={"csrf_token": _current_csrf(reviewer), "revision": "1", "amount_cents": "7899"},
    )
    assert repeated.status_code == 409
    assert "DUPLICATE_REIMBURSEMENT_BLOCKED" in repeated.text

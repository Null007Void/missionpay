"""The judge demo is gone from the real app; the genuine controls remain.

This file has two jobs. First, prove the judge-only machinery cannot come back by accident:
it is structural, so the tests fail if a file, a CLI command, a config field or a page banner
reappears. Second, prove the genuine startup and payment controls still refuse what they must,
entirely offline — no provider is reachable from this file.
"""

from __future__ import annotations

import inspect
import pathlib
import socket

import pytest

from missionpay.bootstrap import bootstrap_demo_store
from missionpay.config import ConfigurationError, ProviderMode, RuntimeConfig
from missionpay.paypal import PayPalSandboxAdapter
from missionpay.service import PaymentService, PaymentServiceRefused

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "missionpay"


# ---- the removal is structural ----------------------------------------------------------


def test_the_judge_demo_module_does_not_exist():
    import importlib.util

    assert importlib.util.find_spec("missionpay.judgedemo") is None


def test_no_source_file_mentions_the_judge_demo():
    """Narrow on purpose: "judge" also appears in ordinary prose about judging a result."""
    markers = ("judge_demo", "judge-demo", "judgedemo", "MISSIONPAY_JUDGE_DEMO", "DEMO STORE")
    offenders = [
        path.name
        for path in SRC.glob("*.py")
        if any(marker in path.read_text(encoding="utf-8") for marker in markers)
    ]
    assert offenders == []


def test_the_cli_has_no_demo_reset_command():
    from missionpay import cli

    with pytest.raises(SystemExit):
        cli.main(["demo-reset", "--confirm"])


def test_the_config_has_no_judge_demo_field():
    assert "judge_demo" not in RuntimeConfig.__dataclass_fields__


def test_bootstrap_takes_no_store_kind():
    """The kind existed only so a demo store could be told apart. With no demo, there is none."""
    assert "kind" not in inspect.signature(bootstrap_demo_store).parameters


def test_the_store_has_no_demo_identity_api_at_all(tmp_path):
    """The metadata table, the kind marker and the simulated ledger all belonged to the demo."""
    from missionpay.store import MissionPayStore

    path = tmp_path / "plain.sqlite3"
    bootstrap_demo_store(path)
    store = MissionPayStore(path)
    assert not hasattr(store, "get_metadata")
    assert not hasattr(store, "declare_kind")
    assert not hasattr(store, "record_simulated_payout")
    with store._connection() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "store_metadata" not in tables
    assert "simulated_payouts" not in tables


def test_the_app_page_carries_no_demo_banner(tmp_path):
    """The in-app demo banner belonged to the deleted feature."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from missionpay.store import MissionPayStore
    from missionpay.web import create_app

    path = tmp_path / "app.sqlite3"
    bootstrap_demo_store(path)
    store = MissionPayStore(path)
    client = TestClient(create_app(store), base_url="https://testserver")
    body = client.get("/login").text
    assert "DEMO STORE" not in body
    assert "simulation run" not in body.lower()


# ---- genuine startup refuses what it must -----------------------------------------------


def genuine(**overrides):
    base = dict(
        provider_mode=ProviderMode.PAYPAL_SANDBOX,
        paypal_client_id="sandbox-id",
        paypal_client_secret="sandbox-secret",
        sandbox_recipient="sb-recipient@business.example.com",
        max_payout_cents=10_000,
    )
    base.update(overrides)
    return RuntimeConfig(**base)


def test_a_genuine_configuration_without_credentials_is_refused():
    with pytest.raises(ConfigurationError) as exc:
        RuntimeConfig(provider_mode=ProviderMode.PAYPAL_SANDBOX)
    assert "PAYPAL_CREDENTIALS_REQUIRED" in str(exc.value)


def test_a_genuine_configuration_without_a_recipient_is_refused():
    with pytest.raises(ConfigurationError) as exc:
        genuine(sandbox_recipient=None)
    assert "SANDBOX_RECIPIENT_REQUIRED" in str(exc.value)


def test_a_genuine_configuration_without_a_payout_ceiling_is_refused():
    with pytest.raises(ConfigurationError) as exc:
        genuine(max_payout_cents=None)
    assert "MAX_PAYOUT_CENTS_REQUIRED" in str(exc.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://api-m.paypal.com",
        "http://api-m.sandbox.paypal.com",
        "https://api-m.sandbox.paypal.com.evil.example",
    ],
)
def test_a_genuine_configuration_refuses_a_non_sandbox_endpoint(url):
    with pytest.raises(ConfigurationError) as exc:
        genuine(paypal_base_url=url)
    assert "PAYPAL_SANDBOX_ORIGIN_REQUIRED" in str(exc.value)


def test_a_production_environment_switch_is_refused():
    from missionpay.config import from_environment

    with pytest.raises(ConfigurationError) as exc:
        from_environment(
            {
                "MISSIONPAY_PROVIDER_MODE": "paypal_sandbox",
                "PAYPAL_ENVIRONMENT": "live",
                "PAYPAL_CLIENT_ID": "id",
                "PAYPAL_CLIENT_SECRET": "secret",
                "MISSIONPAY_MAX_PAYOUT_CENTS": "100",
            }
        )
    assert "PAYPAL_PRODUCTION_REFUSED" in str(exc.value)


def test_the_adapter_refuses_a_non_sandbox_origin():
    from missionpay.paypal import SandboxEndpointRequired

    with pytest.raises(SandboxEndpointRequired):
        PayPalSandboxAdapter(base_url="https://api-m.paypal.com")


# ---- payment controls, offline ----------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """A store with the demo users and fund, matching what the money tests use."""
    from datetime import date

    from argon2 import PasswordHasher

    from missionpay.auth import Role
    from missionpay.domain import FundPolicy, Money
    from missionpay.store import MissionPayStore

    fund_id, policy_version = "river-cleanup-fund", "2026-03"

    result = MissionPayStore(
        tmp_path / "missionpay.db",
        password_hasher=PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1),
    )
    result.initialize()
    result.register_user("volunteer-alex", "alex@example.org", Role.VOLUNTEER, "volunteer password")
    result.register_user("reviewer-morgan", "morgan@example.org", Role.REVIEWER, "reviewer password")
    result.create_fund(
        FundPolicy(
            fund_id=fund_id,
            version=policy_version,
            allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
            allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        )
    )
    return result


def test_mock_mode_refuses_execution_before_any_adapter_exists(store):
    service = PaymentService(store, RuntimeConfig())
    with pytest.raises(PaymentServiceRefused) as exc:
        service.execute("missionpay:whatever", owner_id="worker-A")
    assert "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS" in str(exc.value)


def test_mock_mode_refuses_reconciliation(store):
    service = PaymentService(store, RuntimeConfig())
    with pytest.raises(PaymentServiceRefused):
        service.observe("missionpay:whatever")


def test_mock_mode_opens_no_socket(store, monkeypatch):
    """Refusal must happen before anything that could reach a provider."""

    def explode(*args, **kwargs):
        raise AssertionError("the mock path opened a socket")

    monkeypatch.setattr(socket, "socket", explode)
    service = PaymentService(store, RuntimeConfig())
    with pytest.raises(PaymentServiceRefused):
        service.execute("missionpay:whatever", owner_id="worker-A")
    with pytest.raises(PaymentServiceRefused):
        service.observe("missionpay:whatever")


def test_the_payout_cap_is_enforced_before_dispatch(store):
    from missionpay.executor import DispatchRefused, PayoutExecutor
    from test_executor import RecordingPayoutAdapter, _intent

    intent = _intent(store, amount=7_899)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=5_000)

    with pytest.raises(DispatchRefused, match="PAYOUT_CAP_EXCEEDED"):
        executor.execute(intent.operation_key, token_provider=lambda: "bearer", owner_id="worker-A")

    assert adapter.calls == []
    assert store.operation_state(intent.operation_key) == "PREPARED"


def test_the_expected_recipient_is_enforced_before_dispatch(store):
    from missionpay.executor import DispatchRefused, PayoutExecutor
    from test_executor import RecordingPayoutAdapter, _intent

    intent = _intent(store)
    adapter = RecordingPayoutAdapter()
    executor = PayoutExecutor(store, adapter, expected_recipient="someone-else@business.example.com")

    with pytest.raises(DispatchRefused, match="SANDBOX_RECIPIENT_MISMATCH"):
        executor.execute(intent.operation_key, token_provider=lambda: "bearer", owner_id="worker-A")

    assert adapter.calls == []

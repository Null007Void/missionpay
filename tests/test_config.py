"""Configuration is the first safety boundary: it must fail closed.

These tests exercise the real ``from_environment`` mapping and the real
dataclass validation. They assert that a genuine run cannot start half-configured,
that a production origin is unreachable, and that a config object can be logged
without disclosing credential material.
"""

import pytest

from missionpay.config import (
    PAYPAL_SANDBOX_ORIGIN,
    AiMode,
    ConfigurationError,
    ProviderMode,
    RuntimeConfig,
    from_environment,
)

CLIENT_ID = "sandbox-client-id-fixture"
CLIENT_SECRET = "sandbox-client-secret-fixture-do-not-log-4f9c1a"

FULL_SANDBOX = {
    "MISSIONPAY_PROVIDER_MODE": "paypal_sandbox",
    "PAYPAL_CLIENT_ID": CLIENT_ID,
    "PAYPAL_CLIENT_SECRET": CLIENT_SECRET,
    "MISSIONPAY_SANDBOX_RECIPIENT": "sb-recipient@business.example.com",
    "MISSIONPAY_MAX_PAYOUT_CENTS": "500",
}


def _assignment(key: str, value: str) -> str:
    """Assemble a KEY=VALUE line at runtime.

    Written this way so the source file never contains a literal credential-shaped
    string: the public-tree secret scan is deliberately strict and should keep
    flagging such shapes rather than carrying an exception for test fixtures.
    """
    return key + "=" + value


@pytest.fixture(autouse=True)
def _isolate_credentials_file(tmp_path, monkeypatch):
    """Keep these tests hermetic.

    ``from_environment`` also consults the owner's private credential file. Without
    this fixture, a developer shell that exports MISSIONPAY_CREDENTIALS_FILE would
    silently satisfy the "credential missing" cases and the suite would pass or fail
    depending on who ran it.
    """
    monkeypatch.setenv("MISSIONPAY_CREDENTIALS_FILE", str(tmp_path / "no-such-credentials.env"))


def test_ambient_credential_file_cannot_satisfy_a_missing_key(tmp_path, monkeypatch):
    """Regression: an exported credential file must not leak into these assertions."""
    path = tmp_path / "ambient.env"
    path.write_text(
        _assignment("PAYPAL_CLIENT_ID", "ambient-id")
        + "\n"
        + _assignment("PAYPAL_CLIENT_SECRET", "ambient-secret")
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MISSIONPAY_CREDENTIALS_FILE", str(path))

    env = dict(FULL_SANDBOX)
    del env["PAYPAL_CLIENT_SECRET"]
    # The file supplies it, so this is a *valid* configuration by design; the point
    # of the assertion is that the fixture above is what keeps the other tests honest.
    config = from_environment(env)
    assert config.paypal_client_secret == "ambient-secret"


def test_default_is_mock_and_never_touches_a_provider():
    config = from_environment({})
    assert config.provider_mode is ProviderMode.MOCK
    assert config.ai_mode is AiMode.FIXTURE
    assert config.is_genuine is False
    assert config.secure_cookies is True
    assert config.paypal_client_id is None
    assert "MOCK" in config.mode_label


def test_full_sandbox_configuration_is_accepted():
    config = from_environment(FULL_SANDBOX)
    assert config.provider_mode is ProviderMode.PAYPAL_SANDBOX
    assert config.is_genuine is True
    assert config.max_payout_cents == 500
    assert config.paypal_base_url == PAYPAL_SANDBOX_ORIGIN
    assert "SANDBOX" in config.mode_label


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("PAYPAL_CLIENT_ID", "PAYPAL_CREDENTIALS_REQUIRED"),
        ("PAYPAL_CLIENT_SECRET", "PAYPAL_CREDENTIALS_REQUIRED"),
        ("MISSIONPAY_SANDBOX_RECIPIENT", "SANDBOX_RECIPIENT_REQUIRED"),
        ("MISSIONPAY_MAX_PAYOUT_CENTS", "MAX_PAYOUT_CENTS_REQUIRED"),
    ],
)
def test_genuine_mode_refuses_to_start_incomplete(missing, expected):
    env = dict(FULL_SANDBOX)
    del env[missing]
    with pytest.raises(ConfigurationError, match=expected):
        from_environment(env)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api-m.paypal.com",
        "https://api.paypal.com",
        "http://api-m.sandbox.paypal.com",
        "https://api-m.sandbox.paypal.com.attacker.example",
    ],
)
def test_production_or_lookalike_origins_are_refused(base_url):
    env = dict(FULL_SANDBOX, PAYPAL_BASE_URL=base_url)
    with pytest.raises(ConfigurationError, match="PAYPAL_SANDBOX_ORIGIN_REQUIRED"):
        from_environment(env)


@pytest.mark.parametrize("switch", ["PAYPAL_ENVIRONMENT", "PAYPAL_MODE"])
def test_live_switch_is_a_hard_refusal(switch):
    env = dict(FULL_SANDBOX, **{switch: "live"})
    with pytest.raises(ConfigurationError, match="PAYPAL_PRODUCTION_REFUSED"):
        from_environment(env)


def test_unknown_modes_are_refused_not_defaulted():
    with pytest.raises(ConfigurationError, match="UNKNOWN_PROVIDER_MODE"):
        from_environment({"MISSIONPAY_PROVIDER_MODE": "live"})
    with pytest.raises(ConfigurationError, match="UNKNOWN_AI_MODE"):
        from_environment({"MISSIONPAY_AI_MODE": "cloud"})


def test_ollama_mode_requires_a_local_explicit_model():
    with pytest.raises(ConfigurationError, match="OLLAMA_MODEL_REQUIRED"):
        from_environment({"MISSIONPAY_AI_MODE": "ollama"})
    with pytest.raises(ConfigurationError, match="CLOUD_MODEL_REFUSED"):
        from_environment(
            {"MISSIONPAY_AI_MODE": "ollama", "MISSIONPAY_OLLAMA_MODEL": "qwen3:4b:cloud"}
        )
    with pytest.raises(ConfigurationError, match="OLLAMA_LOOPBACK_REQUIRED"):
        from_environment(
            {
                "MISSIONPAY_AI_MODE": "ollama",
                "MISSIONPAY_OLLAMA_MODEL": "qwen3:4b",
                "MISSIONPAY_OLLAMA_URL": "https://ollama.example.com",
            }
        )


def test_ollama_mode_accepts_loopback():
    config = from_environment(
        {
            "MISSIONPAY_AI_MODE": "ollama",
            "MISSIONPAY_OLLAMA_MODEL": "qwen3:4b",
            "MISSIONPAY_OLLAMA_URL": "http://127.0.0.1:11434",
        }
    )
    assert config.ai_mode is AiMode.OLLAMA
    assert config.ollama_model == "qwen3:4b"


def test_cap_must_be_a_positive_integer():
    for bad in ("0", "-100", "abc", "5.5"):
        env = dict(FULL_SANDBOX, MISSIONPAY_MAX_PAYOUT_CENTS=bad)
        with pytest.raises(ConfigurationError):
            from_environment(env)


def test_secrets_are_absent_from_repr_and_description():
    config = from_environment(FULL_SANDBOX)
    rendered = repr(config)
    described = repr(config.describe())

    assert CLIENT_SECRET not in rendered
    assert CLIENT_SECRET not in described
    assert CLIENT_ID not in rendered
    assert config.describe()["paypal_credentials_present"] is True


def test_insecure_cookies_is_opt_in_only():
    assert from_environment({}).secure_cookies is True
    assert (
        from_environment({"MISSIONPAY_ALLOW_INSECURE_LOCALHOST": "1"}).secure_cookies is False
    )


def test_direct_construction_validates_the_same_way():
    with pytest.raises(ConfigurationError, match="PAYPAL_CREDENTIALS_REQUIRED"):
        RuntimeConfig(provider_mode=ProviderMode.PAYPAL_SANDBOX)

"""Credential entry must never put a secret anywhere it can be read back.

Every value used in these tests is synthetic. The assertions that matter are the
negative ones: that no reporting path, error message or repr contains a value.
"""

from pathlib import Path

import pytest

from missionpay.credentials import (
    CREDENTIALS_FILE_ENV,
    CredentialError,
    default_credentials_path,
    parse,
    read_file,
    repo_root,
    resolve,
    status,
    write_template,
)

SECRET = "synthetic-secret-value-8b21c4"
CLIENT_ID = "synthetic-client-id-77aa"


def _fill(path: Path, *, client_id: str = CLIENT_ID, secret: str = SECRET) -> Path:
    text = (
        "# comment line, ignored\n"
        "\n"
        f"PAYPAL_CLIENT_ID={client_id}\n"
        f"PAYPAL_CLIENT_SECRET={secret}\n"
        "MISSIONPAY_MAX_PAYOUT_CENTS=500\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------- parsing


def test_parse_ignores_comments_and_blanks():
    values = parse("# c\n\nPAYPAL_CLIENT_ID=abc\n")
    assert values == {"PAYPAL_CLIENT_ID": "abc"}


def test_parse_strips_surrounding_quotes():
    assert parse('PAYPAL_CLIENT_ID="abc"\n')["PAYPAL_CLIENT_ID"] == "abc"
    assert parse("PAYPAL_CLIENT_ID='abc'\n")["PAYPAL_CLIENT_ID"] == "abc"


def test_parse_keeps_equals_signs_inside_a_value():
    assert parse("PAYPAL_CLIENT_SECRET=a=b=c\n")["PAYPAL_CLIENT_SECRET"] == "a=b=c"


def test_parse_rejects_unknown_keys_without_echoing_them():
    with pytest.raises(CredentialError, match=r"CREDENTIALS_UNKNOWN_KEY:line=1"):
        parse("PAYPAL_SECRET=oops\n")


def test_parse_rejects_a_malformed_line_by_number_only():
    with pytest.raises(CredentialError, match=r"CREDENTIALS_LINE_MALFORMED:line=2"):
        parse("PAYPAL_CLIENT_ID=abc\nnot-a-pair\n")


def test_unknown_key_cannot_leak_a_secret_pasted_before_equals():
    """Regression for review finding F3.

    A user who pastes ``<secret>=value`` into the wrong shape must not have the
    secret echoed back in an error message, a log line, or a page.
    """
    secret = "synthetic-secret-before-equals"
    with pytest.raises(CredentialError) as excinfo:
        parse(f"{secret}=value\n")
    message = str(excinfo.value)
    assert secret not in message
    assert "value" not in message.replace("CREDENTIALS_UNKNOWN_KEY", "")
    assert message == "CREDENTIALS_UNKNOWN_KEY:line=1"


def test_parse_error_never_echoes_a_value_after_the_equals():
    with pytest.raises(CredentialError) as excinfo:
        parse(f"UNKNOWN_KEY={SECRET}\n")
    assert SECRET not in str(excinfo.value)


# ------------------------------------------------------------ file rules


def test_template_is_created_and_refuses_overwrite(tmp_path):
    target = tmp_path / "creds" / "credentials.env"
    written = write_template(target)
    assert written == target
    assert "PAYPAL_CLIENT_ID=" in target.read_text(encoding="utf-8")

    with pytest.raises(CredentialError, match="CREDENTIALS_FILE_EXISTS"):
        write_template(target)


def test_template_inside_the_repository_is_refused(tmp_path):
    inside = repo_root() / "credentials.env"
    with pytest.raises(CredentialError, match="CREDENTIALS_FILE_MUST_BE_OUTSIDE_REPOSITORY"):
        write_template(inside)


def test_reading_a_file_inside_the_repository_is_refused():
    with pytest.raises(CredentialError, match="CREDENTIALS_FILE_MUST_BE_OUTSIDE_REPOSITORY"):
        read_file(repo_root() / "credentials.env")


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert read_file(tmp_path / "absent.env") == {}


# ------------------------------------------------------------- precedence


def test_environment_wins_over_the_file(tmp_path):
    path = _fill(tmp_path / "c.env")
    merged = resolve({"PAYPAL_CLIENT_SECRET": "from-environment"}, path)
    assert merged["PAYPAL_CLIENT_SECRET"] == "from-environment"
    assert merged["PAYPAL_CLIENT_ID"] == CLIENT_ID


def test_blank_environment_value_does_not_override_the_file(tmp_path):
    path = _fill(tmp_path / "c.env")
    merged = resolve({"PAYPAL_CLIENT_SECRET": "   "}, path)
    assert merged["PAYPAL_CLIENT_SECRET"] == SECRET


def test_default_path_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv(CREDENTIALS_FILE_ENV, raising=False)
    path = default_credentials_path()
    with pytest.raises(ValueError):
        path.resolve().relative_to(repo_root())


# ----------------------------------------------------------------- status


def test_status_reports_presence_only_and_leaks_nothing(tmp_path):
    path = _fill(tmp_path / "c.env")
    report = status(path)
    described = str(report.describe())

    assert report.exists is True
    assert report.readable is True
    assert report.complete is True
    assert set(report.keys_present) == {"PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET"}

    assert SECRET not in described
    assert CLIENT_ID not in described
    assert SECRET not in repr(report)


def test_status_names_missing_keys(tmp_path):
    path = tmp_path / "c.env"
    path.write_text("PAYPAL_CLIENT_ID=only-id\n", encoding="utf-8")
    report = status(path)

    assert report.complete is False
    assert report.keys_missing == ("PAYPAL_CLIENT_SECRET",)
    assert "only-id" not in str(report.describe())


def test_status_of_absent_file_is_not_an_error(tmp_path):
    report = status(tmp_path / "absent.env")
    assert report.exists is False
    assert report.complete is False
    assert report.permission_warning is None


# ------------------------------------------------------- config integration


def test_config_uses_the_credential_file_for_genuine_mode(tmp_path, monkeypatch):
    from missionpay.config import ProviderMode, from_environment

    path = _fill(tmp_path / "c.env")
    monkeypatch.setenv(CREDENTIALS_FILE_ENV, str(path))

    config = from_environment(
        {
            "MISSIONPAY_PROVIDER_MODE": "paypal_sandbox",
            "MISSIONPAY_SANDBOX_RECIPIENT": "sb@business.example.com",
        }
    )
    assert config.provider_mode is ProviderMode.PAYPAL_SANDBOX
    assert config.paypal_client_id == CLIENT_ID
    assert config.paypal_client_secret == SECRET
    assert config.max_payout_cents == 500
    assert SECRET not in repr(config)
    assert SECRET not in repr(config.describe())


def test_mock_mode_still_starts_with_a_malformed_credential_file(tmp_path, monkeypatch):
    from missionpay.config import ProviderMode, from_environment

    path = tmp_path / "bad.env"
    path.write_text("NOT_A_KEY=value\n", encoding="utf-8")
    monkeypatch.setenv(CREDENTIALS_FILE_ENV, str(path))

    config = from_environment({})
    assert config.provider_mode is ProviderMode.MOCK

    with pytest.raises(CredentialError):
        from_environment({"MISSIONPAY_PROVIDER_MODE": "paypal_sandbox"})


def test_genuine_mode_without_file_or_env_still_fails_closed(monkeypatch, tmp_path):
    from missionpay.config import ConfigurationError, from_environment

    monkeypatch.setenv(CREDENTIALS_FILE_ENV, str(tmp_path / "absent.env"))
    with pytest.raises(ConfigurationError, match="PAYPAL_CREDENTIALS_REQUIRED"):
        from_environment({"MISSIONPAY_PROVIDER_MODE": "paypal_sandbox"})
